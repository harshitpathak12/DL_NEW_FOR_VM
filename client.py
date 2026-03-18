#!/usr/bin/env python3
"""
Simple WebSocket client for the driving license detection server.

Runs on a local computer with a webcam:
- Captures frames from the default camera
- Encodes each frame as JPEG
- Sends JPEG bytes to the server over WebSocket
- Receives annotated JPEG frames from the server and displays them

This matches the server endpoint in `driver_license_detection/main.py`:
- WebSocket URL: ws://<server_ip>:5001/stream?qwen=1

When `qwen=1` is set on the URL, the server sends:
- First: JSON with detections (includes `dl_numbers`, `validation_label`, etc.)
- Second: Annotated JPEG frame.
"""

import asyncio
import json
import time
from typing import Optional

import cv2
import numpy as np
import websockets


SERVER_WS_URL = "ws://127.0.0.1:5001/stream?qwen=1"
JPEG_QUALITY = 75
CAP_WIDTH, CAP_HEIGHT = 640, 480
TARGET_FPS = 5  # limit client send rate to reduce load


async def stream_to_server(ws_url: str) -> None:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)

    if not cap.isOpened():
        print("Error: cannot open webcam")
        return

    try:
        async with websockets.connect(
            ws_url,
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
        ) as ws:
            print(f"Connected to {ws_url}")
            loop = asyncio.get_running_loop()

            while True:
                # Read frame from webcam in executor to avoid blocking event loop
                ret, frame = await loop.run_in_executor(None, cap.read)
                if not ret:
                    print("Failed to read frame from webcam")
                    break

                ok, buffer = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                )
                if not ok:
                    continue

                await ws.send(buffer.tobytes())

                try:
                    # With qwen=1, server sends:
                    # 1) JSON text with detections
                    # 2) Annotated JPEG bytes
                    msg1 = await ws.recv()
                    msg2: Optional[bytes | str] = None

                    if isinstance(msg1, str):
                        json_str = msg1
                        msg2 = await ws.recv()
                        image_bytes = msg2 if isinstance(msg2, (bytes, bytearray)) else None
                    elif isinstance(msg1, (bytes, bytearray)):
                        # No JSON (qwen=0 case) → msg1 is image bytes
                        json_str = None
                        image_bytes = msg1
                    else:
                        print("Unexpected message type:", type(msg1))
                        continue

                    if json_str:
                        try:
                            info = json.loads(json_str)
                            for i, d in enumerate(info.get("detections", [])):
                                dl_nums = d.get("dl_numbers") or []
                                print(
                                    f"Det {i+1}: conf={d.get('confidence',0):.2f} | "
                                    f"OCR: {d.get('ocr_text','')[:80]} | "
                                    f"Val: {d.get('validation_label','')} | "
                                    f"DL: {dl_nums[0] if dl_nums else '-'}"
                                )
                        except Exception as e:
                            print("Failed to parse JSON from server:", e)

                    if image_bytes:
                        annotated_img = cv2.imdecode(
                            np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR
                        )
                    else:
                        annotated_img = frame

                    if annotated_img is not None:
                        cv2.imshow("License Detection (Server)", annotated_img)
                    else:
                        cv2.imshow("License Detection (Server)", frame)

                    # Press 'x' to exit
                    if cv2.waitKey(1) & 0xFF == ord("x"):
                        break

                    # Throttle FPS to reduce server load
                    if TARGET_FPS > 0:
                        time.sleep(1.0 / TARGET_FPS)
                except Exception as e:
                    print(f"Server disconnected or error: {e}")
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(stream_to_server(SERVER_WS_URL))

