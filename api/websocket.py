"""
WebSocket endpoint for real-time driving license detection.

Protocol (same as driver_safety_system):
- Client sends raw JPEG bytes (single frame) per message.
- Server runs YOLOv8s license detection, plus optional Qwen OCR + validation,
  and sends back annotated JPEG (and optionally JSON detections).
"""

import asyncio
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ultralytics import YOLO

# Project root: driver_license_detection/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.pipeline import run_pipeline

router = APIRouter()

# Stream settings
STREAM_JPEG_QUALITY = 60
IMGSZ = 480
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "best.pt"

_executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="license_stream")
_yolo_model: YOLO | None = None


def _get_yolo(weights_path: Path) -> YOLO:
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(str(weights_path))
    return _yolo_model


def _decode_sync(data: bytes):
    """Decode JPEG to numpy frame."""
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return frame


def _run_pipeline_sync(
    frame: np.ndarray,
    weights_path: Path,
    qwen_enabled: bool,
    send_json: bool,
):
    """
    Run detection + optional OCR + ViT, draw, encode.
    Returns (payload_bytes, json_str or None).
    """
    if frame is None:
        return None, None
    model = _get_yolo(weights_path)
    detections, out_img = run_pipeline(
        frame,
        model,
        imgsz=IMGSZ,
        run_ocr_enabled=qwen_enabled,
        run_vit_enabled=False,
    )
    # Build JSON for client if requested (always send when send_json so client gets 2 messages every time)
    json_str = None
    if send_json:
        payload_list = [
            {
                "bbox": d["bbox"],
                "confidence": d["confidence"],
                "class": d["class"],
                "ocr_text": d.get("ocr_text", ""),
                "ocr_lines": d.get("ocr_lines", []),
                "validation_label": d.get("validation_label", "unknown"),
                "validation_confidence": d.get("validation_confidence", 0.0),
                "validation_reason": d.get("validation_reason", ""),
                "dl_numbers": d.get("dl_numbers", []),
            }
            for d in detections
        ]
        json_str = json.dumps({"detections": payload_list})
    ok, buffer = cv2.imencode(
        ".jpg", out_img, [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY]
    )
    payload = buffer.tobytes() if ok else None
    return payload, json_str


@router.websocket("/stream")
async def stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time driving license detection.

    Query params:
    - weights: YOLO weights path (default weights/best.pt)
    - qwen: 1 to enable Qwen OCR + validation on crops

    Protocol:
    - Client sends raw JPEG frame bytes (binary) per message.
    - If qwen=1: server sends text message (JSON detections) then binary (annotated JPEG).
    - Otherwise: server sends only binary (annotated JPEG).
    """
    await websocket.accept()
    print("WebSocket client connected (license detection)")

    weights_path = Path(websocket.query_params.get("weights", str(DEFAULT_WEIGHTS)))
    if not weights_path.is_absolute():
        weights_path = PROJECT_ROOT / weights_path
    if not weights_path.exists():
        await websocket.close(code=1011, reason=f"Weights not found: {weights_path}")
        return

    qwen_enabled = websocket.query_params.get("qwen", "0") == "1"
    send_json = qwen_enabled

    try:
        loop = asyncio.get_running_loop()
        while True:
            data = await websocket.receive_bytes()

            frame = await loop.run_in_executor(_executor, _decode_sync, data)
            if frame is None:
                continue

            payload, json_str = await loop.run_in_executor(
                _executor,
                _run_pipeline_sync,
                frame,
                weights_path,
                send_json,
                qwen_enabled,
            )
            if send_json and json_str:
                await websocket.send_text(json_str)
            if payload:
                await websocket.send_bytes(payload)

    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print("WebSocket error:", e)
