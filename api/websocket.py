"""
GPU-accelerated driving license detection & validation.

Endpoints:
  WS  /stream    — real-time streaming (YOLO every frame, optional async OCR)
  POST /validate — single-shot full pipeline: YOLO → crop → Qwen OCR → Indian DL rules
"""

import asyncio
import concurrent.futures
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.image_utils import crop_bbox
from utils.license_rules import validate_indian_dl
from utils.visualization import draw_bbox, draw_validation_status

router = APIRouter()

STREAM_JPEG_QUALITY = int(os.environ.get("STREAM_JPEG_QUALITY", "55"))
IMGSZ = int(os.environ.get("IMGSZ", "480"))
QWEN_INTERVAL = int(os.environ.get("QWEN_INTERVAL", "10"))
SAVE_VALIDATE_FRAMES = os.environ.get("SAVE_VALIDATE_FRAMES", "0").strip().lower() in {"1", "true", "yes", "on"}
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF = DEVICE == "cuda"

_yolo_model: YOLO | None = None
_encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY]

# IMPORTANT: keep validation responsive even if /stream is busy.
_stream_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="yolo_stream",
)
_validate_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="yolo_validate",
)


def _get_yolo(weights_path: Path) -> YOLO:
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model

    print(f"[ws] Loading YOLO from {weights_path} on {DEVICE} (half={USE_HALF})")
    _yolo_model = YOLO(str(weights_path))
    try:
        _yolo_model.model.float()
    except Exception:
        pass
    _yolo_model.fuse()
    _yolo_model.to(DEVICE)
    if USE_HALF:
        _yolo_model.model.half()
    _warmup_yolo(_yolo_model)
    return _yolo_model


def _warmup_yolo(model: YOLO, runs: int = 3) -> None:
    dummy = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
    for _ in range(runs):
        model.predict(
            dummy, imgsz=IMGSZ, conf=0.25, verbose=False,
            half=USE_HALF, device=DEVICE,
        )
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    print("[ws] YOLO warmup done")


# ---------------------------------------------------------------------------
# Background OCR worker (unchanged — runs in its own thread, never blocks YOLO)
# ---------------------------------------------------------------------------
_ocr_lock = threading.Lock()
_ocr_crop: np.ndarray | None = None
_ocr_pending = False
_ocr_result_text: str = ""
_ocr_result_rule: dict | None = None
_ocr_thread: threading.Thread | None = None
_ocr_stop = threading.Event()


def _ocr_worker():
    global _ocr_crop, _ocr_pending, _ocr_result_text, _ocr_result_rule
    from utils.qwen_ocr import qwen_ocr_and_validate

    while not _ocr_stop.is_set():
        crop = None
        with _ocr_lock:
            if _ocr_pending and _ocr_crop is not None:
                crop = _ocr_crop
                _ocr_pending = False

        if crop is None:
            _ocr_stop.wait(0.01)
            continue

        try:
            t0 = time.perf_counter()
            result = qwen_ocr_and_validate(crop)
            dt = time.perf_counter() - t0
            rule = validate_indian_dl(result.text or "")
            with _ocr_lock:
                _ocr_result_text = result.text or ""
                _ocr_result_rule = rule
            print(f"[ocr] Qwen done in {dt*1000:.0f}ms: {result.text[:80]!r}")
        except Exception as e:
            print(f"[ocr] Error: {e}")


def _ensure_ocr_thread():
    global _ocr_thread
    if _ocr_thread is None or not _ocr_thread.is_alive():
        _ocr_stop.clear()
        _ocr_thread = threading.Thread(target=_ocr_worker, daemon=True, name="qwen_ocr")
        _ocr_thread.start()


def _submit_ocr_crop(crop: np.ndarray):
    global _ocr_crop, _ocr_pending
    with _ocr_lock:
        _ocr_crop = crop
        _ocr_pending = True


def _get_ocr_result() -> tuple[str, dict | None]:
    with _ocr_lock:
        return _ocr_result_text, _ocr_result_rule


# ---------------------------------------------------------------------------
# YOLO detection + annotation (runs in _yolo_executor thread)
# ---------------------------------------------------------------------------
_frame_index = 0


def _fast_detect(
    frame: np.ndarray, model: YOLO, qwen_enabled: bool,
) -> tuple[list[dict], bytes | None]:
    """YOLO detect → draw → JPEG encode.  Returns (detections, jpeg_bytes)."""
    global _frame_index

    with torch.inference_mode():
        results = model.predict(
            frame, imgsz=IMGSZ, conf=0.02, verbose=False,
            device=DEVICE, half=USE_HALF,
        )

    out_img = frame.copy()
    detections: list[dict] = []

    if not results or len(results) == 0 or results[0].boxes is None:
        ok, buf = cv2.imencode(".jpg", out_img, _encode_params)
        return detections, buf.tobytes() if ok else None

    boxes = results[0].boxes
    confs = boxes.conf.detach().float().cpu().numpy()
    xyxys = boxes.xyxy.detach().float().cpu().numpy()
    indices = np.argsort(-confs)

    _frame_index += 1
    submit_ocr = qwen_enabled and (_frame_index % QWEN_INTERVAL == 0)
    ocr_text, ocr_rule = _get_ocr_result()

    for rank, idx in enumerate(indices.tolist()):
        conf = float(confs[idx])
        bbox = xyxys[idx].tolist()

        det: dict = {
            "bbox": bbox,
            "confidence": conf,
            "class": "driving_license",
            "ocr_lines": [],
            "ocr_text": "",
            "validation_label": "unknown",
            "validation_confidence": 0.0,
            "validation_reason": "",
        }

        if conf >= 0.02:
            crop = crop_bbox(frame, bbox, padding=10)
            if crop is not None and qwen_enabled and rank == 0:
                if submit_ocr:
                    ch, cw = crop.shape[:2]
                    if max(ch, cw) < 400:
                        scale = 400 / max(ch, cw)
                        crop = cv2.resize(
                            crop, (int(cw * scale), int(ch * scale)),
                            interpolation=cv2.INTER_CUBIC,
                        )
                    _submit_ocr_crop(crop)

                if ocr_text:
                    det["ocr_text"] = ocr_text
                    det["ocr_lines"] = [{"text": ocr_text, "confidence": 1.0}]
                if ocr_rule:
                    det["validation_label"] = ocr_rule.get("label", "unknown")
                    det["validation_confidence"] = ocr_rule.get("confidence", 0.0)
                    det["validation_reason"] = ocr_rule.get("reason", "")
                    det["dl_numbers"] = ocr_rule.get("dl_numbers", [])

        detections.append(det)

        label_text = "driving_license"
        vl = det.get("validation_label", "unknown")
        if vl and vl != "unknown":
            label_text = vl.upper()
        color = (
            (0, 255, 0) if vl == "valid"
            else (0, 0, 255) if vl == "invalid"
            else (0, 200, 255)
        )
        draw_bbox(out_img, bbox, label=label_text, confidence=conf, color=color)

    for d in detections:
        vl = d.get("validation_label")
        reason = d.get("validation_reason", "")
        if vl in ("valid", "invalid"):
            draw_validation_status(out_img, vl, reason)
            break
        if vl == "unknown" and reason and "could not read" in reason.lower():
            draw_validation_status(out_img, "unknown", reason)
            break

    ok, buf = cv2.imencode(".jpg", out_img, _encode_params)
    return detections, buf.tobytes() if ok else None


# ---------------------------------------------------------------------------
# WebSocket endpoint — simple sequential loop (no fragile async task juggling)
# ---------------------------------------------------------------------------
@router.websocket("/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    print("[ws] Client connected")

    weights_path = Path(
        websocket.query_params.get("weights", str(DEFAULT_WEIGHTS))
    )
    if not weights_path.is_absolute():
        weights_path = PROJECT_ROOT / weights_path
    if not weights_path.exists():
        await websocket.close(
            code=1011, reason=f"Weights not found: {weights_path}",
        )
        return

    qwen_enabled = websocket.query_params.get("qwen", "0") == "1"
    send_json = qwen_enabled

    if qwen_enabled:
        _ensure_ocr_thread()

    model = _get_yolo(weights_path)
    loop = asyncio.get_running_loop()
    frame_count = 0
    t_start = time.perf_counter()

    try:
        while True:
            data = await websocket.receive_bytes()

            frame = cv2.imdecode(
                np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR,
            )
            if frame is None:
                continue

            t0 = time.perf_counter()
            detections, jpeg_bytes = await loop.run_in_executor(
                _stream_executor, _fast_detect, frame, model, qwen_enabled,
            )
            dt = time.perf_counter() - t0

            frame_count += 1
            if frame_count % 50 == 0 and dt > 0:
                elapsed_total = time.perf_counter() - t_start
                avg_fps = frame_count / elapsed_total if elapsed_total > 0 else 0
                print(
                    f"[ws] frame {frame_count}: {dt*1000:.0f}ms "
                    f"(inst {1/dt:.0f} fps, avg {avg_fps:.0f} fps)"
                )

            if jpeg_bytes:
                await websocket.send_bytes(jpeg_bytes)

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
                await websocket.send_text(
                    json.dumps({"detections": payload_list})
                )

    except WebSocketDisconnect:
        print(f"[ws] Client disconnected (processed {frame_count} frames)")
    except Exception as e:
        print(f"[ws] Error after {frame_count} frames: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# POST /validate — single-shot full pipeline (YOLO → Qwen OCR → Indian DL rules)
# ---------------------------------------------------------------------------
def _full_validate(frame: np.ndarray, model: YOLO) -> dict:
    """
    Synchronous full pipeline: YOLO detect → crop → Qwen OCR → Indian DL rules.
    Returns a dict with the verdict.
    """
    from utils.qwen_ocr import qwen_ocr_and_validate

    t_total = time.perf_counter()
    h, w = frame.shape[:2]
    print(f"[validate] start: frame={w}x{h} bytes? (decoded)")

    # 1) YOLO detection
    t0 = time.perf_counter()
    with torch.inference_mode():
        results = model.predict(
            frame, imgsz=IMGSZ, conf=0.02, verbose=False,
            device=DEVICE, half=USE_HALF,
        )
    yolo_ms = (time.perf_counter() - t0) * 1000

    if not results or len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
        return {
            "verdict": "unknown",
            "message": "No driving licence detected in frame",
            "timings": {"yolo_ms": round(yolo_ms)},
        }

    boxes = results[0].boxes
    best_idx = int(boxes.conf.argmax())
    bbox = boxes.xyxy[best_idx].detach().float().cpu().numpy().tolist()
    det_conf = float(boxes.conf[best_idx])

    # 2) Crop
    crop = crop_bbox(frame, bbox, padding=10)
    if crop is None:
        return {
            "verdict": "unknown",
            "message": "Failed to crop detected region",
            "detection_confidence": det_conf,
            "timings": {"yolo_ms": round(yolo_ms)},
        }

    ch, cw = crop.shape[:2]
    if max(ch, cw) < 400:
        scale = 400 / max(ch, cw)
        crop = cv2.resize(
            crop, (int(cw * scale), int(ch * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    # 3) Qwen OCR (synchronous — runs on GPU)
    t0 = time.perf_counter()
    ocr_result = qwen_ocr_and_validate(crop)
    ocr_ms = (time.perf_counter() - t0) * 1000

    ocr_text = ocr_result.text or ""
    print(f"[validate] Qwen OCR ({ocr_ms:.0f}ms): {ocr_text[:120]!r}")

    # 4) Indian DL rule-based validation
    rule = validate_indian_dl(ocr_text)

    total_ms = (time.perf_counter() - t_total) * 1000
    label = rule.get("label", "unknown")

    return {
        "verdict": label,
        "message": (
            "VALID Indian Driving Licence" if label == "valid"
            else "NOT a valid Driving Licence" if label == "invalid"
            else "Could not determine validity"
        ),
        "detection_confidence": det_conf,
        "bbox": bbox,
        "ocr_text": ocr_text,
        "qwen_verdict": ocr_result.is_valid,
        "qwen_confidence": ocr_result.confidence,
        "qwen_reason": ocr_result.reason,
        "rule_label": label,
        "rule_confidence": rule.get("confidence", 0.0),
        "rule_reason": rule.get("reason", ""),
        "dl_numbers": rule.get("dl_numbers", []),
        "timings": {
            "yolo_ms": round(yolo_ms),
            "ocr_ms": round(ocr_ms),
            "total_ms": round(total_ms),
        },
    }


@router.post("/validate")
async def validate_endpoint(request: Request):
    """
    Accept a raw JPEG image body and run the full validation pipeline.
    Content-Type: image/jpeg (raw bytes) or multipart/form-data.
    """
    t_req = time.perf_counter()
    body = await request.body()
    if not body:
        return JSONResponse({"verdict": "error", "message": "Empty body"}, 400)

    frame = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"verdict": "error", "message": "Invalid image data"}, 400)

    if SAVE_VALIDATE_FRAMES:
        try:
            out_dir = PROJECT_ROOT / "output" / "validate_debug"
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time() * 1000)
            cv2.imwrite(str(out_dir / f"in_{ts}.jpg"), frame)
            print(f"[validate] saved input frame: output/validate_debug/in_{ts}.jpg")
        except Exception as e:
            print(f"[validate] save input failed: {e}")

    model = _get_yolo(DEFAULT_WEIGHTS)
    loop = asyncio.get_running_loop()
    print(f"[validate] received {len(body)} bytes, decoded shape={frame.shape}")
    result = await loop.run_in_executor(_validate_executor, _full_validate, frame, model)
    print(f"[validate] done in {(time.perf_counter()-t_req)*1000:.0f}ms verdict={result.get('verdict')}")
    return result
