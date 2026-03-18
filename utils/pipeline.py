"""
Single pipeline: YOLOv8s detection (GPU FP16) -> crop -> Qwen OCR + validation.
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from utils.image_utils import crop_bbox
from utils.visualization import draw_bbox, draw_validation_status
from utils.qwen_ocr import QwenOCRResult, qwen_ocr_and_validate
from utils.license_rules import validate_indian_dl

QWEN_INTERVAL = 10
_frame_index = 0
_last_rule_result: dict | None = None
_last_ocr_text: str = ""


def run_pipeline(
    frame: np.ndarray,
    model: YOLO,
    imgsz: int = 640,
    conf_threshold: float = 0.02,
    yolo_conf: float = 0.02,
    run_ocr_enabled: bool = True,
    run_vit_enabled: bool = False,
    crop_padding: int = 10,
    device: str = "cuda",
    half: bool = True,
) -> tuple[list[dict], np.ndarray]:
    """
    Run detection, then for each bbox above conf_threshold:
        crop -> Qwen OCR + licence validation.

    Args:
        frame: BGR image.
        model: Loaded YOLO model (already on target device).
        imgsz: YOLO input size.
        conf_threshold: Only run Qwen for detections with confidence >= this.
        yolo_conf: YOLO internal confidence threshold.
        run_ocr_enabled: Whether to run Qwen OCR/validation on crops.
        run_vit_enabled: Ignored (ViT removed in favour of Qwen).
        crop_padding: Padding around bbox for crop.
        device: 'cuda' or 'cpu'.
        half: Use FP16 inference (only on CUDA).
    """
    global _frame_index, _last_rule_result, _last_ocr_text

    if frame is None:
        return [], np.array([])

    results = model.predict(
        frame,
        imgsz=imgsz,
        conf=yolo_conf,
        verbose=False,
        device=device,
        half=half and device == "cuda",
    )
    out_img = frame.copy()
    detections: list[dict] = []

    if not results or len(results) == 0 or results[0].boxes is None:
        return detections, out_img

    boxes = results[0].boxes
    indices = sorted(
        range(len(boxes)),
        key=lambda i: float(boxes.conf[i]),
        reverse=True,
    )
    max_qwen_detections = 1

    _frame_index += 1
    run_qwen_this_frame = (_frame_index % QWEN_INTERVAL) == 0

    qwen_count = 0
    for idx in indices:
        box = boxes[idx]
        xyxy = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        bbox = xyxy.tolist()

        det: dict[str, Any] = {
            "bbox": bbox,
            "confidence": conf,
            "class": "driving_license",
            "ocr_lines": [],
            "ocr_text": "",
            "validation_label": "unknown",
            "validation_confidence": 0.0,
            "validation_reason": "",
        }

        if conf >= conf_threshold:
            crop = crop_bbox(frame, bbox, padding=crop_padding)
            if crop is not None and run_ocr_enabled and qwen_count < max_qwen_detections:
                qwen_count += 1
                ch, cw = crop.shape[:2]
                if max(ch, cw) < 400:
                    scale = 400 / max(ch, cw)
                    crop_for_ocr = cv2.resize(
                        crop,
                        (int(cw * scale), int(ch * scale)),
                        interpolation=cv2.INTER_CUBIC,
                    )
                else:
                    crop_for_ocr = crop

                if run_qwen_this_frame or _last_rule_result is None:
                    qwen_out: QwenOCRResult = qwen_ocr_and_validate(crop_for_ocr)
                    det["ocr_text"] = qwen_out.text
                    if qwen_out.text:
                        det["ocr_lines"] = [{"text": qwen_out.text, "confidence": qwen_out.confidence}]
                    _last_ocr_text = qwen_out.text or ""
                    _last_rule_result = validate_indian_dl(_last_ocr_text)
                else:
                    det["ocr_text"] = _last_ocr_text
                    if _last_ocr_text:
                        det["ocr_lines"] = [{"text": _last_ocr_text, "confidence": 1.0}]

                rule = _last_rule_result or {}
                det["validation_label"] = rule.get("label", "unknown")
                det["validation_confidence"] = rule.get("confidence", 0.0)
                det["validation_reason"] = rule.get("reason", "")
                det["dl_numbers"] = rule.get("dl_numbers", [])

                print(
                    f"[pipeline] YOLO conf={conf:.3f}, crop={cw}x{ch}, "
                    f"rule_label={det['validation_label']}, conf={det['validation_confidence']:.2f}, "
                    f"text={det['ocr_text'][:120]!r} (qwen_run={run_qwen_this_frame})"
                )

        detections.append(det)

        label_text = "driving_license"
        if det.get("validation_label") and det["validation_label"] != "unknown":
            label_text = det["validation_label"].upper()
        color = (
            (0, 255, 0)
            if det.get("validation_label") == "valid"
            else (0, 0, 255)
            if det.get("validation_label") == "invalid"
            else (0, 200, 255)
        )
        out_img = draw_bbox(out_img, bbox, label=label_text, confidence=conf, color=color)

    for d in detections:
        vl = d.get("validation_label")
        reason = d.get("validation_reason", "")
        if vl in ("valid", "invalid"):
            out_img = draw_validation_status(out_img, vl, reason)
            break
        if vl == "unknown" and reason and "could not read" in reason.lower():
            out_img = draw_validation_status(out_img, "unknown", reason)
            break

    return detections, out_img
