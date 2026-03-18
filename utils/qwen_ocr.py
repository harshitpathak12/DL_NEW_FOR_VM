"""
Qwen-based OCR and validation for driving license crops.

This module uses a local Qwen vision-language model (via Hugging Face
`transformers`) to:

- Read text from a cropped license image
- Optionally validate whether the card looks like a REAL driving licence

The model is prompted to return a small JSON object so the rest of the
pipeline can stay simple.

Default model: Qwen2.5-VL 3B Instruct (you can change MODEL_ID below).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"  # you can switch to a smaller VLM when Qwen releases one (e.g. Qwen2.5-VL-1.8B-Instruct)

_processor: AutoProcessor | None = None
_model: AutoModelForVision2Seq | None = None


@dataclass
class QwenOCRResult:
    text: str
    is_valid: str  # "valid" | "invalid" | "unknown"
    confidence: float
    reason: str


def _load_qwen():
    """
    Lazy-load Qwen model and processor.

    Uses half-precision on CUDA if available, otherwise float32 on CPU.
    """
    global _processor, _model
    if _processor is not None and _model is not None:
        return _processor, _model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    _processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    _model.eval()
    return _processor, _model


def _build_prompt() -> str:
    """
    Instruction for Qwen: read license text and judge validity.
    """
    return (
        "You are an OCR and document validation assistant.\n"
        "You are given a cropped image of a driving licence card.\n"
        "1. Read all text you can see on the card.\n"
        "2. Decide if it looks like a REAL driving licence (not a random paper or other card).\n"
        "3. Respond ONLY with a JSON object, no extra text.\n"
        'The JSON schema must be:\n'
        '{\n'
        '  "text": "<all text you can read as one string>",\n'
        '  "is_valid": "valid" | "invalid" | "unknown",\n'
        '  "confidence": <number between 0 and 1>,\n'
        '  "reason": "<short explanation of why you think it is valid/invalid or unknown>"\n'
        "}\n"
    )


def qwen_ocr_and_validate(image_bgr: np.ndarray, max_size: int = 1024) -> QwenOCRResult:
    """
    Run Qwen OCR + validation on a license crop (BGR np.ndarray).

    Returns:
        QwenOCRResult with text, is_valid, confidence, reason.
    """
    if image_bgr is None or image_bgr.size == 0:
        return QwenOCRResult(text="", is_valid="unknown", confidence=0.0, reason="empty image")

    # Resize very large crops to keep memory reasonable
    import cv2

    h, w = image_bgr.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Convert BGR -> RGB PIL.Image
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)

    processor, model = _load_qwen()

    prompt = _build_prompt()
    messages: list[Dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": pil_img},
            ],
        }
    ]

    # Qwen-VL conversation formatting
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = None, None
    try:
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )
    except Exception:
        inputs = processor(
            images=[pil_img],
            text=[text_input],
            return_tensors="pt",
        )

    inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    # Try to extract JSON from the response
    import json
    import re

    json_str = generated_text.strip()
    # If model wrapped JSON in extra text or markdown, try to find the first {...} block
    match = re.search(r"\{.*\}", json_str, flags=re.DOTALL)
    if match:
        json_str = match.group(0)

    try:
        data = json.loads(json_str)
    except Exception:
        # Fallback: return raw text
        return QwenOCRResult(
            text=generated_text.strip(),
            is_valid="unknown",
            confidence=0.0,
            reason="could not parse JSON from model output",
        )

    text = str(data.get("text", "")).strip()
    is_valid = str(data.get("is_valid", "unknown")).lower()
    if is_valid not in {"valid", "invalid", "unknown"}:
        is_valid = "unknown"
    try:
        confidence = float(data.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    reason = str(data.get("reason", "")).strip()

    return QwenOCRResult(
        text=text,
        is_valid=is_valid,
        confidence=confidence,
        reason=reason,
    )

