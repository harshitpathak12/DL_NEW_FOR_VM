"""
Qwen-based OCR and validation for driving license crops (GPU-accelerated).

Uses Qwen2.5-VL 3B Instruct with:
- FP16 / BF16 on CUDA (auto-selected based on GPU capability)
- Flash Attention 2 when available
- torch.inference_mode for faster forward passes
- Prompt template pre-built once
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

_processor: AutoProcessor | None = None
_model: Qwen2_5_VLForConditionalGeneration | None = None
_device: str = "cpu"

_PROMPT = (
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


@dataclass
class QwenOCRResult:
    text: str
    is_valid: str
    confidence: float
    reason: str


def _select_dtype() -> torch.dtype:
    """Pick the best dtype for the available GPU."""
    if not torch.cuda.is_available():
        return torch.float32
    cap = torch.cuda.get_device_capability()
    if cap >= (8, 0):
        return torch.bfloat16
    return torch.float16


def _load_qwen():
    """Lazy-load Qwen model and processor onto GPU with optimal settings."""
    global _processor, _model, _device
    if _processor is not None and _model is not None:
        return _processor, _model

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _select_dtype()

    print(f"[qwen] Loading {MODEL_ID} on {_device} (dtype={dtype})")
    t0 = time.perf_counter()

    _processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
    }
    if _device == "cuda":
        model_kwargs["device_map"] = "auto"
        try:
            import flash_attn  # noqa: F401
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("[qwen] Using Flash Attention 2")
        except ImportError:
            print("[qwen] flash_attn not installed, using default attention")

    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, **model_kwargs
    )
    _model.eval()

    if _device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"[qwen] Model loaded in {dt:.1f}s")

    return _processor, _model


def qwen_ocr_and_validate(image_bgr: np.ndarray, max_size: int = 1024) -> QwenOCRResult:
    """Run Qwen OCR + validation on a license crop (BGR np.ndarray)."""
    if image_bgr is None or image_bgr.size == 0:
        return QwenOCRResult(text="", is_valid="unknown", confidence=0.0, reason="empty image")

    h, w = image_bgr.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)

    processor, model = _load_qwen()

    messages: list[Dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _PROMPT},
                {"type": "image", "image": pil_img},
            ],
        }
    ]

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

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, input_len:]
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    json_str = generated_text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", json_str, flags=re.DOTALL)
    if fence:
        json_str = fence.group(1)
    else:
        bare = re.search(r"\{.*\}", json_str, flags=re.DOTALL)
        if bare:
            json_str = bare.group(0)

    try:
        data = json.loads(json_str)
    except Exception:
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

    return QwenOCRResult(text=text, is_valid=is_valid, confidence=confidence, reason=reason)
