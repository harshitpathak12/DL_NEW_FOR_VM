"""
Rule-based validation for Indian driving licence (based on OCR text).

Supports standard format used across Indian states, e.g.:
- "Indian Union Driving Licence" (Bihar, etc.)
- "Issued by Government of [State]"
- License number: State code (2 letters) + RTO (2 digits) + Year (4) + 7 digits
  e.g. BR22 20250006557, MH02 20191234567, DL-06-20190001234
- Fields: Name, Date Of Birth, Blood Group, Validity (NT/TR), Issue Date, Address, etc.
"""

import re
from typing import Any


# Keywords that suggest an Indian driving licence (Union/state format)
LICENCE_KEYWORDS = [
    # Standard title (all states use this or similar)
    "indian union driving licence",
    "indian union driving license",
    "union driving licence",
    "union driving license",
    "driving licence",
    "driving license",
    "driving licen",
    "driving lic",
    # Issuing authority
    "issued by government",
    "government of",
    "government of india",
    "republic of india",
    # Common field labels
    "licence no",
    "license no",
    "licence no.",
    "license no.",
    "licence number",
    "license number",
    "issue date",
    "date of issue",
    "date of first issue",
    "validity (nt)",
    "validity (tr)",
    "valid from",
    "valid till",
    "valid to",
    "date of birth",
    "date of birth:",
    "blood group",
    "blood group :",
    "name:",
    "address:",
    "address",
    "son/daughter/wife",
    "father's name",
    "father name",
    "holder's signature",
    "organ donor",
    "issuing authority",
    "motor vehicle",
    "rto",
    "regional transport",
    "transport office",
    "authorisation to drive",
    "categories",
    "permanent address",
]

# Indian DL number: 2 letter state + 2 digit RTO + (space/hyphen) + 4 digit year + 7 digits
# e.g. BR22 20250006557, MH02 20191234567, DL-06-20190001234, HR0619850034761
DL_NUMBER_PATTERN = re.compile(
    r"\b[A-Z]{2}[\s\-]?\d{2}[\s\-]?(?:19|20)\d{2}\s*\d{7}\b",
    re.IGNORECASE,
)
# With spaces between parts: BR 22 2025 0006557
DL_NUMBER_SPACED = re.compile(
    r"\b[A-Z]{2}\s+\d{2}\s+(?:19|20)\d{2}\s*\d{5,7}\b",
    re.IGNORECASE,
)
# Hyphen form at bottom of card: BR-D2217017627
DL_NUMBER_HYPHEN = re.compile(
    r"\b[A-Z]{2}\s*-\s*[A-Z]?\d{10,15}\b",
    re.IGNORECASE,
)
# Relaxed: 2 letters + digits (catches OCR errors / variants)
DL_NUMBER_RELAXED = re.compile(
    r"\b[A-Z]{2}[\s\-]?\d{2,}[\s\-]?\d{4,}\b",
    re.IGNORECASE,
)


def validate_indian_dl(ocr_text: str) -> dict[str, Any]:
    """
    Rule-based validation for Indian driving licence using OCR text.
    Matches standard format: Indian Union Driving Licence, Issued by Government of [State],
    licence number (e.g. BR22 20250006557), Validity, Name, DOB, Blood Group, Address, etc.

    Returns:
        {
            "label": "valid" | "invalid" | "unknown",
            "confidence": float (0-1),
            "reason": str (short reason for display),
        }
    """
    if not ocr_text or not ocr_text.strip():
        return {
            "label": "unknown",
            "confidence": 0.0,
            "reason": "No text from licence",
        }

    text = ocr_text.strip().lower()
    raw = ocr_text.strip()
    reasons_ok = []
    reasons_fail = []

    # 1) Must have at least one licence-related keyword
    found_keyword = any(kw in text for kw in LICENCE_KEYWORDS)
    if not found_keyword:
        return {
            "label": "invalid",
            "confidence": 0.0,
            "reason": "Not a driving licence",
        }
    reasons_ok.append("Licence keyword found")

    # 2) Look for licence number pattern (any supported format)
    numbers: list[str] = []
    for pat in (DL_NUMBER_PATTERN, DL_NUMBER_SPACED, DL_NUMBER_HYPHEN, DL_NUMBER_RELAXED):
        for m in pat.finditer(raw):
            num = m.group(0).strip()
            if num not in numbers:
                numbers.append(num)

    has_strict = any(DL_NUMBER_PATTERN.search(n) for n in numbers)
    has_spaced = any(DL_NUMBER_SPACED.search(n) for n in numbers)
    has_hyphen = any(DL_NUMBER_HYPHEN.search(n) for n in numbers)
    has_relaxed = any(DL_NUMBER_RELAXED.search(n) for n in numbers)

    if has_strict or has_spaced or has_hyphen:
        reasons_ok.append("DL number format OK")
    elif has_relaxed:
        reasons_ok.append("DL number (relaxed) OK")
    else:
        # Strong Union/state keywords: accept as valid even without clear number (OCR may miss it)
        strong_keywords = (
            "indian union",
            "union driving",
            "issued by government",
            "government of",
            "validity (nt)",
            "validity (tr)",
            "issue date",
            "date of first issue",
            "date of birth",
            "blood group",
            "name:",
            "address:",
        )
        if any(k in text for k in strong_keywords):
            reasons_ok.append("Indian DL format (keywords)")
        else:
            reasons_fail.append("No licence number pattern")

    # 3) Reject if too short (likely not a full licence)
    if len(text) < 15:
        return {
            "label": "invalid",
            "confidence": 0.0,
            "reason": "Too little text",
        }

    if reasons_fail:
        return {
            "label": "invalid",
            "confidence": 0.3,
            "reason": "; ".join(reasons_fail),
            "dl_numbers": numbers,
        }

    # Valid Indian DL
    reason = "Indian DL"
    if has_strict or has_spaced:
        reason = "VALID (Indian DL)"
    elif has_hyphen or has_relaxed:
        reason = "VALID (Indian DL)"
    else:
        reason = "VALID (Indian DL)"
    confidence = 0.9 if (has_strict or has_spaced) else 0.85 if (has_hyphen or has_relaxed) else 0.8

    return {
        "label": "valid",
        "confidence": confidence,
        "reason": reason,
        "dl_numbers": numbers,
    }
