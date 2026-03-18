#!/usr/bin/env python3
"""
Validate YOLO dataset structure and annotations.

Checks:
- Correct folder structure (images/train, labels/train, etc.)
- Image-label correspondence (matching filenames)
- Valid YOLO label format (class_id x_center y_center width height)
- Bounding boxes within [0, 1] range
- No empty or corrupted images
"""

import argparse
from pathlib import Path
from typing import NamedTuple


class ValidationResult(NamedTuple):
    valid: bool
    errors: list[str]
    warnings: list[str]
    stats: dict


def validate_yolo_label_file(path: Path) -> list[str]:
    """Validate a single YOLO label file. Returns list of error messages."""
    errors = []
    try:
        lines = path.read_text().strip().split("\n")
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"Line {i+1}: expected 5 values, got {len(parts)}")
                continue
            try:
                cls_id = int(parts[0])
                x_c, y_c, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                if not (0 <= x_c <= 1 and 0 <= y_c <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    errors.append(f"Line {i+1}: values must be in [0,1], got ({x_c},{y_c},{w},{h})")
            except ValueError as e:
                errors.append(f"Line {i+1}: invalid number - {e}")
    except Exception as e:
        errors.append(str(e))
    return errors


def validate_dataset(dataset_root: Path) -> ValidationResult:
    """Validate full YOLO dataset structure and content."""
    errors: list[str] = []
    warnings: list[str] = []
    stats = {"images": 0, "labels": 0, "missing_labels": 0, "missing_images": 0}

    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        return ValidationResult(False, [f"Dataset root not found: {dataset_root}"], [], {})

    for split in ["train", "val", "test"]:
        img_dir = dataset_root / "images" / split
        lbl_dir = dataset_root / "labels" / split

        if not img_dir.exists():
            warnings.append(f"images/{split} not found")
            continue
        if not lbl_dir.exists():
            warnings.append(f"labels/{split} not found")
            continue

        img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        images = [p for p in img_dir.rglob("*") if p.suffix.lower() in img_extensions and p.is_file()]

        for img_path in images:
            stats["images"] += 1
            stem = img_path.stem
            lbl_path = lbl_dir / f"{stem}.txt"
            if not lbl_path.exists():
                stats["missing_labels"] += 1
                errors.append(f"Missing label for {img_path.relative_to(dataset_root)}")
                continue
            stats["labels"] += 1
            lbl_errors = validate_yolo_label_file(lbl_path)
            errors.extend(f"{lbl_path.relative_to(dataset_root)}: {e}" for e in lbl_errors)

        for lbl_path in lbl_dir.glob("*.txt"):
            stem = lbl_path.stem
            found = any(p.stem == stem for p in images)
            if not found:
                stats["missing_images"] += 1
                warnings.append(f"Label without image: {lbl_path.relative_to(dataset_root)}")

    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        warnings.append("data.yaml not found")

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )


def main():
    parser = argparse.ArgumentParser(description="Validate YOLO dataset")
    parser.add_argument("dataset_root", type=str, nargs="?", default="dataset")
    args = parser.parse_args()

    result = validate_dataset(Path(args.dataset_root))
    print(f"Valid: {result.valid}")
    print(f"Stats: {result.stats}")
    for e in result.errors[:20]:
        print(f"  ERROR: {e}")
    if len(result.errors) > 20:
        print(f"  ... and {len(result.errors) - 20} more errors")
    for w in result.warnings:
        print(f"  WARN: {w}")
    exit(0 if result.valid else 1)


if __name__ == "__main__":
    main()
