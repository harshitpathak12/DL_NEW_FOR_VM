#!/usr/bin/env python3
"""
Convert MIDV-500 annotations from polygon/quad format to YOLO bounding box format.

MIDV-500 annotations provide document corners as a 4-point polygon (quad):
  [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

YOLO format requires:
  class_id x_center y_center width height
  (all coordinates normalized to [0, 1] relative to image dimensions)

Usage:
    python convert_annotations.py --dataset_root MIDV500 --output_dir dataset
"""

import argparse
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm


# JSON keys that may contain quadrilateral/polygon annotations
ANNOTATION_KEYS = ["quad", "quadrilateral", "polygon", "points"]

# Document type folders that contain driving licenses in MIDV-500
DRIVING_LICENSE_FOLDERS = [
    "02_aut_drvlic_new",
    "12_deu_drvlic_new",
    "13_deu_drvlic_old",
    "19_esp_drvlic",
    "23_fin_drvlic",
    "26_hrv_drvlic",
    "29_irn_drvlic",
    "30_ita_drvlic",
    "31_jpn_drvlic",
    "35_nor_drvlic",
    "36_pol_drvlic",
    "38_rou_drvlic",
]

# Class ID for driving_license (single class)
CLASS_ID = 0

# Supported image extensions
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def polygon_to_bbox(points: list[list[float]]) -> tuple[float, float, float, float]:
    """
    Convert 4-point polygon to axis-aligned bounding box.
    Returns (x_min, y_min, x_max, y_max) in pixel coordinates.
    """
    pts = np.array(points, dtype=np.float32)
    x_min = float(np.min(pts[:, 0]))
    x_max = float(np.max(pts[:, 0]))
    y_min = float(np.min(pts[:, 1]))
    y_max = float(np.max(pts[:, 1]))
    return x_min, y_min, x_max, y_max


def bbox_to_yolo(
    x_min: float, y_min: float, x_max: float, y_max: float,
    img_width: int, img_height: int
) -> tuple[float, float, float, float]:
    """
    Convert pixel bbox to YOLO normalized format.

    YOLO format: class_id x_center y_center width height
    All values normalized to [0, 1].

    Returns:
        (x_center_norm, y_center_norm, width_norm, height_norm)
    """
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min

    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height

    # Clamp to [0, 1]
    x_center_norm = max(0.0, min(1.0, x_center_norm))
    y_center_norm = max(0.0, min(1.0, y_center_norm))
    width_norm = max(0.0, min(1.0, width_norm))
    height_norm = max(0.0, min(1.0, height_norm))

    return x_center_norm, y_center_norm, width_norm, height_norm


def get_annotation_path(image_path: Path, dataset_root: Path) -> Path | None:
    """
    Resolve the corresponding annotation JSON path for an image.
    MIDV-500 structure: images/... -> ground_truth/... (replace images with ground_truth, .tif->.json)
    """
    try:
        rel_path = image_path.relative_to(dataset_root)
        parts = list(rel_path.parts)
        if "images" in parts:
            idx = parts.index("images")
            parts[idx] = "ground_truth"
        else:
            # Maybe image is next to annotation
            return image_path.with_suffix(".json") if image_path.with_suffix(".json").exists() else None

        json_path = dataset_root / Path(*parts).with_suffix(".json")
        return json_path if json_path.exists() else None
    except ValueError:
        return None


def parse_annotation(annotation_path: Path) -> list[list[float]] | None:
    """
    Load quadrilateral/polygon from MIDV-500 JSON annotation.

    Returns:
        List of 4 points [[x,y], ...] or None if invalid
    """
    try:
        data = json.loads(annotation_path.read_text())
        for key in ANNOTATION_KEYS:
            if key in data and isinstance(data[key], (list, tuple)) and len(data[key]) >= 4:
                pts = data[key][:4]
                result = []
                for pt in pts:
                    if isinstance(pt, dict):
                        result.append([float(pt.get("x", 0)), float(pt.get("y", 0))])
                    elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        result.append([float(pt[0]), float(pt[1])])
                if len(result) == 4:
                    return result
        return None
    except Exception:
        return None


def find_annotated_images(
    dataset_root: Path,
    images_root: Path,
    driving_license_only: bool = True,
) -> list[tuple[Path, Path]]:
    """
    Find all images that have corresponding annotations.
    Returns list of (image_path, annotation_path) tuples.
    """
    pairs = []
    for img_path in images_root.rglob("*"):
        if img_path.suffix.lower() in IMG_EXTENSIONS and img_path.is_file():
            if driving_license_only:
                if not any(dl in img_path.parts for dl in DRIVING_LICENSE_FOLDERS):
                    continue
            ann_path = get_annotation_path(img_path, dataset_root)
            if ann_path and ann_path.exists():
                quad = parse_annotation(ann_path)
                if quad is not None:
                    pairs.append((img_path, ann_path))
    return pairs


def convert_image(
    image_path: Path,
    annotation_path: Path,
    output_labels_dir: Path,
    output_images_dir: Path | None = None,
    copy_images: bool = False,
) -> bool:
    """
    Convert one image's annotation to YOLO format and optionally copy image.

    Returns True on success.
    """
    import cv2

    quad = parse_annotation(annotation_path)
    if quad is None:
        return False

    img = cv2.imread(str(image_path))
    if img is None:
        return False

    h, w = img.shape[:2]
    x_min, y_min, x_max, y_max = polygon_to_bbox(quad)
    x_c, y_c, w_norm, h_norm = bbox_to_yolo(x_min, y_min, x_max, y_max, w, h)

    # YOLO label line: class_id x_center y_center width height
    label_line = f"{CLASS_ID} {x_c:.6f} {y_c:.6f} {w_norm:.6f} {h_norm:.6f}\n"

    # Use image stem for label filename (images and labels must have same base name)
    stem = image_path.stem
    label_path = output_labels_dir / f"{stem}.txt"
    label_path.write_text(label_line)

    if copy_images and output_images_dir:
        import shutil
        dest_img = output_images_dir / f"{stem}{image_path.suffix}"
        shutil.copy2(image_path, dest_img)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert MIDV-500 annotations to YOLO format"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Path to MIDV-500 dataset root",
    )
    parser.add_argument(
        "--images_root",
        type=str,
        default=None,
        help="Path to images (default: dataset_root/images)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset",
        help="Output directory for YOLO dataset structure",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which split to convert (or 'all' for single directory)",
    )
    parser.add_argument(
        "--all_documents",
        action="store_true",
        help="Include all document types, not just driving licenses",
    )
    parser.add_argument(
        "--copy_images",
        action="store_true",
        help="Copy images to output structure (otherwise only labels are written)",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    images_root = Path(args.images_root) if args.images_root else dataset_root / "images"
    output_dir = Path(args.output_dir)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    # Find annotated images
    pairs = find_annotated_images(
        dataset_root,
        images_root,
        driving_license_only=not args.all_documents,
    )

    if not pairs:
        print("No annotated images found. Ensure:")
        print("  - Dataset has ground_truth/ matching images/ structure")
        print("  - JSON files contain 'quad' or similar key with 4 points")
        return

    # Create output directories
    labels_dir = output_dir / "labels"
    if args.split == "all":
        labels_dir = labels_dir / "all"
    else:
        labels_dir = labels_dir / args.split
    labels_dir.mkdir(parents=True, exist_ok=True)

    images_dir = None
    if args.copy_images:
        images_out = output_dir / "images"
        if args.split == "all":
            images_dir = images_out / "all"
        else:
            images_dir = images_out / args.split
        images_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    for img_path, ann_path in tqdm(pairs, desc="Converting"):
        if convert_image(
            img_path,
            ann_path,
            labels_dir,
            output_images_dir=images_dir,
            copy_images=args.copy_images,
        ):
            converted += 1

    print(f"Converted {converted}/{len(pairs)} annotations to {labels_dir}")


if __name__ == "__main__":
    main()
