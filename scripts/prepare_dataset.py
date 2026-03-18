#!/usr/bin/env python3
"""
Orchestrate the full MIDV-500 dataset preparation pipeline.

This script:
1. Optionally downloads the MIDV-500 dataset (via midv500 package or from Kaggle)
2. Extracts frames from videos (optional, for additional training data)
3. Filters driving license document types
4. Converts annotations to YOLO format
5. Creates train/validation/test splits

Usage:
    python prepare_dataset.py --dataset_root MIDV500 --output_dir dataset
    python prepare_dataset.py --dataset_root MIDV500 --extract_frames --frame_interval 5
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import Literal

import sys

# Add project root for imports when run as script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.extract_frames import find_videos, extract_frames_from_video
from scripts.convert_annotations import (
    find_annotated_images,
    parse_annotation,
    polygon_to_bbox,
    bbox_to_yolo,
    get_annotation_path,
)
import cv2
import yaml
from tqdm import tqdm


# Default split ratios: 80% train, 10% val, 10% test
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

CLASS_NAMES = {0: "driving_license"}


def create_data_yaml(output_dir: Path, num_classes: int = 1) -> Path:
    """Create data.yaml for YOLO training."""
    data = {
        "path": str(output_dir.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": CLASS_NAMES,
        "nc": num_classes,
    }
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return yaml_path


def convert_and_write_label(
    image_path: Path,
    annotation_path: Path,
    img_width: int,
    img_height: int,
    label_path: Path,
) -> bool:
    """Convert annotation to YOLO format and write to file."""
    quad = parse_annotation(annotation_path)
    if quad is None:
        return False
    x_min, y_min, x_max, y_max = polygon_to_bbox(quad)
    x_c, y_c, w_norm, h_norm = bbox_to_yolo(x_min, y_min, x_max, y_max, img_width, img_height)
    label_line = f"0 {x_c:.6f} {y_c:.6f} {w_norm:.6f} {h_norm:.6f}\n"
    label_path.write_text(label_line)
    return True


def prepare_from_images(
    dataset_root: Path,
    output_dir: Path,
    split_ratios: tuple[float, float, float] = (TRAIN_RATIO, VAL_RATIO, TEST_RATIO),
    driving_license_only: bool = True,
    seed: int = 42,
) -> int:
    """
    Prepare dataset from existing MIDV-500 images (with annotations).
    Creates train/val/test splits and YOLO labels.
    """
    # MIDV-500 structure: dataset_root/doc_type/images/... and doc_type/ground_truth/...
    images_root = dataset_root

    pairs = find_annotated_images(
        dataset_root,
        images_root,
        driving_license_only=driving_license_only,
    )

    if not pairs:
        return 0

    random.seed(seed)
    random.shuffle(pairs)

    n = len(pairs)
    t, v, te = split_ratios
    n_train = int(n * t)
    n_val = int(n * v)
    n_test = n - n_train - n_val

    splits: dict[str, list[tuple[Path, Path]]] = {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val :],
    }

    for split_name, pair_list in splits.items():
        img_dir = output_dir / "images" / split_name
        lbl_dir = output_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, ann_path in tqdm(pair_list, desc=f"Preparing {split_name}"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            stem = img_path.stem
            label_path = lbl_dir / f"{stem}.txt"
            if not convert_and_write_label(img_path, ann_path, w, h, label_path):
                continue

            dest_img = img_dir / f"{stem}{img_path.suffix}"
            shutil.copy2(img_path, dest_img)

    return len(pairs)


def prepare_from_extracted_frames(
    extracted_frames_dir: Path,
    dataset_root: Path,
    output_dir: Path,
    split_ratios: tuple[float, float, float] = (TRAIN_RATIO, VAL_RATIO, TEST_RATIO),
    seed: int = 42,
) -> int:
    """
    Prepare dataset from frames extracted from videos.
    Note: Extracted video frames may not have annotations in standard MIDV layout.
    This function is for when annotations exist (e.g., interpolated or separate source).
    """
    # If extracted frames don't have annotations, we cannot use them for supervised training.
    # Check if there's an annotation source (e.g., same structure as images)
    images_root = extracted_frames_dir
    pairs = find_annotated_images(dataset_root, images_root, driving_license_only=True)
    if not pairs:
        return 0
    return prepare_from_images(
        dataset_root,
        output_dir,
        split_ratios=split_ratios,
        driving_license_only=True,
        seed=seed,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MIDV-500 dataset for YOLO driving license detection"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Path to MIDV-500 dataset root",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset",
        help="Output directory for YOLO dataset",
    )
    parser.add_argument(
        "--extract_frames",
        action="store_true",
        help="Also extract frames from videos (adds to existing images)",
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=5,
        help="Frame extraction interval (when --extract_frames)",
    )
    parser.add_argument(
        "--max_frames_per_video",
        type=int,
        default=None,
        help="Max frames per video (when --extract_frames)",
    )
    parser.add_argument(
        "--all_documents",
        action="store_true",
        help="Include all document types, not just driving licenses",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=TRAIN_RATIO,
        help="Train split ratio",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=VAL_RATIO,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_root.exists():
        print("Dataset root not found. Please download MIDV-500 first.")
        print("  Option 1: pip install midv500 && python -c \"import midv500; midv500.download_dataset('midv500_data/', 'all')\"")
        print("  Option 2: Download from https://www.kaggle.com/datasets/kontheeboonmeeprakob/midv500")
        return

    # Step 1: Optionally extract frames from videos
    if args.extract_frames:
        videos = find_videos(dataset_root, driving_license_only=not args.all_documents)
        extract_out = dataset_root / "images" / "extracted"
        extract_out.mkdir(parents=True, exist_ok=True)
        for vpath in tqdm(videos, desc="Extracting frames"):
            extract_frames_from_video(
                vpath,
                extract_out,
                frame_interval=args.frame_interval,
                max_frames=args.max_frames_per_video,
            )

    # Step 2: Prepare from existing images (primary source)
    n = prepare_from_images(
        dataset_root,
        output_dir,
        split_ratios=(args.train_ratio, args.val_ratio, 1.0 - args.train_ratio - args.val_ratio),
        driving_license_only=not args.all_documents,
        seed=args.seed,
    )

    if n == 0:
        print("No annotated images found. Check dataset structure.")
        return

    # Step 3: Create data.yaml
    create_data_yaml(output_dir)
    print(f"Dataset prepared: {n} samples -> {output_dir}")
    print(f"  data.yaml created at {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
