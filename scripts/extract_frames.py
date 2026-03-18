#!/usr/bin/env python3
"""
Extract frames from MIDV-500 video files.

This script reads video files from the MIDV-500 dataset and extracts frames
at configurable intervals. Frames are saved to the dataset/images directory
structure. Used for building a training dataset from video clips.

Usage:
    python extract_frames.py --dataset_root MIDV500 --frame_interval 5
    python extract_frames.py --dataset_root MIDV500 --frame_interval 10 --max_frames_per_video 50
"""

import argparse
from pathlib import Path
import cv2
from tqdm import tqdm


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

# Video file extensions
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def find_videos(root: Path, driving_license_only: bool = True) -> list[Path]:
    """
    Find all video files in the dataset root.

    Args:
        root: Path to MIDV-500 dataset root (contains document type folders)
        driving_license_only: If True, only include driving license document folders

    Returns:
        List of paths to video files
    """
    videos = []
    for path in root.rglob("*"):
        if path.suffix.lower() in VIDEO_EXTENSIONS and path.is_file():
            if driving_license_only:
                # Check if this video is inside a driving license folder
                parts = path.parts
                for folder in DRIVING_LICENSE_FOLDERS:
                    if folder in parts:
                        videos.append(path)
                        break
            else:
                videos.append(path)
    return videos


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    frame_interval: int = 5,
    max_frames: int | None = None,
) -> list[Path]:
    """
    Extract frames from a single video file.

    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame (1 = every frame)
        max_frames: Maximum frames to extract per video (None = no limit)

    Returns:
        List of paths to saved frame images
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    saved_paths = []
    frame_count = 0
    extracted_count = 0

    # Create unique output subfolder based on video name
    video_stem = video_path.stem
    video_output_dir = output_dir / video_stem
    video_output_dir.mkdir(parents=True, exist_ok=True)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=min(total_frames, max_frames or total_frames), desc=video_stem, leave=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_name = f"{video_stem}_frame_{frame_count:06d}.jpg"
            output_path = video_output_dir / frame_name
            cv2.imwrite(str(output_path), frame)
            saved_paths.append(output_path)
            extracted_count += 1
            pbar.update(1)

            if max_frames and extracted_count >= max_frames:
                break

        frame_count += 1

    cap.release()
    pbar.close()
    return saved_paths


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from MIDV-500 video files for training"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Path to MIDV-500 dataset root (e.g., MIDV500 or midv500_data/midv500)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for extracted frames (default: dataset_root/images/extracted)",
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=5,
        help="Extract every Nth frame (default: 5)",
    )
    parser.add_argument(
        "--max_frames_per_video",
        type=int,
        default=None,
        help="Maximum frames to extract per video (default: no limit)",
    )
    parser.add_argument(
        "--all_documents",
        action="store_true",
        help="Include all document types, not just driving licenses",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    output_dir = Path(args.output_dir) if args.output_dir else dataset_root / "images" / "extracted"
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos(dataset_root, driving_license_only=not args.all_documents)
    if not videos:
        print("No video files found. Check that:")
        print("  1. dataset_root contains MIDV-500 structure (document_type/videos/*.mp4)")
        print("  2. Use --all_documents if you want non-driving-license videos")
        return

    print(f"Found {len(videos)} video(s). Extracting frames...")
    total_saved = 0

    for video_path in tqdm(videos, desc="Videos"):
        paths = extract_frames_from_video(
            video_path=video_path,
            output_dir=output_dir,
            frame_interval=args.frame_interval,
            max_frames=args.max_frames_per_video,
        )
        total_saved += len(paths)

    print(f"Extracted {total_saved} frames to {output_dir}")


if __name__ == "__main__":
    main()
