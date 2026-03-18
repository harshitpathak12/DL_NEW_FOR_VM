#!/usr/bin/env python3
"""
Train YOLOv8 model for driving license detection.

Loads pretrained YOLOv8 weights, applies augmentation, trains on the prepared
dataset, and saves checkpoints. Uses TensorBoard for experiment tracking.

Usage:
    python train_model.py --data dataset/data.yaml --epochs 50 --batch 16
    python train_model.py --data dataset/data.yaml --model yolov8m --device cuda
"""

import argparse
from pathlib import Path
import yaml

from ultralytics import YOLO


def load_config(config_path: Path | None) -> dict:
    """Load training config from YAML if provided."""
    if config_path and config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for driving license detection"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data.yaml (e.g., dataset/data.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s",
        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
        help="YOLOv8 model size: n (fastest), s (balanced), m (higher accuracy)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cuda, cpu, or 0, 1 for specific GPU",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs",
        help="Project directory for saving runs",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="license_detection",
        help="Experiment name",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training_config.yaml for additional options",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    args = parser.parse_args()

    # Load optional config
    config = {}
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            config = load_config(config_path)

    # Build model string (e.g., yolov8s.pt)
    model_name = f"{args.model}.pt"

    # Load pretrained YOLOv8 model
    # Ultralytics auto-downloads weights on first use
    print(f"Loading {model_name} (pretrained on COCO)...")
    model = YOLO(model_name)

    # Training with TensorBoard logging
    # Augmentation is built into YOLOv8 - we can extend via config
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        lr0=args.lr0,
        exist_ok=True,
        resume=args.resume,
        # TensorBoard logging (enabled by default in ultralytics)
        # Logs go to runs/name/
    )

    print(f"Training complete. Best model saved to {results.save_dir}")
    print("Export best model with: python -c \"from ultralytics import YOLO; YOLO('runs/license_detection/weights/best.pt').export(format='onnx')\"")


if __name__ == "__main__":
    main()
