#!/usr/bin/env python3
"""
Export trained model to various formats.

Formats:
- PyTorch (.pt): Native format, best for continued training or Python inference.
- ONNX: Cross-platform, optimized for deployment (CPU/GPU, various runtimes).
- TensorRT: NVIDIA GPU-only, fastest inference (when deployment is on GPU).

Usage:
    python export_model.py --weights runs/.../best.pt --format onnx
    python export_model.py --weights runs/.../best.pt --format all
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


# Format descriptions for README
FORMAT_NOTES = """
Format usage:
- PyTorch (.pt): Use for Python inference, fine-tuning, or when you need flexibility.
- ONNX: Use for production deployment, cross-platform (Python, C++, mobile), TensorFlow Lite.
- TensorRT: Use when deploying on NVIDIA GPU and need maximum inference speed.
"""


def main():
    parser = argparse.ArgumentParser(
        description="Export driving license detection model",
        epilog=FORMAT_NOTES,
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained model (best.pt)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=["pt", "onnx", "engine", "all"],
        help="Export format: pt (PyTorch), onnx, engine (TensorRT), all",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size for export",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: same as weights)",
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model = YOLO(str(weights_path))

    # Ultralytics export formats: torchscript, onnx, engine (TensorRT), etc.
    # .pt is already PyTorch format - no export needed
    if args.format == "pt":
        print("Model is already in PyTorch (.pt) format. No export needed.")
        return

    formats = ["onnx"] if args.format == "onnx" else (["onnx", "engine"] if args.format == "all" else [args.format])
    format_map = {"engine": "engine"}  # TensorRT

    for fmt in formats:
        export_format = format_map.get(fmt, fmt)
        try:
            export_path = model.export(
                format=export_format,
                imgsz=args.imgsz,
            )
            print(f"Exported {fmt} to {export_path}")
        except Exception as e:
            print(f"Export {fmt} failed: {e}")


if __name__ == "__main__":
    main()
