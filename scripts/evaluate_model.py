#!/usr/bin/env python3
"""
Evaluate trained driving license detection model.

Computes mAP@50, mAP@50-95, precision, recall, F1 score.
Generates precision-recall curve and confusion matrix.

Usage:
    python evaluate_model.py --weights runs/license_detection/weights/best.pt --data dataset/data.yaml
"""

import argparse
from pathlib import Path
import json

from ultralytics import YOLO
import numpy as np

# Add project root for utils
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.visualization import plot_precision_recall_curve, plot_confusion_matrix


def main():
    parser = argparse.ArgumentParser(description="Evaluate driving license detection model")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained model weights (e.g., runs/.../best.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data.yaml",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports",
        help="Directory for evaluation reports and visualizations",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test"],
        default="val",
        help="Which split to evaluate on",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.weights}")
    model = YOLO(args.weights)

    # Run validation - returns validator with metrics
    results = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
    )

    # Extract metrics: model.val() returns object with results.box (map50, map, mp, mr)
    metrics = getattr(results, "box", results)
    map50 = float(getattr(metrics, "map50", 0) or 0)
    map50_95 = float(getattr(metrics, "map", 0) or 0)
    precision = float(getattr(metrics, "mp", 0) or 0)
    recall = float(getattr(metrics, "mr", 0) or 0)

    # F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    report = {
        "mAP@50": map50,
        "mAP@50-95": map50_95,
        "precision": precision,
        "recall": recall,
        "F1": f1,
    }

    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    for k, v in report.items():
        print(f"  {k}: {v:.4f}")
    print("=" * 50)

    # Save JSON report
    report_path = output_dir / "evaluation_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")

    # Precision-Recall curve (if metrics contain p, r arrays)
    try:
        det_metrics = getattr(results, "metrics", getattr(results, "box", None))
        if det_metrics:
            p_vals = getattr(det_metrics, "p", None)
            r_vals = getattr(det_metrics, "r", None)
            if p_vals is not None and r_vals is not None:
                p_arr = np.atleast_1d(np.array(p_vals)).flatten()
                r_arr = np.atleast_1d(np.array(r_vals)).flatten()
                if len(p_arr) > 1 and len(r_arr) > 1:
                    plot_precision_recall_curve(
                        p_arr, r_arr, map50_95,
                        save_path=output_dir / "precision_recall_curve.png",
                    )
    except Exception as e:
        print(f"Could not generate P-R curve: {e}")

    # Confusion matrix (if available from results.metrics)
    try:
        det_metrics = getattr(results, "metrics", None)
        if det_metrics:
            cm_obj = getattr(det_metrics, "confusion_matrix", None)
            if cm_obj is not None and hasattr(cm_obj, "matrix"):
                cm = np.array(cm_obj.matrix)
                plot_confusion_matrix(
                    cm,
                    ["background", "driving_license"],
                    save_path=output_dir / "confusion_matrix.png",
                )
    except Exception as e:
        print(f"Could not generate confusion matrix: {e}")

    # Target check
    target_map50 = 0.95
    if map50 >= target_map50:
        print(f"\n✓ mAP@50 ({map50:.2%}) meets target (>95%)")
    else:
        print(f"\n✗ mAP@50 ({map50:.2%}) below target (>95%). Consider more epochs or augmentation.")


if __name__ == "__main__":
    main()
