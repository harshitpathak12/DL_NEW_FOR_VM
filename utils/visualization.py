"""
Visualization utilities for driving license detection.

- Draw bounding boxes on images (in-place for speed)
- Plot precision-recall curves
- Render confusion matrices
"""

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def draw_bbox(
    image: np.ndarray,
    bbox: Sequence[float],
    label: str = "driving_license",
    confidence: float | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding box on image in-place. Returns same array."""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    text = label
    if confidence is not None:
        text += f" {confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
    cv2.putText(
        image, text, (x1, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
    )
    return image


def draw_validation_status(
    image: np.ndarray,
    label: str,
    reason: str = "",
) -> np.ndarray:
    """Draw a prominent status banner at top of frame. Modifies in-place."""
    h, w = image.shape[:2]
    text = label.upper()
    if reason:
        text += f" - {reason[:40]}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.8, 2)
    pad = 10
    y2 = th + 2 * pad
    cv2.rectangle(image, (0, 0), (w, y2), (0, 0, 0), -1)
    color = (
        (0, 255, 0) if label.lower() == "valid"
        else (0, 0, 255) if label.lower() == "invalid"
        else (200, 200, 200)
    )
    cv2.putText(image, text, (pad, th + pad), font, 0.8, color, 2)
    return image


def plot_precision_recall_curve(
    precisions: np.ndarray,
    recalls: np.ndarray,
    ap: float,
    save_path: Path | None = None,
) -> None:
    """Plot precision-recall curve and save to file."""
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.fill_between(recalls, precisions, alpha=0.2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (AP = {ap:.3f})")
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    save_path: Path | None = None,
) -> None:
    """Plot confusion matrix as heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
