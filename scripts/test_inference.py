#!/usr/bin/env python3
"""
Test inference with trained driving license detection model.

Architecture:
- YOLOv8s for card detection
- Qwen vision-language model for OCR + validity judgement (local, no external API)

Supports:
- Single image
- Folder of images
- Webcam (real-time)

Output format per detection:
{
  "bbox": [x1, y1, x2, y2],
  "confidence": float,
  "class": "driving_license",
  "ocr_text": "...",                    # from Qwen (if --qwen)
  "ocr_lines": [{"text", "confidence"}],
  "validation_label": "valid|invalid|unknown",
  "validation_confidence": float,
  "validation_reason": "short explanation"
}

Usage:
    python test_inference.py --weights weights/best.pt --source test_images/
    python test_inference.py --weights weights/best.pt --source 0  # webcam
    python test_inference.py --weights weights/best.pt --source img.jpg --qwen
"""

import argparse
import json
from pathlib import Path
import time

import cv2
from ultralytics import YOLO

# Add project root for utils
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.visualization import draw_bbox


CLASS_NAME = "driving_license"


def format_detections(results) -> list[dict]:
    """Convert YOLO results to specified output format."""
    detections = []
    if not results or len(results) == 0:
        return detections

    result = results[0]
    if result.boxes is None:
        return detections

    for box in result.boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        detections.append({
            "bbox": xyxy.tolist(),
            "confidence": conf,
            "class": CLASS_NAME,
        })
    return detections


def run_inference(model: YOLO, source: str | Path, imgsz: int = 640) -> list[dict]:
    """Run inference and return detections in standard format."""
    results = model.predict(source=source, imgsz=imgsz, verbose=False)
    return format_detections(results)


def main():
    parser = argparse.ArgumentParser(description="Test driving license detection inference")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained model (best.pt)",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image path, folder path, or 0 for webcam",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Save annotated images to this directory",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display results (for single image or webcam)",
    )
    parser.add_argument(
        "--json_output",
        action="store_true",
        help="Print detections as JSON to stdout",
    )
    parser.add_argument(
        "--qwen",
        action="store_true",
        help="Run Qwen OCR + validation on detected license crops",
    )
    args = parser.parse_args()

    use_pipeline = args.qwen
    model = YOLO(args.weights)
    def run_detection(frame, model, imgsz):
        if use_pipeline:
            from utils.pipeline import run_pipeline
            detections, out_img = run_pipeline(
                frame,
                model,
                imgsz=imgsz,
                run_ocr_enabled=True,
                run_vit_enabled=False,
            )
            return detections, out_img
        results = model.predict(frame, imgsz=imgsz, verbose=False)
        detections = format_detections(results)
        out_img = frame.copy()
        for d in detections:
            out_img = draw_bbox(out_img, d["bbox"], confidence=d["confidence"])
        return detections, out_img

    def print_detection_info(detections, source_label=""):
        if not use_pipeline or not detections:
            return
        for i, d in enumerate(detections):
            if source_label:
                print(f"  [{source_label}] detection {i+1}: conf={d['confidence']:.2f}")
            if d.get("ocr_text"):
                print(f"    OCR: {d['ocr_text'][:200]}{'...' if len(d.get('ocr_text','')) > 200 else ''}")
            if d.get("validation_label") and d["validation_label"] != "unknown":
                print(f"    Validation: {d['validation_label']} ({d.get('validation_confidence', 0):.2f})")

    if args.source == "0" or args.source == "webcam":
        # Webcam mode
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open webcam")
            return

        print("Webcam inference. Press 'q' to quit.")
        total_time = 0
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t0 = time.perf_counter()
            detections, out_img = run_detection(frame, model, args.imgsz)
            t1 = time.perf_counter()
            total_time += t1 - t0
            frame_count += 1

            # FPS
            fps = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0
            cv2.putText(
                out_img, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
            )
            cv2.imshow("License Detection", out_img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        if frame_count > 0:
            avg_ms = (total_time / frame_count) * 1000
            print(f"Average inference: {avg_ms:.1f} ms ({1000/avg_ms:.1f} FPS)")

    elif Path(args.source).is_file():
        # Single image
        img = cv2.imread(str(args.source))
        if img is None:
            print(f"Cannot read image: {args.source}")
        else:
            detections, out_img = run_detection(img, model, args.imgsz)
            if args.json_output:
                print(json.dumps(detections, indent=2))
            print_detection_info(detections, Path(args.source).name)

            if args.show or args.output_dir:
                if args.output_dir:
                    out_path = Path(args.output_dir) / f"det_{Path(args.source).name}"
                    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_path), out_img)
                    print(f"Saved to {out_path}")
                if args.show:
                    cv2.imshow("Result", out_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

    elif Path(args.source).is_dir():
        # Folder of images
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}
        images = [p for p in Path(args.source).rglob("*") if p.suffix.lower() in exts]
        output_dir = Path(args.output_dir) if args.output_dir else Path(args.source) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        all_results = {}
        total_time = 0

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            t0 = time.perf_counter()
            detections, out_img = run_detection(img, model, args.imgsz)
            total_time += time.perf_counter() - t0
            all_results[str(img_path)] = detections
            print_detection_info(detections, img_path.name)
            out_path = output_dir / f"det_{img_path.name}"
            cv2.imwrite(str(out_path), out_img)

        n = len(images)
        avg_ms = (total_time / n) * 1000 if n else 0
        print(f"Processed {n} images. Avg inference: {avg_ms:.1f} ms")
        print(f"Annotated images saved to {output_dir}")
        if args.json_output:
            print(json.dumps(all_results, indent=2))

    else:
        print(f"Invalid source: {args.source}")


if __name__ == "__main__":
    main()
