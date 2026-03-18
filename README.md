# Driving License Detection - ML Training Pipeline

A complete machine learning project for training and fine-tuning a driving license detection model using the **MIDV-500** dataset and **YOLOv8**.

## Project Goal

Train a computer vision model that:

1. Detects driving license cards in images (YOLOv8s)
2. Returns bounding box coordinates `[x1, y1, x2, y2]` and confidence
3. Crops the detected card and sends it to a **local Qwen vision-language model**
4. Qwen performs **OCR + card validity judgement** (no external API)
5. Works well with mobile camera images

End-to-end: **Camera → YOLOv8s (card detection) → Crop → Qwen (OCR + validation)**

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download MIDV-500 dataset
# Option A: Using midv500 package
pip install midv500
python -c "import midv500; midv500.download_dataset('midv500_data/', 'midv500')"

# Option B: Download from Kaggle
# https://www.kaggle.com/datasets/kontheeboonmeeprakob/midv500

# 3. Prepare dataset (filter driving licenses, convert to YOLO format)
cd driver_license_detection
python scripts/prepare_dataset.py --dataset_root ../midv500_data/midv500 --output_dir dataset

# 4. Train YOLOv8s card detector
python scripts/train_model.py --data dataset/data.yaml --model yolov8s --epochs 50 --batch 16 --device cuda

# 5. Evaluate detector
python scripts/evaluate_model.py --weights runs/license_detection/weights/best.pt --data dataset/data.yaml

# 6. Test end-to-end pipeline (YOLOv8s + Qwen OCR/validation)
python scripts/test_inference.py --weights runs/license_detection/weights/best.pt --source test_images/ --qwen --output_dir output
python scripts/test_inference.py --weights runs/license_detection/weights/best.pt --source image.jpg --qwen --show

# 7. Run WebSocket server (YOLOv8s + Qwen)
python main.py
# Server: ws://localhost:5001/stream?qwen=1

# 8. Run local webcam client (on any machine with network access to server)
python client.py  # uses SERVER_WS_URL in client.py to connect
```

---

## Project Structure

```
driver_license_detection/
├── api/
│   └── websocket.py            # WebSocket /stream (detection + optional Qwen OCR + validation)
├── configs/
│   ├── training_config.yaml   # Training hyperparameters
│   └── pipeline_config.yaml   # Pipeline options (thresholds, image size, etc.)
├── dataset/                    # Output: YOLO-format dataset
│   ├── images/ train/ val/ test/
│   ├── labels/ train/ val/ test/
│   └── data.yaml
├── models/                     # (YOLO training, optional extra models)
├── weights/                    # best.pt (YOLOv8s weights; download/produce yourself)
├── scripts/
│   ├── test_inference.py      # Inference with optional --qwen (OCR + validation)
│   ├── train_model.py, evaluate_model.py, export_model.py, ...
│   └── ...
├── utils/
│   ├── image_utils.py         # crop_bbox helper
│   ├── qwen_ocr.py            # Qwen OCR (text only, local)
│   ├── license_rules.py       # Rule-based Indian DL validation using Qwen OCR text
│   ├── pipeline.py            # YOLOv8s detection → Crop → Qwen OCR → licence_rules
│   ├── augmentation.py, dataset_validator.py, visualization.py
│   └── ...
├── main.py                     # WebSocket server (port 5001)
├── requirements.txt
└── README.md
```

---

## Dataset: MIDV-500

- **500 video clips** across **50 document types**
- Includes **13 driving license** types (Austria, Germany, Spain, Finland, etc.)
- Ground truth: document quadrilateral corners (4-point polygon)
- Source: [Kaggle](https://www.kaggle.com/datasets/kontheeboonmeeprakob/midv500) | [midv500 package](https://pypi.org/project/midv500/)

### MIDV-500 Structure

```
midv500/
  02_aut_drvlic_new/     # Austria driving license
    ground_truth/        # JSON annotations (quad corners)
    images/              # TIF images
    videos/              # Video clips
  12_deu_drvlic_new/     # German driving license
  ...
```

---

## Scripts

### Frame Extraction

```bash
python scripts/extract_frames.py --dataset_root MIDV500 --frame_interval 5 --max_frames_per_video 100
```

Extracts frames from videos at configurable intervals. By default, only driving license document folders are processed.

### Annotation Conversion

Converts MIDV polygon annotations to YOLO format:

- **Input**: `[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]` (quadrilateral)
- **Output**: `class_id x_center y_center width height` (normalized 0–1)

### Data Augmentation (Albumentations)

Augmentations in `utils/augmentation.py` simulate mobile capture:

| Augmentation    | Purpose                                              |
|-----------------|------------------------------------------------------|
| Rotation        | Document held at angles, phone tilted                 |
| Motion blur     | Hand shake during capture                            |
| Brightness/contrast | Varying lighting, over/underexposure             |
| Random crop     | Partial document views, zoom variation              |
| Perspective     | Document at an angle to camera                       |
| Noise           | Sensor noise, JPEG compression artifacts             |

YOLOv8 includes built-in augmentation; Albumentations can be used for offline dataset expansion.

### Training

```bash
python scripts/train_model.py \
  --data dataset/data.yaml \
  --model yolov8s \
  --epochs 50 \
  --batch 16 \
  --imgsz 640 \
  --device cuda
```

**Model options**: `yolov8n` (fastest), `yolov8s` (balanced), `yolov8m` (higher accuracy)

**TensorBoard**: Logs in `runs/`. View with `tensorboard --logdir runs`

### Evaluation

```bash
python scripts/evaluate_model.py \
  --weights runs/license_detection/weights/best.pt \
  --data dataset/data.yaml \
  --output_dir reports
```

Metrics: mAP@50, mAP@50-95, precision, recall, F1. Generates precision-recall curve and confusion matrix.

### Inference

```bash
# Single image
python scripts/test_inference.py --weights weights/best.pt --source image.jpg --show

# Folder
python scripts/test_inference.py --weights weights/best.pt --source test_images/ --output_dir output

# Webcam
python scripts/test_inference.py --weights weights/best.pt --source 0 --show

# With OCR (PaddleOCR) and ViT validation
python scripts/test_inference.py --weights weights/best.pt --source image.jpg --ocr --validate --vit_weights weights/vit_validator.pt --json_output
```

Output format (per detection):

```json
{
  "bbox": [x1, y1, x2, y2],
  "confidence": 0.95,
  "class": "driving_license",
  "ocr_text": "extracted text from crop",
  "ocr_lines": [{"text": "...", "confidence": 0.9}],
  "validation_label": "valid",
  "validation_confidence": 0.88
}
```

**Indian DL validation:** With `--ocr`, rule-based validation runs on the OCR text and marks the licence as **VALID** or **INVALID** based on Indian format (e.g. "Indian Union Driving Licence", "Issued by Government of [State]", licence number like BR22 20250006557). Install **PaddleOCR** (`pip install paddlepaddle paddleocr`) so OCR can read the card; otherwise the status stays "unknown" and you may see "Could not read text (install PaddleOCR?)".

**ViT weights:** Optional. The validator can use a trained checkpoint at `weights/vit_validator.pt` (timm ViT with 2-class head). Rule-based Indian DL validation does not require ViT.

### WebSocket Stream (Real-Time)

Same pattern as **driver_safety_system**: client sends JPEG frames, server returns annotated frames (and optionally JSON with OCR + validation).

```bash
python main.py
# Server runs on http://localhost:5001
# WebSocket: ws://localhost:5001/stream
```

**Query params:**

| Param       | Description |
|------------|-------------|
| `weights`  | YOLO weights path (default `weights/best.pt`) |
| `ocr=1`    | Enable PaddleOCR on detected license crops |
| `validate=1` | Enable ViT validation (requires `vit_weights`) |
| `vit_weights` | Path to ViT checkpoint (default `weights/vit_validator.pt`) |

**Protocol:**
- Client sends raw JPEG frame bytes (binary) per message.
- If `ocr=1` or `validate=1`: server sends **text message** (JSON with `detections`: bbox, ocr_text, validation_label) then **binary** (annotated JPEG).
- Otherwise: server sends only binary (annotated JPEG).

**Example client (browser):**
```javascript
const ws = new WebSocket('ws://localhost:5001/stream?ocr=1&validate=1');
ws.binaryType = 'arraybuffer';
ws.onmessage = (e) => {
  if (typeof e.data === 'string') console.log('Detections:', JSON.parse(e.data));
  else /* e.data is annotated JPEG */;
};
```

### Model Export

```bash
# ONNX (cross-platform)
python scripts/export_model.py --weights best.pt --format onnx

# All formats
python scripts/export_model.py --weights best.pt --format all
```

| Format    | Use case                                              |
|----------|--------------------------------------------------------|
| PyTorch (.pt) | Python inference, fine-tuning, flexibility           |
| ONNX     | Production, cross-platform, mobile runtimes            |
| TensorRT | NVIDIA GPU deployment, maximum inference speed         |

---

## Performance Targets

- **mAP@50**: > 95%
- **Inference**: < 30 ms per frame on GPU

---

## Dependencies

- ultralytics, torch, torchvision
- opencv-python, albumentations
- numpy, pyyaml, tqdm, matplotlib, tensorboard, seaborn
- **OCR / ViT:** paddlepaddle, paddleocr, timm (see requirements.txt)

---

## Pipeline Overview

**Training:** MIDV-500 → prepare_dataset → train_model (YOLOv8) → evaluate_model → export_model.

**Inference / WebSocket:** Frame → YOLO detection → for each bbox: **crop** → **PaddleOCR** (text) + **ViT** (valid/invalid) → annotated frame + optional JSON.

```
Frame → YOLO → Bboxes → Crop → OCR (PaddleOCR) → text
                    └→ ViT validator → valid/invalid
```

ViT validator weights (`weights/vit_validator.pt`) must be trained or provided separately (e.g. timm ViT with 2-class head on valid/invalid license crops).
