"""
Albumentations-based augmentation for driving license detection.

Simulates mobile camera capture conditions:
- Rotation: Documents held at angles, phone tilted
- Motion blur: Hand shake, moving capture
- Brightness/contrast: Different lighting, over/underexposure
- Random crop: Partial document views, zoom variation
- Perspective: Document at an angle to camera
- Noise: Sensor noise, compression artifacts

Usage:
    from utils.augmentation import get_training_augmentation
    transform = get_training_augmentation()
    augmented = transform(image=img, bboxes=bboxes, category_ids=classes)
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_training_augmentation(
    image_size: int = 640,
    rotation_limit: int = 15,
    motion_blur_limit: int = 5,
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
) -> A.Compose:
    """
    Build Albumentations pipeline for training.

    All transforms preserve bounding boxes (bbox_params with 'yolo' or 'pascal_voc' format).
    """
    return A.Compose(
        [
            # Rotation: simulates tilted phone or angled document
            A.Rotate(limit=rotation_limit, border_mode=0, p=0.5),
            # Motion blur: hand shake during capture
            A.MotionBlur(blur_limit=motion_blur_limit, p=0.3),
            # Brightness: varying ambient light
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=0.5,
            ),
            # Random crop with scale: zoom variation, partial views
            A.RandomResizedCrop(
                height=image_size,
                width=image_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                p=0.5,
            ),
            # Perspective: document at an angle
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            # Gaussian noise: sensor noise, JPEG artifacts
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            # Horizontal flip (documents can be either orientation)
            A.HorizontalFlip(p=0.5),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


def get_validation_transform(image_size: int = 640) -> A.Compose:
    """Resize only for validation (no augmentation)."""
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
        ),
    )
