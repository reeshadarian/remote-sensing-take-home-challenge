import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def get_transforms(train=True):
    """
    Returns albumentations transforms for object detection.
    Faster R-CNN natively expects 'pascal_voc' format: [xmin, ymin, xmax, ymax].
    """
    if train:
        # We keep augmentations light to save time during the challenge
        transform = A.Compose(
            [
                A.Resize(height=512, width=512),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )
    else:
        transform = A.Compose(
            [
                A.Resize(height=512, width=512),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )

    return transform


def apply_transforms(transform, image_pil, target):
    """
    Wrapper to convert PIL images to NumPy arrays for Albumentations,
    apply the transform, and repackage for PyTorch.
    """
    # Albumentations expects numpy arrays (H, W, C)
    image_np = np.array(image_pil)

    # Extract boxes and labels
    boxes = target["boxes"].numpy().tolist()
    labels = target["labels"].numpy().tolist()

    # Apply transformations
    transformed = transform(image=image_np, bboxes=boxes, labels=labels)

    # Repackage into PyTorch format
    import torch

    target["boxes"] = torch.tensor(transformed["bboxes"], dtype=torch.float32)
    target["labels"] = torch.tensor(transformed["labels"], dtype=torch.int64)

    return transformed["image"], target
