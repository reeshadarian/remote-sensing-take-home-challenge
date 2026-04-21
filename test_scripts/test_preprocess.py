import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from src.ingest import get_unified_dataloader
from src.preprocess import get_transforms


def unnormalize(tensor):
    """Reverts the ImageNet normalization so we can plot it."""
    # Reshape mean and std to match image dimensions (C, H, W)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Unnormalize: image * std + mean
    tensor = tensor * std + mean
    return tensor


def check_preprocessing():
    # 1. Grab the training transforms (includes resizing and normalization)
    transform = get_transforms(train=True)
    TRAIN_RARE_IMG = "data/rareplanes/train/PS-RGB_tiled"
    TRAIN_RARE_GEO = "data/rareplanes/train/geojson_aircraft_tiled"
    TRAIN_XVIEW_IMG = "data/xview/train_images/train_images"
    TRAIN_XVIEW_GEO = "data/xview/train_labels/xView_train.geojson"

    # 2. Load the DataLoader WITH the transforms applied
    dataloader = get_unified_dataloader(
        TRAIN_RARE_IMG,
        TRAIN_RARE_GEO,
        TRAIN_XVIEW_IMG,
        TRAIN_XVIEW_GEO,
        batch_size=1,
        transform=transform,
    )

    images, targets = next(iter(dataloader))

    # 3. Unnormalize and convert PyTorch tensor back to Numpy for Matplotlib
    img_tensor = unnormalize(images[0])
    img_np = (
        img_tensor.permute(1, 2, 0).numpy().clip(0, 1)
    )  # Clip keeps values in [0, 1] range
    boxes = targets[0]["boxes"].numpy()

    # 4. Plot the image and bounding boxes
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img_np)

    for box in boxes:
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        # Using a bright cyan color to stand out against the satellite imagery
        rect = patches.Rectangle(
            (xmin, ymin), width, height, linewidth=2, edgecolor="cyan", facecolor="none"
        )
        ax.add_patch(rect)

    plt.title(f"Preprocessing Check: 512x512 image with {len(boxes)} Planes")
    plt.savefig("preprocess_check.png")
    print("Saved preprocessed visual test to preprocess_check.png")


if __name__ == "__main__":
    check_preprocessing()
