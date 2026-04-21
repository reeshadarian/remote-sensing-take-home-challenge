import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
import rasterio


class RarePlanesDataset(Dataset):
    def __init__(self, image_dir, geojson_dir, transform=None):
        """
        Args:
            image_dir (str): Directory with all the tiled images.
            geojson_dir (str): Directory with the corresponding GeoJSON files.
            transform (callable, optional): Albumentations transform pipeline.
        """
        self.image_dir = image_dir
        self.geojson_dir = geojson_dir
        self.transform = transform

        # Filter for actual image files
        valid_extensions = (".png", ".tif", ".tiff", ".jpg")
        self.image_files = [
            f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)
        ]

        # To save time, let's only keep images that have a matching GeoJSON file
        self.valid_samples = []
        for img_name in self.image_files:
            base_name = os.path.splitext(img_name)[0]
            geojson_path = os.path.join(self.geojson_dir, f"{base_name}.geojson")

            if os.path.exists(geojson_path):
                self.valid_samples.append((img_name, geojson_path))

    def _extract_bboxes_from_geojson(self, geojson_path, img_path):
        with open(geojson_path, "r") as f:
            data = json.load(f)

        boxes = []
        labels = []

        # Open the image with rasterio to grab the spatial transform
        with rasterio.open(img_path) as src:
            transform = src.transform
            inverse_transform = ~transform  # Invert to go from Lat/Lon -> Pixels

            for feature in data.get("features", []):
                geom = feature.get("geometry", {})

                if geom and geom.get("type") in ["Polygon", "MultiPolygon"]:
                    if geom["type"] == "Polygon":
                        coords = np.array(geom["coordinates"][0])
                    else:
                        coords = np.array(geom["coordinates"][0][0])

                    # Convert Lat/Lon to Pixel Coordinates
                    pixel_coords = [
                        inverse_transform * (lon, lat) for lon, lat in coords
                    ]
                    pixel_coords = np.array(pixel_coords)

                    xmin, xmax = pixel_coords[:, 0].min(), pixel_coords[:, 0].max()
                    ymin, ymax = pixel_coords[:, 1].min(), pixel_coords[:, 1].max()

                    # Ensure they fit within standard image boundaries (and clip if necessary)
                    xmin, ymin = max(0, xmin), max(0, ymin)

                    if xmax > xmin and ymax > ymin:
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(1)

        return boxes, labels

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        # 1. Path Setup
        img_name, geojson_path = self.valid_samples[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # 2. Load Image and Get Dimensions
        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        # 3. Extract Raw Bounding Boxes and Labels
        # Note: Your _extract_bboxes_from_geojson handles Lat/Lon to Pixel conversion
        boxes_raw, labels_raw = self._extract_bboxes_from_geojson(
            geojson_path, img_path
        )

        # 4. Robust Data Validation: Clipping & Filtering
        if len(boxes_raw) > 0:
            boxes = np.array(boxes_raw)
            labels = np.array(labels_raw)

            # Clip coordinates to ensure they are within [0, width] and [0, height]
            # Albumentations requires x_min/y_min to be >= 0
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)

            # Filter out boxes that are invalid after clipping (zero or negative area)
            # This ensures xmax > xmin and ymax > ymin strictly
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[keep]
            labels = labels[keep]
        else:
            # Handle empty image scenario
            boxes = np.empty((0, 4))
            labels = np.empty((0,))

        # 5. Package for Torchvision Models
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }

        # 6. Apply Preprocessing Transforms (Albumentations)
        if self.transform:
            image_np = np.array(image)
            try:
                transformed = self.transform(
                    image=image_np,
                    bboxes=target["boxes"].tolist(),
                    labels=target["labels"].tolist(),
                )
                image = transformed["image"]
                target["boxes"] = torch.as_tensor(
                    transformed["bboxes"], dtype=torch.float32
                )
                target["labels"] = torch.as_tensor(
                    transformed["labels"], dtype=torch.int64
                )
            except Exception as e:
                # Fallback: If transform fails, log the error and return standard tensors
                print(
                    f" Transform failed on {img_name}: {e}. Returning untransformed."
                )
                from torchvision.transforms import ToTensor

                image = ToTensor()(image)
        else:
            # Standard Fallback
            from torchvision.transforms import ToTensor

            image = ToTensor()(image)

        # Ensure boxes is at least (0, 4) even if empty
        if target["boxes"].shape[0] == 0:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)

        return image, target


class XViewDataset(Dataset):
    def __init__(self, image_dir, geojson_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # xView Aircraft Class IDs: 11 (Fixed-wing), 12 (Small), 13 (Passenger/Cargo), 15 (Helicopter)
        self.aircraft_ids = {11, 12, 13, 15}

        with open(geojson_path, "r") as f:
            data = json.load(f)

        # Group annotations by image_id (e.g., '10.tif')
        self.annotations = {}
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            image_id = props.get("image_id")
            type_id = props.get("type_id")

            # Filter for aircraft and map to Class 1
            if type_id in self.aircraft_ids:
                if image_id not in self.annotations:
                    self.annotations[image_id] = []

                # xView provides pixel coords as "xmin,ymin,xmax,ymax"
                coords = props.get("bounds_imcoords", "").split(",")
                if len(coords) == 4:
                    self.annotations[image_id].append([float(c) for c in coords])

        # Filter for images that exist in the directory
        all_files = set(os.listdir(image_dir))
        self.valid_images = [img for img in self.annotations.keys() if img in all_files]
        print(
            f"xView Ingestion: Found {len(self.valid_images)} images with aircraft."
        )

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        img_name = self.valid_images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        raw_boxes = self.annotations[img_name]

        # Robust clipping for xView
        boxes = np.array(raw_boxes)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)

        # Filter for validity
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        valid_boxes = boxes[keep]
        valid_labels = [1] * len(valid_boxes)

        target = {
            "boxes": torch.tensor(valid_boxes, dtype=torch.float32),
            "labels": torch.tensor(valid_labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }

        if self.transform:
            image_np = np.array(image)
            try:
                transformed = self.transform(
                    image=image_np,
                    bboxes=target["boxes"].tolist(),
                    labels=target["labels"].tolist(),
                )
                image = transformed["image"]
                target["boxes"] = torch.tensor(
                    transformed["bboxes"], dtype=torch.float32
                )
                target["labels"] = torch.tensor(
                    transformed["labels"], dtype=torch.int64
                )
            except Exception as e:
                # Log error and return original (for robustness)
                print(f"Transform error on {img_name}: {e}")
                from torchvision.transforms import ToTensor

                image = ToTensor()(image)

        return image, target


def get_unified_dataloader(
    rare_img_dir,
    rare_geo_dir,
    xview_img_dir,
    xview_geo_dir,
    batch_size=4,
    transform=None,
    return_dataset_only=False,
):
    """Combines RarePlanes and xView into a single training stream."""
    rare_ds = RarePlanesDataset(rare_img_dir, rare_geo_dir, transform=transform)
    xview_ds = XViewDataset(xview_img_dir, xview_geo_dir, transform=transform)

    combined_ds = ConcatDataset([rare_ds, xview_ds])

    if return_dataset_only:
        return combined_ds

    def collate_fn(batch):
        return tuple(zip(*batch))

    return DataLoader(
        combined_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )


if __name__ == "__main__":
    print("Testing Ingestion Pipeline...")

    TRAIN_RARE_IMG = "data/rareplanes/train/PS-RGB_tiled"
    TRAIN_RARE_GEO = "data/rareplanes/train/geojson_aircraft_tiled"
    TRAIN_XVIEW_IMG = "data/xview/train_images/train_images"
    TRAIN_XVIEW_GEO = "data/xview/train_labels/xView_train.geojson"
    try:
        dataloader = get_unified_dataloader(
            TRAIN_RARE_IMG, TRAIN_RARE_GEO, TRAIN_XVIEW_IMG, TRAIN_XVIEW_GEO
        )
        images, targets = next(iter(dataloader))
        print(f"DataLoader instantiated successfully.")
        print(f"Batch size: {len(images)}")
        print(f"Image Tensor Shape: {images[0].shape}")
        print(f"Target Keys: {list(targets[0].keys())}")
        print(f"Bounding Box Tensor Shape: {targets[0]['boxes'].shape}")

    except Exception as e:
        print(f"Pipeline Failed: {e}")
