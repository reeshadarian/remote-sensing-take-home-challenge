import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ingest import get_unified_dataloader
from preprocess import get_transforms
from train import get_model
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader


def denormalize(img_tensor):
    """Reverses ImageNet normalization for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return img_tensor * std + mean


def calculate_quantitative_metrics(model, dataloader, device):
    """
    Calculates mAP, mAP_50, and mAP_75 across the provided dataloader.
    """
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy")

    print("Calculating mAP on evaluation set...")
    with torch.no_grad():
        for images, targets in tqdm(dataloader):
            images = [img.to(device) for img in images]
            outputs = model(images)

            # Format outputs and targets for torchmetrics
            preds = [
                {
                    "boxes": out["boxes"].cpu(),
                    "scores": out["scores"].cpu(),
                    "labels": out["labels"].cpu(),
                }
                for out in outputs
            ]

            target_list = [
                {
                    "boxes": t["boxes"].cpu(),
                    "labels": t["labels"].cpu(),
                }
                for t in targets
            ]

            metric.update(preds, target_list)

    results = metric.compute()

    print("\n--- Quantitative Evaluation Results ---")
    print(f"mAP (0.5:0.95): {results['map']:.4f}")
    print(f"mAP @ 50 IoU:   {results['map_50']:.4f}")
    print(f"mAP @ 75 IoU:   {results['map_75']:.4f}")

    return results


def calculate_error_types(
    model, dataloader, device, iou_threshold=0.5, conf_threshold=0.5
):
    """
    Option 1: Categorizes detections into True Positives (TP),
    False Positives (FP), and False Negatives (FN).
    """
    model.eval()
    total_tp, total_fp, total_fn = 0, 0, 0

    print(f"--- Analyzing Error Types (IoU > {iou_threshold}) ---")
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Analyzing Errors"):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i in range(len(outputs)):
                pred_boxes = outputs[i]["boxes"].cpu()
                pred_scores = outputs[i]["scores"].cpu()
                gt_boxes = targets[i]["boxes"].cpu()

                # Filter by confidence
                mask = pred_scores >= conf_threshold
                pred_boxes = pred_boxes[mask]

                if len(gt_boxes) == 0:
                    total_fp += len(pred_boxes)
                    continue
                if len(pred_boxes) == 0:
                    total_fn += len(gt_boxes)
                    continue

                # Calculate IoU matrix [num_preds, num_gts]
                iou_matrix = box_iou(pred_boxes, gt_boxes)

                # Match predictions to ground truths
                matched_gt = set()
                tp_in_sample = 0
                for iou_row in iou_matrix:
                    best_gt_idx = iou_row.argmax().item()
                    if (
                        iou_row[best_gt_idx] >= iou_threshold
                        and best_gt_idx not in matched_gt
                    ):
                        tp_in_sample += 1
                        matched_gt.add(best_gt_idx)

                total_tp += tp_in_sample
                total_fp += len(pred_boxes) - tp_in_sample
                total_fn += len(gt_boxes) - len(matched_gt)

    print(f"Results: TP: {total_tp} | FP: {total_fp} | FN: {total_fn}")
    return {"TP": total_tp, "FP": total_fp, "FN": total_fn}


def plot_precision_recall_curve(
    model, dataloader, device, output_path="outputs/pr_curve.png"
):
    """
    Option 3: Generates a PR curve by sweeping confidence thresholds.
    """
    model.eval()
    all_scores = []
    all_matches = []  # 1 for TP, 0 for FP
    total_gts = 0

    print("--- Generating Precision-Recall Curve ---")
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Gathering PR Data"):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i in range(len(outputs)):
                pred_boxes = outputs[i]["boxes"].cpu()
                pred_scores = outputs[i]["scores"].cpu()
                gt_boxes = targets[i]["boxes"].cpu()
                total_gts += len(gt_boxes)

                if len(pred_boxes) == 0:
                    continue
                if len(gt_boxes) == 0:
                    for s in pred_scores:
                        all_scores.append(s.item())
                        all_matches.append(0)
                    continue

                iou_matrix = box_iou(pred_boxes, gt_boxes)
                matched_gt = set()

                # Sort by score to process highest confidence first
                indices = torch.argsort(pred_scores, descending=True)
                for idx in indices:
                    all_scores.append(pred_scores[idx].item())
                    best_gt_idx = iou_matrix[idx].argmax().item()
                    if (
                        iou_matrix[idx][best_gt_idx] >= 0.5
                        and best_gt_idx not in matched_gt
                    ):
                        all_matches.append(1)
                        matched_gt.add(best_gt_idx)
                    else:
                        all_matches.append(0)

    # Calculate Precision and Recall at each threshold
    indices = np.argsort(all_scores)[::-1]
    all_matches = np.array(all_matches)[indices]
    tps = np.cumsum(all_matches)
    fps = np.cumsum(1 - all_matches)

    precision = tps / (tps + fps)
    recall = tps / total_gts

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="blue", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Airplane Detection)")
    plt.grid(True)
    plt.savefig(output_path)
    print(f"PR Curve saved to {output_path}")


@torch.no_grad()
def run_evaluation(
    model_path,
    rare_img_dir,
    rare_geo_dir,
    xview_img_dir,
    xview_geo_dir,
    output_dir="outputs/eval/",
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. Load Test Data
    transform = get_transforms(train=False)
    dataloader = get_unified_dataloader(
        rare_img_dir=rare_img_dir,
        rare_geo_dir=rare_geo_dir,
        xview_img_dir=xview_img_dir,
        xview_geo_dir=xview_geo_dir,
        batch_size=1,
        transform=transform,
        shuffle=True,
    )

    print(f"Running inference...")

    for i, (images, targets) in enumerate(dataloader):
        if i >= 5:
            break  # Only visualize 5 samples to save time

        images = [img.to(device) for img in images]
        outputs = model(images)

        # Process the first image in the batch
        img = denormalize(images[0].cpu()).permute(1, 2, 0).numpy().clip(0, 1)
        boxes = outputs[0]["boxes"].cpu().numpy()
        scores = outputs[0]["scores"].cpu().numpy()

        # Filter by a confidence threshold
        threshold = 0.5
        keep = scores > threshold
        boxes = boxes[keep]
        scores = scores[keep]

        # 3. Visualization
        fig, ax = plt.subplots(1, figsize=(12, 12))
        ax.imshow(img)

        # Draw Predicted Boxes (Red)
        for box, score in zip(boxes, scores):
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                xmin,
                ymin,
                f"Plane: {score:.2f}",
                color="white",
                fontsize=8,
                bbox=dict(facecolor="red", alpha=0.5),
            )

        # Draw Ground Truth Boxes (Green) for comparison
        gt_boxes = targets[0]["boxes"].numpy()
        for box in gt_boxes:
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                linewidth=1,
                edgecolor="lime",
                facecolor="none",
                linestyle="--",
            )
            ax.add_patch(rect)

        plt.title(
            f"Inference Results - Sample {i}\nRed: Prediction (Conf > {threshold}), Green: Ground Truth"
        )
        plt.axis("off")
        save_path = os.path.join(output_dir, f"eval_sample_{i}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved visualization to {save_path}")

def main():
    # --- Configuration ---
    MODEL_PATH = "outputs/model_epoch_2.pth"
    MAX_EVAL_SAMPLES = 200  # Truncate for speed 
    BATCH_SIZE = 4
    
    VAL_RARE_IMG = "data/rareplanes/train/PS-RGB_tiled"
    VAL_RARE_GEO = "data/rareplanes/train/geojson_aircraft_tiled"
    VAL_XVIEW_IMG = "data/xview/train_images/train_images"
    VAL_XVIEW_GEO = "data/xview/train_labels/xView_train.geojson"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)

    # 2. Setup Data
    transform = get_transforms(train=False)
    
    # Get the initial dataloader to access the full dataset object
    full_dataloader = get_unified_dataloader(
        VAL_RARE_IMG,
        VAL_RARE_GEO,
        VAL_XVIEW_IMG,
        VAL_XVIEW_GEO,
        batch_size=BATCH_SIZE,
        transform=transform,
    )
    full_dataset = full_dataloader.dataset

    # 3. Truncate Data [cite: 29]
    if len(full_dataset) > MAX_EVAL_SAMPLES:
        print(f"Truncating evaluation data: {len(full_dataset)} -> {MAX_EVAL_SAMPLES} samples.")
        indices = np.random.choice(len(full_dataset), MAX_EVAL_SAMPLES, replace=False)
        eval_subset = Subset(full_dataset, indices)
    else:
        eval_subset = full_dataset

    # Create the truncated DataLoader
    dataloader = DataLoader(
        eval_subset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=lambda x: tuple(zip(*x))
    )

    # 4. RUN QUANTITATIVE EVALUATION
    results = calculate_quantitative_metrics(model, dataloader, device)

    print("\n--- Final Metrics ---")
    print(f"mAP @ [0.5:0.95]: {results['map'].item():.4f}")
    print(f"mAP @ 50 IoU:     {results['map_50'].item():.4f}")

    # 5. Error Type Analysis
    errors = calculate_error_types(model, dataloader, device)

    # 6. PR Curve
    plot_precision_recall_curve(model, dataloader, device)

    # 7. Summary
    precision = (
        errors["TP"] / (errors["TP"] + errors["FP"])
        if (errors["TP"] + errors["FP"]) > 0
        else 0
    )
    recall = (
        errors["TP"] / (errors["TP"] + errors["FN"])
        if (errors["TP"] + errors["FN"]) > 0
        else 0
    )
    print(f"\nFinal Summary: Precision={precision:.2f}, Recall={recall:.2f}")

if __name__ == "__main__":
    main()