import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from src.ingest import get_unified_dataloader
from src.preprocess import get_transforms

def test_sanity_check():
    print("--- Starting Pipeline Sanity Check ---")
    
    # 1. Setup (Using the paths you defined)
    RARE_IMG = "data/rareplanes/train/PS-RGB_tiled"
    RARE_GEO = "data/rareplanes/train/geojson_aircraft_tiled"
    XVIEW_IMG = "data/xview/train_images/train_images"
    XVIEW_GEO = "data/xview/train_labels/xView_train.geojson"
    
    os.makedirs("outputs", exist_ok=True)
    transform = get_transforms(train=True)
    
    # 2. Initialize
    dataloader = get_unified_dataloader(
        RARE_IMG, RARE_GEO, XVIEW_IMG, XVIEW_GEO, 
        batch_size=4, transform=transform
    )

    # 3. THE HUNT: Find a sample that actually has objects
    print("Searching for a tile with objects...")
    sample_img, sample_target = None, None
    
    for images, targets in dataloader:
        for i in range(len(targets)):
            if len(targets[i]['boxes']) > 0:
                sample_img = images[i]
                sample_target = targets[i]
                break
        if sample_img is not None:
            break

    if sample_img is None:
        print("Searched several batches but found 0 objects. Check your GeoJSON paths!")
        return

    print(f"Found a sample with {len(sample_target['boxes'])} objects.")

    # 4. Visualization logic
    # Unnormalize for plotting
    img = sample_img.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img * std + mean).clip(0, 1)

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img)
    
    # Fix: New Matplotlib syntax for colormaps
    cmap = plt.get_cmap('tab20')
    
    for box, label in zip(sample_target['boxes'], sample_target['labels']):
        xmin, ymin, xmax, ymax = box.tolist()
        
        # Use label to pick a consistent color
        color = cmap((label.item() % 20) / 20)
        
        rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, 
                                 linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin, f"Cls {label.item()}", color='white', 
                fontsize=8, bbox=dict(facecolor=color, alpha=0.5))

    plt.title(f"Sanity Check: {len(sample_target['boxes'])} Objects Found")
    plt.savefig("outputs/sanity_check.png")
    print("Success! Check 'outputs/sanity_check.png' for boxes.")

if __name__ == "__main__":
    test_sanity_check()