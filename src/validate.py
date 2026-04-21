import os
import json
import numpy as np
import matplotlib.pyplot as plt

def analyze_dataset_statistics(geojson_dir):
    """
    Analyzes the distribution of object sizes and counts to 
    inform model hyperparameter decisions (like anchor box sizes).
    """
    box_widths = []
    box_heights = []
    counts_per_image = []
    
    geojson_files = [f for f in os.listdir(geojson_dir) if f.endswith('.geojson')]
    
    for f_name in geojson_files:
        with open(os.path.join(geojson_dir, f_name), 'r') as f:
            data = json.load(f)
            features = data.get('features', [])
            counts_per_image.append(len(features))
            
            for feature in features:
                # This assumes you've already handled the Lat/Lon -> Pixel conversion
                # If checking raw GeoJSON, use the properties or geom for size estimation
                geom = feature.get('geometry', {}).get('coordinates', [[]])[0]
                if len(geom) > 0:
                    coords = np.array(geom)
                    w = coords[:, 0].max() - coords[:, 0].min()
                    h = coords[:, 1].max() - coords[:, 1].min()
                    box_widths.append(w)
                    box_heights.append(h)

    # Output stats to console
    print(f"--- Dataset Statistical Summary ---")
    print(f"Total Objects Found: {sum(counts_per_image)}")
    print(f"Avg Objects per Image: {np.mean(counts_per_image):.2f}")
    print(f"Avg Box Size: {np.mean(box_widths):.1f}x{np.mean(box_heights):.1f} pixels")
    
    # Optional: Plotting distributions for the 'Deliverables' [cite: 22]
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(counts_per_image, bins=20, color='skyblue', edgecolor='black')
    plt.title("Objects per Tile")
    
    plt.subplot(1, 2, 2)
    plt.scatter(box_widths, box_heights, alpha=0.3, s=1)
    plt.title("Object Dimensions (W vs H)")
    plt.xlabel("Width")
    plt.ylabel("Height")
    
    plt.tight_layout()
    plt.savefig("outputs/dataset_stats.png")
    print("Statistical plots saved to outputs/dataset_stats.png")

if __name__ == "__main__":
    TRAIN_GEOJSON_DIR = "data/rareplanes/train/geojson_aircraft_tiled"
    analyze_dataset_statistics(TRAIN_GEOJSON_DIR)