import json
import os

def peek_geojson(path, label):
    if not os.path.exists(path):
        print(f"{label} path not found: {path}")
        return
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    features = data.get('features', [])
    if not features:
        print(f" {label} GeoJSON exists but has NO features.")
        return
    
    print(f"\n--- {label} Metadata Peek ---")
    print(f"Keys available in properties: {list(features[0]['properties'].keys())}")
    print(f"Sample Geometry Type: {features[0]['geometry']['type']}")
    # Print the first feature's coordinates to check the scale
    print(f"Sample Coordinates: {features[0]['geometry']['coordinates'][0][0]}")

# Test your specific paths
peek_geojson("data/rareplanes/train/geojson_aircraft_tiled/80_104001003B099300_tile_46.geojson", "RarePlanes")
peek_geojson("data/xview/train_labels/xView_train.geojson", "xView")