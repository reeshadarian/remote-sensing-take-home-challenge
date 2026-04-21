# Satellite Imagery Aircraft Detection Pipeline

This project implements an end-to-end computer vision pipeline for identifying aircraft in high-resolution satellite imagery using the **RarePlanes** and **xView** datasets.

## Performance Summary
The model was trained on a truncated subset of 500 images for two epochs.
- **mAP @ 0.5 IoU:** 0.8494
- **mAP @ [0.5:0.95]:** 0.5204
- **Precision:** 0.89
- **Recall:** 0.80
- **Error Analysis:** TP: 560 | FP: 69 | FN: 136

## Project Structure
- `src/ingest.py`: Handles data ingestion from multiple sources (RarePlanes/xView). Harmonizes GeoJSON annotations and unifies classes.
- `src/preprocess.py`: Implements the `Albumentations` pipeline for image resizing (512x512), normalization (ImageNet stats), and data augmentation.
- `src/train.py`: Contains the model definition (Faster R-CNN ResNet-50 FPN) and the training loop with Mixed Precision (AMP) support.
- `src/evaluate.py`: Calculates COCO mAP metrics, categorizes error types (FP vs FN), and generates Precision-Recall curves.

## Data Structure
- **Rareplane:** `data/rareplanes/train/PS-RGB_tiled/` and `data/rareplanes/test/PS-RGB_tiled/`
- **Xview** `data/xview/train_images/train_images/` and `data/xview/val_images/val_images/`

## Data Processing & Strategy
- **Multi-Source Unification:** Harmonized xView and RarePlanes by mapping diverse aircraft sub-types to a single "Airplane" category.
- **Coordinate Handling:** Implemented a robust parsing strategy that handles both raw pixel coordinates and geographic (Lat/Lon) coordinates found in GeoJSON metadata.
- **Resilient Ingestion:** Integrated automatic coordinate clipping and filtering to remove "noisy" annotations that bleed over image tile boundaries, preventing pipeline crashes.
- **Efficiency:** Utilized a truncated training set (500 images) and Mixed Precision training to fit the complete pipeline development within a 4-hour window.

## Main Assumptions
- **Single-Class Ontology:** Aircraft sub-types (cargo, small, helicopter) are treated as a single "Airplane" class (Class 1).
- **RGB Standard:** All multispectral satellite data is processed as 8-bit RGB imagery.
- **Transfer Learning:** The pre-trained COCO weights of the Faster R-CNN backbone are sufficient for extracting meaningful features from top-down aerial views.
- **Sparsity:** Random image tiles may be empty; the pipeline includes logic to search for tiles with objects for visual validation.

## Evaluation Deliverables
The pipeline generates the following in the `outputs/` directory:
- `model_epoch_2.pth`: Trained model weights.
- `pr_curve.png`: Precision-Recall curve visualizing model performance trade-offs.
