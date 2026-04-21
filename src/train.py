import os
import torch
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

# Import our custom modules
from ingest import get_unified_dataloader
from preprocess import get_transforms

def get_model(num_classes):
    """
    Initializes a Faster R-CNN model with a ResNet-50 backbone.
    Pre-trained on COCO to leverage transfer learning.
    """
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train_one_epoch(model, dataloader, optimizer, device, epoch, scaler):
    """
    Training loop using tqdm for compact progress tracking and modern AMP syntax.
    """
    model.train()
    total_loss = 0.0
    
    # tqdm for a compact, single-line progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch", leave=True)
    
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # MODERN AMP SYNTAX: torch.amp.autocast
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        with torch.amp.autocast(device_type=device_type):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        
        # Gradient Scaling
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        
        current_loss = losses.item()
        total_loss += current_loss
        
        # Update progress bar with current loss
        pbar.set_postfix(loss=f"{current_loss:.4f}")
        
    return total_loss / len(dataloader)

def main():
    # Configuration for 1-class (Background + Airplane)
    NUM_CLASSES = 2  
    BATCH_SIZE = 2
    NUM_EPOCHS = 2
    MAX_TRAIN_SAMPLES = 500 
    LEARNING_RATE = 0.005
    
    # Paths
    RARE_IMG = "data/rareplanes/train/PS-RGB_tiled"
    RARE_GEO = "data/rareplanes/train/geojson_aircraft_tiled"
    XVIEW_IMG = "data/xview/train_images/train_images"
    XVIEW_GEO = "data/xview/train_labels/xView_train.geojson"
    
    os.makedirs("outputs", exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # 1. Setup Data
    transform = get_transforms(train=True)
    full_loader = get_unified_dataloader(
        RARE_IMG, RARE_GEO, XVIEW_IMG, XVIEW_GEO, 
        batch_size=BATCH_SIZE, transform=transform
    )
    
    # 2. Dataset Truncation
    indices = torch.randperm(len(full_loader.dataset))[:MAX_TRAIN_SAMPLES].tolist()
    train_subset = Subset(full_loader.dataset, indices)
    train_loader = DataLoader(
        train_subset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=lambda x: tuple(zip(*x))
    )

    # 3. Model & Optimizer
    model = get_model(num_classes=NUM_CLASSES)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    
    # MODERN AMP SYNTAX: torch.amp.GradScaler
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    scaler = torch.amp.GradScaler(device_type)

    # 4. Training Loop
    print(f"Starting training on {len(train_subset)} images...")
    for epoch in range(1, NUM_EPOCHS + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, scaler)
        print(f"Average Loss for Epoch {epoch}: {avg_loss:.4f}")
        
        # Save weights
        torch.save(model.state_dict(), f"outputs/model_epoch_{epoch}.pth")

    print("Training Complete.")

if __name__ == "__main__":
    main()