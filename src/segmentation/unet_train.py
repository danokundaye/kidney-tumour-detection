# Step 6.4 — U-Net Segmentation Training
#
#   This script trains a U-Net with ResNet50 encoder to segment tumour and 
#   cyst regions from cropped kidney images.
#
# Architecture:
#   - Encoder : ResNet50 (pre-trained on ImageNet)
#   - Decoder : U-Net decoder with skip connections
#   - Output  : Binary segmentation mask (0 = healthy, 1 = abnormal)
#
# Steps:
#   - Sample healthy slices at 3:1 ratio with abnormal to reduce class imbalance
#   - Combined Dice + BCE loss
#   - Adam optimizer, lr=0.0001, cosine annealing schedule
#   - Early stopping: patience = 30 epochs based on validation Dice
#   - Max epochs: 150
#
# Input:
#   - unet_crops/case_id/images/     CT crops (256x256 grayscale)
#   - unet_crops/case_id/masks/      Binary masks (256x256, 0/255)
#   - unet_crops/case_id/region_types/ Region type txt files
#   - splits/unet_train.csv          108 training cases
#   - splits/unet_val.csv            12 validation cases
#
# Execution: Google Colab
#
# Output:
#   - results/phase6_unet/weights/best.pt    Best model by val Dice
#   - results/phase6_unet/weights/last.pt    Final epoch model
#   - results/phase6_unet/metrics.csv        Per-epoch metrics log
# =============================================================================

import os
import csv
import yaml
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Load config.yaml
def load_config(config_path: str) -> dict:
    """
    Load the config.yaml file
    All paths and settings come from here
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Dataset definition

class KidneySegDataset(Dataset):
    """
    Dataset for U-Net segmentation training.

    Loads cropped kidney images and their binary masks.
    Applies sampling to balance healthy vs abnormal slices at 3:1 ratio.

    Args:
        case_ids      : list of case IDs to include
        crops_dir     : path to unet_crops directory
        augment       : apply data augmentation (True for train, False for val)
        healthy_ratio : ratio of healthy to abnormal slices (default 3:1)
    """
    def __init__(
            self,
            case_ids     : list,
            crops_dir    : Path,
            augment      : bool = False,
            healthy_ratio: int  = 3
        ):

        self.crops_dir = crops_dir
        self.augment   = augment

        # Separate slices into abnormal and healthy
        abnormal_slices = []
        healthy_slices  = []

        for case_id in case_ids:
            images_dir  = crops_dir / case_id / "images"
            regions_dir = crops_dir / case_id / "region_types"

            if not images_dir.exists():
                continue

            for img_path in sorted(images_dir.glob("*.png")):
                slice_name   = img_path.stem
                region_file  = regions_dir / f"{slice_name}.txt"

                if region_file.exists():
                    region_type = region_file.read_text().strip()
                else:
                    region_type = "none"

                entry = {
                    'case_id'    : case_id,
                    'slice_name' : img_path.name,
                    'image_path' : img_path,
                    'mask_path'  : crops_dir / case_id / "masks" / img_path.name,
                    'region_type': region_type
                }

                if region_type in ("tumour_only", "cyst_only", "both"):
                    abnormal_slices.append(entry)
                else:
                    healthy_slices.append(entry)

        # Sample healthy slices to achieve healthy_ratio:1
        n_abnormal       = len(abnormal_slices)
        n_healthy_sample = min(len(healthy_slices), n_abnormal * healthy_ratio)

        random.seed(42)
        sampled_healthy = random.sample(healthy_slices, n_healthy_sample)

        self.slices = abnormal_slices + sampled_healthy
        random.shuffle(self.slices)

        print(f"  Abnormal slices  : {n_abnormal}")
        print(f"  Healthy slices   : {n_healthy_sample} (sampled from {len(healthy_slices)})")
        print(f"  Total slices     : {len(self.slices)}")

        # Augmentation pipeline for training
        self.train_transform = A.Compose([
            A.HorizontalFlip(p = 0.5),
            A.Rotate(limit = 15, p = 0.5),
            A.RandomBrightnessContrast(
                brightness_limit = 0.2,
                contrast_limit = 0.2,
                p = 0.5
            ),
            A.GaussNoise(p = 0.3),
            A.ElasticTransform(p = 0.3),
            A.Normalize(mean = [0.485], std = [0.229]),
            ToTensorV2()
        ])

        # Validation — only normalize, no augmentation
        self.val_transform = A.Compose([
            A.Normalize(mean = [0.485], std = [0.229]),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        entry = self.slices[idx]

        # Load image as grayscale, convert to RGB for ResNet50
        # ResNet50 expects 3-channel input
        image = np.array(Image.open(entry['image_path']).convert('RGB'))

        # Load binary mask stored as 0/255, convert to 0/1
        mask = np.array(Image.open(entry['mask_path']))
        mask = (mask > 127).astype(np.float32)  # threshold at 127 → 0 or 1

        # Apply transforms
        if self.augment:
            transformed = self.train_transform(image = image, mask = mask)
        else:
            transformed = self.val_transform(image = image, mask = mask)

        image = transformed['image']                # shape: (3, 256, 256)
        mask  = transformed['mask'].unsqueeze(0)    # shape: (1, 256, 256)

        return image, mask


# Loss functions

class DiceBCELoss(nn.Module):
    """
    Combined Dice Loss + Binary Cross-Entropy Loss.

    Dice loss handles class imbalance by rewarding overlap between
    predicted and ground truth masks.

    BCE loss ensures pixel-wise classification accuracy, particularly
    at tumour boundaries.
    """
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
        self.bce    = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred, target):
        # Apply sigmoid to get probabilities
        pred = torch.sigmoid(pred)

        # Flatten for overlap computation
        pred   = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce  = self.bce(pred, target)           # uses the stored loss function
        return dice + bce


# Computational metrics
def compute_dice(
        pred_logits: torch.Tensor,
        target     : torch.Tensor,
        threshold  : float = 0.5,
        smooth     : float = 1e-6
    ) -> float:
    """
    Compute Dice coefficient from model logits and ground truth mask.
    """
    pred = (torch.sigmoid(pred_logits) > threshold).float()

    pred   = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()


def compute_iou(
        pred_logits: torch.Tensor,
        target     : torch.Tensor,
        threshold  : float = 0.5,
        smooth     : float = 1e-6
    ) -> float:
    """
    Compute Intersection over Union from model logits and ground truth.
    """
    pred = (torch.sigmoid(pred_logits) > threshold).float()

    pred   = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    union        = pred.sum() + target.sum() - intersection
    iou          = (intersection + smooth) / (union + smooth)
    return iou.item()

# Training loop
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """
    Run one full training epoch. Returns average loss and Dice.
    """
    model.train()

    total_loss = 0.0
    total_dice = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += compute_dice(outputs, masks)

    n = len(loader)
    return total_loss / n, total_dice / n


def validate(model, loader, loss_fn, device):
    """
    Run validation. 
    Returns average loss, Dice and IoU.
    """
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou  = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks  = masks.to(device)

            outputs = model(images)
            loss    = loss_fn(outputs, masks)

            total_loss += loss.item()
            total_dice += compute_dice(outputs, masks)
            total_iou  += compute_iou(outputs, masks)

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n

# Main model training
def train_unet(crops_dir  : Path,
               splits_dir : Path,
               results_dir: Path,
               unet_size  : int,
               max_epochs : int = 150,
               patience   : int = 30,
               batch_size : int = 16,
               lr         : float = 0.0001):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device           : {device}")

    # Load case splits
    train_cases = pd.read_csv(splits_dir / "unet_train.csv")['case_id'].tolist()
    val_cases   = pd.read_csv(splits_dir / "unet_val.csv")['case_id'].tolist()

    print(f"\nTrain cases      : {len(train_cases)}")
    print(f"Val cases        : {len(val_cases)}")

    # Build datasets
    print("\nBuilding train dataset:")
    train_dataset = KidneySegDataset(
        case_ids = train_cases,
        crops_dir = crops_dir,
        augment   = True,
        healthy_ratio = 3
    )

    print("\nBuilding val dataset:")
    val_dataset = KidneySegDataset(
        case_ids = val_cases,
        crops_dir = crops_dir,
        augment   = False,
        healthy_ratio = 3
    )

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = 4,
        pin_memory  = True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 4,
        pin_memory  = True
    )

    # Model: U-Net with ResNet50 encoder
    model = smp.Unet(
        encoder_name    = 'resnet50',
        encoder_weights = 'imagenet',   # pre-trained on ImageNet
        in_channels     = 3,            # RGB (converted from grayscale)
        classes         = 1,            # binary segmentation
        activation      = None          # raw logits — loss handles sigmoid
    )
    model = model.to(device)

    # Loss, optimizer, scheduler
    loss_fn   = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-6
    )

    # Results directory
    weights_dir = results_dir / "weights"
    weights_dir.mkdir(parents = True, exist_ok = True)

    # Metrics CSV
    metrics_path = results_dir / "metrics.csv"
    metrics_file = open(metrics_path, 'w', newline='')
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow([
        'epoch', 'train_loss', 'train_dice',
        'val_loss', 'val_dice', 'val_iou', 'lr'
    ])

    # Training state
    best_val_dice    = 0.0
    epochs_no_improve = 0
    best_epoch       = 0

    print(f"\nStarting training...")
    print(f"Max epochs       : {max_epochs}")
    print(f"Patience         : {patience}")
    print(f"Batch size       : {batch_size}")
    print(f"Learning rate    : {lr}")
    print(f"Results dir      : {results_dir}")
    print("=" * 60)

    for epoch in range(1, max_epochs + 1):

        # Train
        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device
        )

        # Validate
        val_loss, val_dice, val_iou = validate(
            model, val_loader, loss_fn, device
        )

        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics
        metrics_writer.writerow([
            epoch, f"{train_loss:.4f}", f"{train_dice:.4f}",
            f"{val_loss:.4f}", f"{val_dice:.4f}", f"{val_iou:.4f}",
            f"{current_lr:.6f}"
        ])
        metrics_file.flush()

        print(
            f"Epoch {epoch:3d}/{max_epochs} | "
            f"Train Loss: {train_loss:.4f}  Dice: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f}  Dice: {val_dice:.4f}  IoU: {val_iou:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # Save best model
        if val_dice > best_val_dice:
            best_val_dice     = val_dice
            best_epoch        = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(),
                       weights_dir / "best.pt")
            print(f" New best model saved (Val Dice: {best_val_dice:.4f})")
        else:
            epochs_no_improve += 1

        # Save last model every 10 epochs as backup
        if epoch % 10 == 0:
            torch.save(model.state_dict(),
                       weights_dir / "last.pt")

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            print(f"Best Val Dice: {best_val_dice:.4f} at epoch {best_epoch}")
            break

    # Save final last model
    torch.save(model.state_dict(), weights_dir / "last.pt")
    metrics_file.close()

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Best Val Dice  : {best_val_dice:.4f} at epoch {best_epoch}")
    print(f"  Weights saved  : {weights_dir}")
    print(f"  Metrics saved  : {metrics_path}")


# Main

def main():
    config_path = "/content/kidney-tumour-detection/configs/config.yaml"
    config      = load_config(config_path)

    # Local storage paths — read from local for speed
    local_crops_dir = Path("/content/local_data/unet_crops")
    splits_dir      = Path(config['paths']['splits_dir'])
    results_dir     = Path(config['paths']['results_dir']) / "phase6_unet"

    # Drive crops path for rsync
    drive_crops_dir = Path(config['paths']['unet_crops_dir'])

    print("Step 6.4 — U-Net Segmentation Training")

    # Copy crops to local storage if not already present
    if not local_crops_dir.exists() or not any(local_crops_dir.iterdir()):
        print("\nCopying unet_crops to local storage (~210MB)...")
        local_crops_dir.mkdir(parents=True, exist_ok=True)
        src = str(drive_crops_dir) + "/"
        dst = str(local_crops_dir) + "/"
        ret = os.system(f"rsync -a --info=progress2 '{src}' '{dst}'")
        if ret != 0:
            raise RuntimeError("rsync failed. Check Drive mount.")
        print("Crops copied to local storage.")
    else:
        print("\nLocal crops already exist, skipping copy.")

    train_unet(
        crops_dir   = local_crops_dir,
        splits_dir  = splits_dir,
        results_dir = results_dir,
        unet_size   = config['preprocessing']['unet_input_size'],
        max_epochs  = 150,
        patience    = 30,
        batch_size  = 16,
        lr          = 0.0001
    )


if __name__ == "__main__":
    main()