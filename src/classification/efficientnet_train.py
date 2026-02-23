# Step 7.3 — EfficientNet-B0 Classification Training
#
# This script trains EfficientNet-B0 to classify tumour patches as benign or malignant.
#
# Key design decisions:
#   - All layers trainable (no frozen encoder) — CT texture differs from ImageNet
#   - Class-weighted BCE loss (weight = 3827/125 = 30.6 for benign)
#   - 15x oversampling of benign patches in training set
#   - Early stopping on validation F1 (patience = 10)
#   - LR reduction on plateau (patience = 5)
#   - Patches read from local Colab storage for speed (as seen in the previous notebook cell)
#
# Metrics logged per epoch:
#   - Loss, accuracy, F1 (per class), sensitivity, specificity, AUC
#
# Execution - Google Colab
#
# OUTPUT STRUCTURE:
#   results/
#   ├── phase7_efficientnet/
#   │   ├── weights/
#   │   │   └── best.pt
#   │   └── metrics/
#   │       └── training_log.csv

import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import (
    f1_score, 
    accuracy_score, 
    roc_auc_score,
    confusion_matrix)
from tqdm import tqdm


# Load config.yaml
def load_config(config_path: str) -> dict:
    """
    Load the config.yaml file
    All paths and settings come from here
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Dataset
class PatchDataset(Dataset):
    """
    Loads the 224x224 CT patches from local storage
    Applies augmentation during training only
    Oversamples benign patches by oversample_factor in training
    """
    def __init__(self, df, local_patches_dir, drive_patches_dir, augment = False, oversample_factor = 1):
        # Remap patch paths from Drive to local storage
        self.local_patches_dir = Path(local_patches_dir)
        self.drive_patches_dir = str(drive_patches_dir)

        df = df.copy()
        # Remap path so DataLoader hits local storage instead of Drive
        df['local_path'] = df['patch_path'].apply(self._remap_path)

        # Oversample benign patches
        if oversample_factor > 1:
            benign_df    = df[df['malignant'] == False]
            malignant_df = df[df['malignant'] == True]
            benign_df    = pd.concat([benign_df] * oversample_factor, ignore_index = True)
            df           = pd.concat([malignant_df, benign_df], ignore_index = True)
            df           = df.sample(frac = 1, random_state = 42).reset_index(drop = True)

        self.df      = df
        self.augment = augment

        # Training augmentation — heavier on benign to improve generalisation
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomVerticalFlip(p = 0.5),
            transforms.RandomRotation(degrees = 30),
            transforms.ColorJitter(brightness = 0.3, contrast = 0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485], std = [0.229]),
        ])

        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485], std = [0.229]),
        ])

    def _remap_path(self, drive_path):
        """
        Replace Drive patches dir with local patches dir in path
        """
        return drive_path.replace(self.drive_patches_dir, str(self.local_patches_dir))


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(row['local_path']).convert("L")
        label = torch.tensor(float(row['malignant']), dtype=torch.float32)

        if self.augment:
            img = self.train_transform(img)
        else:
            img = self.val_transform(img)

        # EfficientNet expects 3 channels — repeat grayscale
        img = img.repeat(3, 1, 1)
        return img, label


# Model
def build_model(device):
    """
    EfficientNet-B0 pretrained on ImageNet
    Replace final classifier with single sigmoid output for binary classification
    All layers trainable
    """
    model = models.efficientnet_b0(weights = models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p = 0.2),
        nn.Linear(in_features, 1)
    )
    return model.to(device)


# Metrics
def compute_metrics(labels, probs, threshold=0.5):
    """
    Compute classification metrics from predicted probabilities
    Returns dict with accuracy, F1 per class, sensitivity, specificity, AUC
    """
    preds = (probs >= threshold).astype(int)

    # Confusion matrix — [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels = [0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # malignant recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # benign recall

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0  # only one class present in batch

    return {
        'accuracy'   : accuracy_score(labels, preds),
        'f1_malignant': f1_score(labels, preds, pos_label = 1, zero_division = 0),
        'f1_benign'  : f1_score(labels, preds, pos_label = 0, zero_division = 0),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'auc'        : auc,
    }


# Train / val loops
def run_epoch(model, loader, criterion, optimizer, config, device, is_train):
    """
    Run one epoch. Returns loss and metric dictionary
    """
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_labels = []
    all_probs  = []

    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs).squeeze(1)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    metrics  = compute_metrics(
                    np.array(all_labels),
                    np.array(all_probs),
                    threshold=config['efficientnet']['classification_threshold']
                )
    return avg_loss, metrics


# Main
def main():
    config_path = "/content/kidney-tumour-detection/configs/config.yaml"
    config      = load_config(config_path)
    
    splits_dir    = Path(config['paths']['splits_dir'])
    patches_dir   = Path(config['paths']['patches_dir'])
    results_dir   = Path(config['paths']['results_dir']) / "phase7_efficientnet"
    weights_dir   = results_dir / "weights"
    metrics_dir   = results_dir / "metrics"
    local_patches = Path("/content/patches")

    weights_dir.mkdir(parents = True, exist_ok = True)
    metrics_dir.mkdir(parents = True, exist_ok = True)

    # Settings
    epochs             = config['efficientnet']['epochs']
    batch_size         = config['efficientnet']['batch_size']
    lr                 = config['efficientnet']['learning_rate']
    lr_patience        = config['efficientnet']['lr_reduce_patience']
    es_patience        = config['efficientnet']['early_stopping_patience']
    oversample_factor  = config['efficientnet']['oversample_factor']
    seed               = config['dataset']['random_seed']
    device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    print("Step 7.3 — EfficientNet-B0 Classification Training")
    print(f"Device: {device}")

    # Load splits
    train_df = pd.read_csv(splits_dir / "efficientnet_train.csv")
    val_df   = pd.read_csv(splits_dir / "efficientnet_val.csv")

    print(f"\nTrain patches : {len(train_df)} "
          f"({train_df['malignant'].sum()} mal, {(~train_df['malignant']).sum()} ben)")
    print(f"Val patches   : {len(val_df)} "
          f"({val_df['malignant'].sum()} mal, {(~val_df['malignant']).sum()} ben)")
    print(f"Oversample factor (benign): {oversample_factor}x")

    # Datasets
    train_dataset = PatchDataset(train_df, local_patches, patches_dir,
                                 augment = True, oversample_factor=oversample_factor)
    val_dataset   = PatchDataset(val_df,   local_patches, patches_dir,
                                 augment = False, oversample_factor=1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle = True,  num_workers=2, pin_memory = True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle = False, num_workers=2, pin_memory = True)

    # Class weight for benign — inverse frequency
    n_malignant      = train_df['malignant'].sum()
    n_benign         = (~train_df['malignant']).sum()
    benign_weight    = n_malignant / n_benign
    pos_weight       = torch.tensor([benign_weight], dtype=torch.float32).to(device)

    print(f"Benign class weight: {benign_weight:.2f}")

    # Model, loss, optimizer, scheduler
    model     = build_model(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max',
                                  patience=lr_patience, factor=0.5)

    # Training loop
    best_f1         = 0.0
    epochs_no_improve = 0
    log_records     = []

    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, criterion,
                                               optimizer, config, device, is_train = True)
        val_loss,   val_metrics   = run_epoch(model, val_loader,   criterion,
                                               optimizer, config, device, is_train = False)

        # Use mean of benign and malignant F1 as primary metric
        # Prevents model gaming metric by predicting only malignant
        val_f1 = (val_metrics['f1_malignant'] + val_metrics['f1_benign']) / 2
        scheduler.step(val_f1)

        # Log
        record = {
            'epoch'           : epoch,
            'train_loss'      : round(train_loss, 4),
            'val_loss'        : round(val_loss, 4),
            'train_acc'       : round(train_metrics['accuracy'], 4),
            'val_acc'         : round(val_metrics['accuracy'], 4),
            'val_f1_malignant': round(val_metrics['f1_malignant'], 4),
            'val_f1_benign'   : round(val_metrics['f1_benign'], 4),
            'val_sensitivity' : round(val_metrics['sensitivity'], 4),
            'val_specificity' : round(val_metrics['specificity'], 4),
            'val_auc'         : round(val_metrics['auc'], 4),
        }
        log_records.append(record)

        print(f"Epoch {epoch:03d}/{epochs} | "
              f"Loss {train_loss:.4f}/{val_loss:.4f} | "
              f"Acc {train_metrics['accuracy']:.4f}/{val_metrics['accuracy']:.4f} | "
              f"F1_mal {val_metrics['f1_malignant']:.4f} | "
              f"F1_ben {val_metrics['f1_benign']:.4f} | "
              f"Sens {val_metrics['sensitivity']:.4f} | "
              f"Spec {val_metrics['specificity']:.4f}")

        # Save best checkpoint
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch'            : epoch,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1'           : best_f1,
                'val_metrics'      : val_metrics,
            }, weights_dir / "best.pt")
            print(f" Best model saved (mean F1: {best_f1:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= es_patience:
            print(f"\nEarly stopping triggered at epoch {epoch} "
                  f"(no improvement for {es_patience} epochs)")
            break

        # Save log after every epoch
        pd.DataFrame(log_records).to_csv(
            metrics_dir / "training_log.csv", index = False)

    print(f"\nTraining complete. Best mean F1: {best_f1:.4f}")
    print(f"Weights saved to : {weights_dir / 'best.pt'}")
    print(f"Metrics saved to : {metrics_dir / 'training_log.csv'}")
    print("\nDone.")

if __name__ == "__main__":
    main()