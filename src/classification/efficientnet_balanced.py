import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix
from pathlib import Path
from PIL import Image
import yaml

# Config 
with open("/content/kidney-tumour-detection/configs/config.yaml") as f:
    config = yaml.safe_load(f)

splits_dir    = Path(config['paths']['splits_dir'])
results_dir   = Path(config['paths']['results_dir']) / "phase7_efficientnet_balanced"
weights_dir   = results_dir / "weights"
metrics_dir   = results_dir / "metrics"
patches_dir   = Path(config['paths']['patches_dir'])
local_patches = Path("/content/patches")
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights_dir.mkdir(parents=True, exist_ok=True)
metrics_dir.mkdir(parents=True, exist_ok=True)

# Balanced train set
train_df     = pd.read_csv(splits_dir / "efficientnet_train.csv")
val_df       = pd.read_csv(splits_dir / "efficientnet_val.csv")

malignant_df = train_df[train_df['malignant'] == True].sample(n=125, random_state=42)
benign_df    = train_df[train_df['malignant'] == False]
train_df     = pd.concat([malignant_df, benign_df], ignore_index=True)
train_df     = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Balanced train : {len(train_df)} patches")
print(train_df['malignant'].value_counts().rename({True: 'malignant', False: 'benign'}))
print(f"Val            : {len(val_df)} patches")

# Dataset
class PatchDataset(Dataset):
    def __init__(self, df, local_patches_dir, drive_patches_dir,
                 augment=False, oversample_factor=1):
        self.local_patches_dir = Path(local_patches_dir)
        self.drive_patches_dir = str(drive_patches_dir)
        df = df.copy()
        df['local_path'] = df['patch_path'].apply(self._remap_path)

        if oversample_factor > 1:
            benign_df    = df[df['malignant'] == False]
            malignant_df = df[df['malignant'] == True]
            benign_df    = pd.concat([benign_df] * oversample_factor, ignore_index=True)
            df           = pd.concat([malignant_df, benign_df], ignore_index=True)
            df           = df.sample(frac=1, random_state=42).reset_index(drop=True)

        self.df      = df
        self.augment = augment

        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ])
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ])

    def _remap_path(self, drive_path):
        return drive_path.replace(self.drive_patches_dir, str(self.local_patches_dir))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(row['local_path']).convert("L")
        label = torch.tensor(float(row['malignant']), dtype=torch.float32)
        img   = self.train_transform(img) if self.augment else self.val_transform(img)
        img   = img.repeat(3, 1, 1)
        return img, label

# Model
def build_model(device):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 1)
    )
    return model.to(device)

# Metrics
def compute_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0
    return {
        'accuracy'    : accuracy_score(labels, preds),
        'f1_malignant': f1_score(labels, preds, pos_label=1, zero_division=0),
        'f1_benign'   : f1_score(labels, preds, pos_label=0, zero_division=0),
        'sensitivity' : sensitivity,
        'specificity' : specificity,
        'auc'         : auc,
    }

# Epoch runner
def run_epoch(model, loader, criterion, optimizer, device, is_train):
    model.train() if is_train else model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []

    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs).squeeze(1)
            loss   = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    metrics  = compute_metrics(np.array(all_labels), np.array(all_probs))
    return avg_loss, metrics

# Training
epochs            = config['efficientnet']['epochs']
batch_size        = config['efficientnet']['batch_size']
lr                = config['efficientnet']['learning_rate']
lr_patience       = config['efficientnet']['lr_reduce_patience']
es_patience       = config['efficientnet']['early_stopping_patience']
seed              = config['dataset']['random_seed']

torch.manual_seed(seed)
np.random.seed(seed)

train_dataset = PatchDataset(train_df, local_patches, patches_dir, augment=True)
val_dataset   = PatchDataset(val_df,   local_patches, patches_dir, augment=False)

train_loader  = DataLoader(train_dataset, batch_size=batch_size,
                           shuffle=True,  num_workers=2, pin_memory=True)
val_loader    = DataLoader(val_dataset,   batch_size=batch_size,
                           shuffle=False, num_workers=2, pin_memory=True)

# 1:1 ratio — class weight is 1.0, no imbalance
pos_weight = torch.tensor([1.0], dtype=torch.float32).to(device)

model     = build_model(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=lr_patience, factor=0.5)

best_f1, epochs_no_improve, log_records = 0.0, 0, []

print("\nBalanced 1:1 Experiment — EfficientNet-B0")
print(f"Device: {device}")

for epoch in range(1, epochs + 1):
    train_loss, train_m = run_epoch(model, train_loader, criterion,
                                    optimizer, device, is_train=True)
    val_loss,   val_m   = run_epoch(model, val_loader,   criterion,
                                    optimizer, device, is_train=False)

    val_f1 = (val_m['f1_malignant'] + val_m['f1_benign']) / 2
    scheduler.step(val_f1)

    log_records.append({
        'epoch'           : epoch,
        'train_loss'      : round(train_loss, 4),
        'val_loss'        : round(val_loss, 4),
        'train_acc'       : round(train_m['accuracy'], 4),
        'val_acc'         : round(val_m['accuracy'], 4),
        'val_f1_malignant': round(val_m['f1_malignant'], 4),
        'val_f1_benign'   : round(val_m['f1_benign'], 4),
        'val_sensitivity' : round(val_m['sensitivity'], 4),
        'val_specificity' : round(val_m['specificity'], 4),
        'val_auc'         : round(val_m['auc'], 4),
    })

    print(f"Epoch {epoch:03d}/{epochs} | "
          f"Loss {train_loss:.4f}/{val_loss:.4f} | "
          f"Acc {train_m['accuracy']:.4f}/{val_m['accuracy']:.4f} | "
          f"F1_mal {val_m['f1_malignant']:.4f} | "
          f"F1_ben {val_m['f1_benign']:.4f} | "
          f"Sens {val_m['sensitivity']:.4f} | "
          f"Spec {val_m['specificity']:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save({
            'epoch'           : epoch,
            'model_state_dict': model.state_dict(),
            'val_f1'          : best_f1,
            'val_metrics'     : val_m,
        }, weights_dir / "best.pt")
        print(f"  Best model saved (mean F1: {best_f1:.4f})")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= es_patience:
        print(f"\nEarly stopping at epoch {epoch}")
        break

    pd.DataFrame(log_records).to_csv(metrics_dir / "training_log.csv", index=False)

print(f"\nBest mean F1: {best_f1:.4f}")
print("Done.")