# Step 7.4 — EfficientNet-B0 Evaluation
#
# This script loads the best checkpoint and runs inference on the val set.
# Reports final classification metrics for Chapter 4.
#
# Execution: Google Colab
#
# OUTPUT STRUCTURE:
#   results/
#   ├── phase7_efficientnet/
#   │   └── metrics/
#   │       ├── val_metrics.json
#   │       └── confusion_matrix.csv

import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import (f1_score, accuracy_score, roc_auc_score,
                              confusion_matrix, classification_report)


with open("/content/kidney-tumour-detection/configs/config.yaml") as f:
    config = yaml.safe_load(f)


# Dataset
class PatchDataset(Dataset):
    def __init__(self, df, local_patches_dir, drive_patches_dir):
        self.local_patches_dir = Path(local_patches_dir)
        self.drive_patches_dir = str(drive_patches_dir)
        df = df.copy()
        df['local_path'] = df['patch_path'].apply(self._remap_path)
        self.df = df

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ])

    def _remap_path(self, drive_path):
        return drive_path.replace(self.drive_patches_dir,
                                  str(self.local_patches_dir))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(row['local_path']).convert("L")
        label = torch.tensor(float(row['malignant']), dtype=torch.float32)
        img   = self.transform(img).repeat(3, 1, 1)
        return img, label


# Model
def load_model(checkpoint, device):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 1)
    )
    ckpt  = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
    print(f"Checkpoint val F1: {ckpt['val_f1']:.4f}")
    return model


# Main
def main():
    splits_dir    = Path(config['paths']['splits_dir'])
    results_dir   = Path(config['paths']['results_dir']) / "phase7_efficientnet"
    weights_dir   = results_dir / "weights"
    metrics_dir   = results_dir / "metrics"
    patches_dir   = Path(config['paths']['patches_dir'])
    local_patches = Path("/content/patches")
    threshold     = config['efficientnet']['classification_threshold']
    batch_size    = config['efficientnet']['batch_size']
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Step 7.4 — EfficientNet-B0 Evaluation")
    print(f"Device: {device}")

    # Load val set
    val_df  = pd.read_csv(splits_dir / "efficientnet_val.csv")
    dataset = PatchDataset(val_df, local_patches, patches_dir)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=False, num_workers=2, pin_memory=True)

    print(f"\nVal patches : {len(val_df)} "
          f"({val_df['malignant'].sum()} mal, {(~val_df['malignant']).sum()} ben)")

    # Load model
    model = load_model(weights_dir / "best.pt", device)

    # Run inference
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            logits = model(imgs).squeeze(1)
            probs  = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    all_preds  = (all_probs >= threshold).astype(int)

    # Metrics
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    metrics = {
        'accuracy'    : round(float(accuracy_score(all_labels, all_preds)), 4),
        'f1_malignant': round(float(f1_score(all_labels, all_preds, pos_label=1, zero_division=0)), 4),
        'f1_benign'   : round(float(f1_score(all_labels, all_preds, pos_label=0, zero_division=0)), 4),
        'sensitivity' : round(sensitivity, 4),
        'specificity' : round(specificity, 4),
        'auc'         : round(auc, 4),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }

    # Print report
    print("Final Validation Metrics")
    print(f"Accuracy    : {metrics['accuracy']}")
    print(f"F1 Malignant: {metrics['f1_malignant']}")
    print(f"F1 Benign   : {metrics['f1_benign']}")
    print(f"Sensitivity : {metrics['sensitivity']}  (malignant recall)")
    print(f"Specificity : {metrics['specificity']}  (benign recall)")
    print(f"AUC         : {metrics['auc']}")
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Benign  Malignant")
    print(f"Actual Benign   {tn:4d}     {fp:4d}")
    print(f"Actual Malignant{fn:4d}     {tp:4d}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                                 target_names=['benign', 'malignant'],
                                 zero_division=0))

    # Save
    with open(metrics_dir / "val_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({
        'actual'   : all_labels,
        'predicted': all_preds,
        'prob_malignant': all_probs,
    }).to_csv(metrics_dir / "val_predictions.csv", index=False)

    print(f"Metrics saved to: {metrics_dir}")
    print("\nDone.")

if __name__ == "__main__":
    main()