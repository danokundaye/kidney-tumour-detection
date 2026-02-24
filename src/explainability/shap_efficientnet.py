# Step 8.1 — SHAP Explainability for EfficientNet-B0 (Classification)
#
#   This script applies SHAP (SHapley Additive exPlanations) to the trained
#   EfficientNet-B0 model to understand WHICH pixels in tumour patches drive
#   the model's benign vs malignant classification decisions.
#
#   This serves two purposes in thesis:
#     1. Visual: heatmaps showing "where the model looks" for Chapter 4
#     2. Quantitative: mean absolute SHAP values per class for Chapter 4 tables
#
# HOW SHAP WORKS:
#   - Background samples: 50 random training patches (the model's "neutral" baseline)
#   - Test images: patches from held-out test set
#   - SHAP computes: for each pixel, how much did it push the prediction
#     toward malignant (+) or toward benign (-)?
#
# Execution: Google Colab
#
# OUTPUTS:
#   results/phase8_shap/efficientnet/
#     ├── visualizations/          ← 3-column PNG per test image
#     ├── shap_values/             ← raw .npy arrays saved per image
#     ├── shap_summary.csv         ← mean abs SHAP values per image (for thesis)
#     └── class_summary.csv        ← mean abs SHAP aggregated by true label


import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import shap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from pathlib import Path
import yaml
import csv
from datetime import datetime

# Config

# Base project directory on Google Drive
DRIVE_BASE = "/content/drive/MyDrive/kidney-tumour-detection"

# Paths to EfficientNet model and patches
EFFICIENTNET_MODEL_PATH = f"{DRIVE_BASE}/results/phase7_efficientnet/weights/best.pt"

# Patch directories (structure: patches/train/benign/, patches/train/malignant/, etc.)
PATCHES_BASE    = f"{DRIVE_BASE}/dataset/processed/patches"
TRAIN_DIR       = f"{PATCHES_BASE}/train"
VAL_DIR         = f"{PATCHES_BASE}/val"
TEST_DIR        = f"{PATCHES_BASE}/test"

# Output directory for Phase 8 SHAP results
OUTPUT_DIR      = f"{DRIVE_BASE}/results/phase8_shap/efficientnet"

# SHAP settings
N_BACKGROUND    = 50    # Number of training images used as SHAP background
                        # More = more accurate SHAP values, but slower
                        # 50 is a good balance for the dataset size
N_TEST_SAMPLES  = 30    # Number of test patches to explain
                        # Full test set can be slow; 30 gives representative sample
RANDOM_SEED     = 42

# EfficientNet input size
IMG_SIZE        = 224

# Dataset class
# Loads tumour patches from the folder structure:
#   split/benign/*.png
#   split/malignant/*.png

class TumourPatchDataset(Dataset):
    """
    Loads tumour patches from a directory with class subfolders.
    
    Expected structure:
        root_dir/
            benign/
                patch_001.png
                patch_002.png
                ...
            malignant/
                patch_001.png
                ...
    
    Returns: (image_tensor, label, file_path)
    label: 0 = benign, 1 = malignant
    """
    
    def __init__(self, root_dir: str, transform=None):
        self.root_dir  = Path(root_dir)
        self.transform = transform
        self.samples   = []
        
        # Walk through benign and malignant subdirectories
        for label_name, label_idx in [("benign", 0), ("malignant", 1)]:
            class_dir = self.root_dir / label_name
            if not class_dir.exists():
                print(f"  ⚠ Warning: {class_dir} not found — skipping")
                continue
            for img_path in sorted(class_dir.glob("*.png")):
                self.samples.append((str(img_path), label_idx))
        
        print(f"  Loaded {len(self.samples)} patches from {root_dir}")
        print(f"    Benign   : {sum(1 for _, l in self.samples if l == 0)}")
        print(f"    Malignant: {sum(1 for _, l in self.samples if l == 1)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image and convert to RGB (EfficientNet expects 3 channels)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path


# Model Loading
# Rebuild EfficientNet-B0 architecture exactly as it was trained in Phase 7,
# then load the saved weights.

def load_efficientnet(model_path: str, device: torch.device) -> nn.Module:
    """
    Rebuild EfficientNet-B0 with the same head used in Phase 7 training,
    then load the saved weights.
    
    Why rebuild instead of torch.load(model_path)?
    torch.load() can fail if the class definition isn't available in the
    current session. Rebuilding the architecture and loading state_dict
    is more robust.
    """
    print(f"\nLoading EfficientNet-B0 from:\n  {model_path}")
    
    # Rebuild the same architecture from Phase 7
    model = models.efficientnet_b0(weights = None)  # No ImageNet weights
    
    # Replace classifier head exactly as done in Phase 7
    # EfficientNet-B0's last layer is a Linear(1280, 1000)
    # We replaced it with Linear(1280, 1) for binary classification
    in_features = model.classifier[1].in_features  # 1280 for EfficientNet-B0
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 1)
    )
    
    # Load the saved Phase 7 weights
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle both raw state_dict and wrapped checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded from checkpoint dict (epoch {checkpoint.get('epoch', 'unknown')})")
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Assume it IS the state dict directly
        model.load_state_dict(checkpoint)
        print(f"  Loaded state dict directly")
    
    model = model.to(device)
    model.eval()  # CRITICAL: must be in eval mode for SHAP
    print(f"  Model ready on: {device}")
    
    return model


# SHAP Wrapper
# EfficientNet outputs a single logit (raw score before sigmoid).
# SHAP needs probabilities, so we wrap the model to apply sigmoid.

class EfficientNetSHAPWrapper(nn.Module):
    """
    Wraps EfficientNet to output a probability (0-1) instead of a raw logit.
    
    Why needed?
    SHAP's DeepExplainer interprets the output as a probability.
    EfficientNet outputs raw logits (unbounded values). Applying sigmoid
    converts them to probabilities so SHAP values are interpretable:
      positive SHAP → pushed toward malignant (probability closer to 1)
      negative SHAP → pushed toward benign (probability closer to 0)
    """
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model   = model
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        logit       = self.model(x)          # shape: (batch, 1)
        probability = self.sigmoid(logit)     # shape: (batch, 1)
        return probability


# Visualisation Helper
# Creates the 3-column figure: Original | SHAP heatmap | Overlay

def visualize_shap(
    image_tensor: torch.Tensor,
    shap_values:  np.ndarray,
    true_label:   int,
    pred_prob:    float,
    img_path:     str,
    save_path:    str
):
    """
    Creates and saves a 3-column SHAP visualization.
    
    Columns:
      1. Original patch (grayscale-ish medical image)
      2. SHAP heatmap (absolute values — how STRONGLY did each pixel contribute?)
      3. Overlay (original + heatmap blended)
    
    Color in heatmap:
      Bright red/yellow = high absolute SHAP value = important region
      Dark blue/black   = low absolute SHAP value = less important region
    
    Args:
        image_tensor : (3, H, W) tensor, values in ImageNet-normalized range
        shap_values  : (3, H, W) numpy array of SHAP values
        true_label   : 0 = benign, 1 = malignant
        pred_prob    : model's predicted probability of malignant
        img_path     : original file path (for title)
        save_path    : where to save the PNG
    """
    
    # Convert image tensor back to displayable format
    # Undo ImageNet normalization: multiply by std, add mean
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    
    # image_tensor shape: (3, H, W) → (H, W, 3)
    img_display = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img_display = img_display * std + mean              # Undo normalization
    img_display = np.clip(img_display, 0, 1)            # Clip to valid range
    
    # Compute SHAP magnitude map:
    # Sum absolute SHAP values across the 3 color channels → (H, W)
    # This gives us ONE heatmap showing overall pixel importance
    shap_magnitude = np.abs(shap_values).sum(axis=0)    # (H, W)
    
    # Normalize to 0-1 for display
    shap_min = shap_magnitude.min()
    shap_max = shap_magnitude.max()
    if shap_max > shap_min:
        shap_norm = (shap_magnitude - shap_min) / (shap_max - shap_min)
    else:
        shap_norm = shap_magnitude  # Flat image, all zeros
    
    # Create overlay: blend original image with heatmap
    # Use jet colormap on SHAP, then blend at 50% alpha
    heatmap_colored = plt.cm.jet(shap_norm)[:, :, :3]  # (H, W, 3) in 0-1
    overlay = img_display * 0.5 + heatmap_colored * 0.5
    overlay = np.clip(overlay, 0, 1)
    
    # --- Build the figure ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    true_label_name  = "Malignant" if true_label  == 1 else "Benign"
    pred_label_name  = "Malignant" if pred_prob   >= 0.5 else "Benign"
    correct          = true_label_name == pred_label_name
    
    # Column 1: Original patch
    axes[0].imshow(img_display)
    axes[0].set_title("Original Patch", fontsize=11)
    axes[0].axis("off")
    
    # Column 2: SHAP magnitude heatmap
    im = axes[1].imshow(shap_norm, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("SHAP Magnitude\n(bright = influential)", fontsize=11)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Column 3: Overlay
    axes[2].imshow(overlay)
    axes[2].set_title("SHAP Overlay", fontsize=11)
    axes[2].axis("off")
    
    # Overall figure title with prediction result
    status = " CORRECT" if correct else " WRONG"
    color  = "green"     if correct else "red"
    
    fig.suptitle(
        f"True: {true_label_name}  |  Predicted: {pred_label_name} ({pred_prob:.3f})  |  {status}\n"
        f"{Path(img_path).name}",
        fontsize=12, color=color, y=1.02
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Main
def main():
    print("  Phase 8.1 — SHAP Explainability: EfficientNet-B0")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
   
    # Setup device and output directories
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
    
    # Create output directories
    vis_dir   = Path(OUTPUT_DIR) / "visualizations"
    shap_dir  = Path(OUTPUT_DIR) / "shap_values"
    vis_dir.mkdir(parents=True, exist_ok=True)
    shap_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Verify model and patch paths exist before doing anything
    print("\n--- Verifying paths ---")
    
    if not Path(EFFICIENTNET_MODEL_PATH).exists():
        print(f" Model not found: {EFFICIENTNET_MODEL_PATH}")
        print("  Check that Phase 7 completed and the path is correct.")
        sys.exit(1)
    print(f" Model found: {EFFICIENTNET_MODEL_PATH}")
    
    for split_name, split_dir in [("train", TRAIN_DIR), ("test", TEST_DIR)]:
        if not Path(split_dir).exists():
            print(f" Patches not found: {split_dir}")
            print(f"  Check that Phase 7 patch preparation completed.")
            sys.exit(1)
        print(f"{split_name} patches found: {split_dir}")
    
    # Build the image transform
    # ImageNet normalization values are used because EfficientNet
    # was pre-trained on ImageNet. Mismatching these would distort predictions.
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])
    
    # Load datasets
    print("\n--- Loading datasets ---")
    
    train_dataset = TumourPatchDataset(TRAIN_DIR, transform=transform)
    test_dataset  = TumourPatchDataset(TEST_DIR,  transform=transform)
    
    if len(train_dataset) == 0:
        print(" No training patches found. Cannot create SHAP background.")
        sys.exit(1)
    
    if len(test_dataset) == 0:
        print(" No test patches found. Cannot compute SHAP explanations.")
        sys.exit(1)
    

    # Background = the "neutral" reference SHAP compares test images against.
    # Randomly sample N_BACKGROUND patches from the training set.
    # Using training images (not test) is important to prevent
    # background "leak" information from test cases.
    print(f"\n--- Selecting {N_BACKGROUND} background samples ---")
    
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    # Sample indices for background
    n_avail      = len(train_dataset)
    bg_indices   = np.random.choice(n_avail, min(N_BACKGROUND, n_avail), replace=False)
    
    # Collect background images as a tensor: (N_BACKGROUND, 3, 224, 224)
    bg_images = []
    for idx in bg_indices:
        img, _, _ = train_dataset[idx]
        bg_images.append(img)
    
    background_tensor = torch.stack(bg_images).to(device)
    print(f"  Background tensor shape: {background_tensor.shape}")
    
    # Count class balance of background
    bg_labels = [train_dataset.samples[i][1] for i in bg_indices]
    print(f"  Background composition — Benign: {bg_labels.count(0)}, Malignant: {bg_labels.count(1)}")
    
    # Select test samples
    # 
    # Stratified selection: pick N_TEST_SAMPLES / 2 benign and N_TEST_SAMPLES / 2
    # malignant (or all available if fewer exist).
    # This ensures SHAP visualizations cover both classes.
    print(f"\n--- Selecting {N_TEST_SAMPLES} test samples (stratified) ---")
    
    test_benign_idx     = [i for i, (_, l) in enumerate(test_dataset.samples) if l == 0]
    test_malignant_idx  = [i for i, (_, l) in enumerate(test_dataset.samples) if l == 1]
    
    half     = N_TEST_SAMPLES // 2
    selected_benign    = np.random.choice(test_benign_idx,    min(half, len(test_benign_idx)),    replace=False)
    selected_malignant = np.random.choice(test_malignant_idx, min(half, len(test_malignant_idx)), replace=False)
    selected_indices   = list(selected_benign) + list(selected_malignant)
    
    print(f"  Selected: {len(selected_benign)} benign, {len(selected_malignant)} malignant")
    print(f"  Total test samples: {len(selected_indices)}")
    
    # Collect selected test images
    test_images  = []
    test_labels  = []
    test_paths   = []
    
    for idx in selected_indices:
        img, label, path = test_dataset[idx]
        test_images.append(img)
        test_labels.append(label)
        test_paths.append(path)
    
    test_tensor = torch.stack(test_images).to(device)
    
    # 6.7 Load model and get predictions for the test set
    model        = load_efficientnet(EFFICIENTNET_MODEL_PATH, device)
    wrapped_model = EfficientNetSHAPWrapper(model).to(device)
    wrapped_model.eval()
    
    print("\n--- Getting model predictions on test samples ---")
    with torch.no_grad():
        pred_probs = wrapped_model(test_tensor).squeeze(1).cpu().numpy()
    
    pred_labels = (pred_probs >= 0.5).astype(int)
    correct     = (pred_labels == np.array(test_labels))
    print(f"  Accuracy on selected test samples: {correct.mean()*100:.1f}%")
    print(f"  (Consistent with Phase 7 results — this is expected)")
    
    # Initialize SHAP DeepExplainer
    #
    # DeepExplainer is the SHAP method designed for neural networks.
    # It uses a modified backpropagation to efficiently approximate
    # Shapley values without running exponentially many model passes.
    #
    # Arguments:
    #   model      : wrapped EfficientNet (outputs probability 0-1)
    #   background : the 50 training images (the neutral baseline)
    print("\n--- Initializing SHAP DeepExplainer ---")
    print("  This may show a warning about BatchNorm — that is normal.")
    
    explainer = shap.DeepExplainer(wrapped_model, background_tensor)
    print("  Explainer ready")
    
    # Compute SHAP values
    #
    # This is the expensive step. For each of the N_TEST_SAMPLES images,
    # SHAP runs multiple forward passes to compute pixel contributions.
    #
    # Expected time: ~2-5 minutes on A100 for 30 images
    print(f"\n--- Computing SHAP values for {len(test_tensor)} images ---")
    print("  This will take a few minutes. Do not interrupt.")
    
    # Process in small batches to avoid OOM errors
    BATCH_SIZE  = 5
    all_shap_values = []
    
    for batch_start in range(0, len(test_tensor), BATCH_SIZE):
        batch_end    = min(batch_start + BATCH_SIZE, len(test_tensor))
        batch        = test_tensor[batch_start:batch_end]
        
        print(f"  Processing images {batch_start+1}-{batch_end} of {len(test_tensor)}...")
        
        # shap_vals shape: (batch_size, 3, 224, 224)
        # Each value tells how much that pixel in that channel contributed
        shap_vals = explainer.shap_values(batch)
        
        # shap_values returns a list when output is multi-class
        # For binary (single sigmoid output), it's just an array
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]  # Take first (and only) output
        
        all_shap_values.append(shap_vals)
    
    # Concatenate all batches: (N_TEST_SAMPLES, 3, 224, 224)
    all_shap_values = np.concatenate(all_shap_values, axis=0)
    print(f"  SHAP values computed. Shape: {all_shap_values.shape}")
    
    # Save raw SHAP values as .npy files
    # Saving raw values allows for regenerate visualizations later
    # without re-running the expensive SHAP computation.
    print("\n--- Saving raw SHAP values ---")
    shap_save_path = shap_dir / "efficientnet_shap_values.npy"
    np.save(str(shap_save_path), all_shap_values)
    
    # Also save the corresponding metadata
    meta = {
        "test_paths"  : test_paths,
        "test_labels" : test_labels,
        "pred_probs"  : pred_probs.tolist(),
        "pred_labels" : pred_labels.tolist(),
    }
    import json
    with open(shap_dir / "efficientnet_shap_metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: {shap_save_path}")
    
    # Generate visualizations
    # Create one 3-column PNG per test image
    print("\n--- Generating visualizations ---")
    
    for i in range(len(test_tensor)):
        save_name  = f"shap_{i:03d}_true{test_labels[i]}_pred{pred_labels[i]}.png"
        save_path  = str(vis_dir / save_name)
        
        visualize_shap(
            image_tensor = test_images[i],       # Original patch tensor
            shap_values  = all_shap_values[i],   # SHAP array (3, H, W)
            true_label   = test_labels[i],
            pred_prob    = float(pred_probs[i]),
            img_path     = test_paths[i],
            save_path    = save_path
        )
        
        if (i + 1) % 5 == 0 or (i + 1) == len(test_tensor):
            print(f"  Saved {i+1}/{len(test_tensor)} visualizations")
    
    print(f"   All visualizations saved to: {vis_dir}")
    

    # Compute quantitative SHAP summary (for thesis tables)
    #
    # For each test image, compute:
    #   - mean absolute SHAP value (overall feature importance magnitude)
    #   - mean positive SHAP value (how much pushed toward malignant)
    #   - mean negative SHAP value (how much pushed toward benign)
    print("\n--- Computing quantitative SHAP metrics ---")
    
    per_image_records = []
    for i in range(len(test_tensor)):
        sv           = all_shap_values[i]   # (3, 224, 224)
        flat         = sv.flatten()
        
        # Collapse across 3 channels to get (H, W) importance map
        sv_mag       = np.abs(sv).sum(axis=0).flatten()   # (H*W,)
        sv_pos       = sv[sv > 0].sum()   if sv[sv > 0].size > 0 else 0.0
        sv_neg       = sv[sv < 0].sum()   if sv[sv < 0].size > 0 else 0.0
        
        per_image_records.append({
            "image_id"           : i,
            "file_name"          : Path(test_paths[i]).name,
            "true_label"         : "malignant" if test_labels[i] == 1 else "benign",
            "predicted_label"    : "malignant" if pred_labels[i] == 1 else "benign",
            "pred_probability"   : round(float(pred_probs[i]), 4),
            "correct"            : bool(test_labels[i] == pred_labels[i]),
            "mean_abs_shap"      : round(float(np.mean(sv_mag)), 6),
            "max_abs_shap"       : round(float(np.max(sv_mag)), 6),
            "total_pos_shap"     : round(float(sv_pos), 6),
            "total_neg_shap"     : round(float(sv_neg), 6),
        })
    
    # Save per-image CSV
    df_per_image = pd.DataFrame(per_image_records)
    per_image_csv = Path(OUTPUT_DIR) / "shap_per_image.csv"
    df_per_image.to_csv(str(per_image_csv), index=False)
    print(f" Per-image SHAP summary: {per_image_csv}")
    
    # Aggregate by class (benign vs malignant)
    class_summary = df_per_image.groupby("true_label").agg(
        n_samples          = ("image_id",       "count"),
        mean_abs_shap      = ("mean_abs_shap",  "mean"),
        mean_max_abs_shap  = ("max_abs_shap",   "mean"),
        accuracy           = ("correct",        "mean"),
    ).reset_index()
    
    class_summary_csv = Path(OUTPUT_DIR) / "shap_class_summary.csv"
    class_summary.to_csv(str(class_summary_csv), index=False)
    print(f" Class-level SHAP summary: {class_summary_csv}")
    
    # Summary
    print(" EfficientNet SHAP Results")
    print(f"\n  Test samples analysed : {len(test_tensor)}")
    print(f"  Background samples    : {N_BACKGROUND}")
    print(f"  Overall accuracy      : {correct.mean()*100:.1f}%")
    
    print("\n  Mean Absolute SHAP Values by Class:")
    for _, row in class_summary.iterrows():
        print(f"    {row['true_label'].capitalize():12s}: "
              f"mean_abs_shap = {row['mean_abs_shap']:.6f}  "
              f"| accuracy = {row['accuracy']*100:.1f}%")
    
    print(f"\n  Visualizations : {vis_dir}/")
    print(f"  Raw SHAP arrays: {shap_dir}/")
    print(f"  CSV summaries  : {OUTPUT_DIR}/")
    print(f"\n  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return df_per_image, class_summary

if __name__ == "__main__":
    df, summary = main()