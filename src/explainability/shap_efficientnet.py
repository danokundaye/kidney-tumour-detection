# Phase 8 - SHAP explainability for EfficientNet-B0 Classification Model
#
#   This script generates pixel level attribution maps showing which image regions
#   influenced efficientnet b0 classification decisions on test patches.
#   It uses shap deep explainer with training patches as background baseline.
#
# Sampling strategy:
#   High dice cases (meaningful segmentation):
#     case_00088  10 patches  dice 0.6820
#     case_00293  10 patches  dice 0.2183
#     case_00292  10 patches  dice 0.1060
#     case_00146  10 patches  dice 0.1155
#   Zero dice cases (contrast):
#     case_00049   5 patches  dice 0.0000
#     case_00001   5 patches  dice 0.0000
#   Total: 50 test patches
#   Background: 50 patches sampled from efficientnet_train.csv
#
# Note on patch quality:
#   - Phase 9 patches are flat pngs with no tier label in the filename.
#   - Cases were selected based on dice score and contour patch count
#     from predictions.csv. patches from zero dice cases may include
#     fallback crops outside efficientnet training distribution.
#   - This is documented as a limitation in chapter 4.
#
# Output Structure:
#   results/phase8_shap/efficientnet/
#   ├── visualizations/
#   │   └── case_xxxxx_slice_xxxx.png   3 column: original / shap / overlay
#   ├── shap_values/
#   │   ├── shap_values.npy             raw shap arrays (n_test x 3 x 224 x 224)
#   │   └── shap_metadata.json          patch paths, labels, predictions
#   ├── shap_per_patch.csv              per patch shap magnitude metrics
#   └── shap_case_summary.csv           aggregated metrics per case
#
# Execution: Google Colab


import os
import sys
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
import shap
import cv2
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from PIL import Image
from torchvision import models, transforms
from datetime import datetime


def load_config(config_path: str) -> dict:
    """
    load the config.yaml file.
    all paths and settings come from here.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_efficientnet(path: str, device: torch.device) -> nn.Module:
    """
    Rebuild efficientnet b0 with same classifier head as phase 7,
    Then load saved weights.
    """
    print(f"  Loading EfficientNet-B0 from: {path}")
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 1)
    )
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    # Disable inplace activations to prevent SHAP deep explainer conflict.
    # EfficientNet uses SiLU with inplace=True which causes gradient errors
    # when SHAP hooks into the backward pass.
    for module in model.modules():
        if hasattr(module, 'inplace'):
            module.inplace = False

    print(f"  EfficientNet loaded")
    return model


class EfficientNetWrapper(nn.Module):
    """
    Wraps EfficientNet-B0 to output sigmoid probabilities instead of raw logits.
    SHAP deep explainer requires the model to output probabilities directly
    so that SHAPley values reflect contribution to the final prediction score.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


def get_transform(img_size: int) -> transforms.Compose:
    """
    ImageNet normalisation transform matching phase 7 training exactly.
    Must be identical to training or shap values will be misleading.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])


def load_patch_tensor(path: str, transform) -> torch.Tensor:
    """
    Load a single patch png and apply the imagenet transform.
    Returns a tensor of shape (1, 3, 224, 224).
    """
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse imagenet normalisation for display purposes.
    Converts tensor (3, h, w) to uint8 numpy array (h, w, 3).
    """
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.permute(1, 2, 0).cpu().numpy()
    img  = img * std + mean
    img  = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def sample_background_patches(train_csv: str, n: int, seed: int,
                               transform) -> torch.Tensor:
    """
    Sample n patches from efficientnet training set to use as shap background.
    The background represents the neutral baseline that shap measures
    attribution relative to. 
    Using training patches is standard practice.
    """
    df      = pd.read_csv(train_csv)
    sampled = df.sample(n = n, random_state = seed)

    tensors = []
    skipped = 0
    for path in sampled['patch_path']:
        if Path(path).exists():
            tensors.append(load_patch_tensor(path, transform))
        else:
            skipped += 1

    if skipped > 0:
        print(f"  Warning: {skipped} background patches not found, skipped")

    print(f"  Background patches loaded: {len(tensors)}")
    return torch.cat(tensors, dim = 0)


def sample_test_patches(patches_dir: str, case_samples: dict,
                        transform) -> tuple:
    """
    Sample patches from selected test cases.

    Args:
      patches_dir   path to phase9_pipeline/patches/
      case_samples  dict of {case_id: n_patches} defining sampling per case
      transform     imagenet transform

    Returns:
      tensors       stacked tensor (n_total, 3, 224, 224)
      metadata      list of dicts with case_id and slice_name per patch
    """
    tensors  = []
    metadata = []

    for case_id, n in case_samples.items():
        case_dir = Path(patches_dir) / case_id
        if not case_dir.exists():
            print(f"  Warning: {case_id} not found in patches dir, skipping")
            continue

        all_patches = sorted(case_dir.glob('*.png'))
        if len(all_patches) == 0:
            print(f"  Warning: no patches found for {case_id}, skipping")
            continue

        # Sample evenly across the case slices rather than taking first n
        # This avoids over representing a narrow slice range
        indices  = np.linspace(0, len(all_patches) - 1, n, dtype=int)
        selected = [all_patches[i] for i in indices]

        for patch_path in selected:
            tensors.append(load_patch_tensor(str(patch_path), transform))
            metadata.append({
                'case_id'   : case_id,
                'slice_name': patch_path.stem,
                'patch_path': str(patch_path),
            })

        print(f"  {case_id}: {len(selected)} patches sampled from {len(all_patches)} available")

    return torch.cat(tensors, dim=0), metadata


def run_inference(model: nn.Module, tensors: torch.Tensor,
                  device: torch.device, threshold: float) -> list:
    """
    Run efficientnet on all test patches and return per patch results.
    Returns list of dicts with prob_malignant and pred_label.
    """
    results = []
    model.eval()
    with torch.no_grad():
        for i in range(len(tensors)):
            inp  = tensors[i].unsqueeze(0).to(device)
            prob = torch.sigmoid(model(inp)).item()
            results.append({
                'prob_malignant': round(prob, 4),
                'pred_label'    : 'malignant' if prob >= threshold else 'benign',
            })
    return results


def compute_shap_values(wrapper: nn.Module, background: torch.Tensor,
                        test_tensors: torch.Tensor,
                        device: torch.device,
                        batch_size: int = 5) -> np.ndarray:
    """
    compute SHAP gradient explainer values for all test patches.

    Use gradient explainer instead of deep explainer because EfficientNet
    uses inplace operations (SiLU activations and residual += additions)
    that conflict with deep explainer's backward hooks.

    Processes in batches to avoid out of memory errors on t4.
    Returns numpy array of shape (n_test, 3, 224, 224).

    SHAP values represent the contribution of each pixel channel
    to the model output relative to the background baseline.
    Positive values push prediction toward malignant.
    Negative values push prediction toward benign.
    """
    background = background.to(device)
    explainer  = shap.GradientExplainer(wrapper, background)

    all_shap = []
    n_test   = len(test_tensors)

    print(f"  Computing shap values for {n_test} patches in batches of {batch_size}")

    for start in range(0, n_test, batch_size):
        end   = min(start + batch_size, n_test)
        batch = test_tensors[start:end].to(device)

        shap_vals = explainer.shap_values(batch)

        # shap_vals shape: (n_batch, 3, 224, 224)
        # squeeze output class dimension if present
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        # remove any trailing dimensions of size 1
        shap_vals = np.squeeze(shap_vals)
        # ensure shape is (n_batch, 3, 224, 224)
        if shap_vals.ndim == 3:
            shap_vals = shap_vals[np.newaxis, ...]

        all_shap.append(shap_vals)
        print(f"  batch {start // batch_size + 1}: patches {start + 1} to {end} done")

    return np.concatenate(all_shap, axis=0)

def visualize_shap(test_tensors: torch.Tensor,
                   shap_values : np.ndarray,
                   metadata    : list,
                   predictions : list,
                   output_dir  : Path) -> None:
    """
    Generate 3 column visualization per patch:
      col 1  original patch (imagenet denormalized)
      col 2  shap magnitude heatmap (sum of absolute values across rgb channels)
      col 3  overlay (60% original + 40% heatmap)

    Title shows: case id | slice | predicted label (probability)
    """
    output_dir.mkdir(parents = True, exist_ok = True)

    for i in range(len(test_tensors)):
        original   = denormalize(test_tensors[i])
        shap_img   = shap_values[i]  # (3, 224, 224)

        # Sum absolute shap values across rgb channels to get magnitude map
        magnitude  = np.sum(np.abs(shap_img), axis = 0)  # (224, 224)

        # Normalize magnitude to 0 to 255 for display
        if magnitude.max() > 0:
            magnitude_norm = (magnitude / magnitude.max() * 255).astype(np.uint8)
        else:
            magnitude_norm = magnitude.astype(np.uint8)

        # Apply jet colormap to magnitude
        heatmap_bgr = cv2.applyColorMap(magnitude_norm, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Overlay: blend original and heatmap
        overlay = cv2.addWeighted(original, 0.6, heatmap_rgb, 0.4, 0)

        # Build title
        case_id    = metadata[i]['case_id']
        slice_name = metadata[i]['slice_name']
        prob       = predictions[i]['prob_malignant']
        pred_label = predictions[i]['pred_label']
        title      = f"{case_id} | {slice_name} | {pred_label} ({prob:.3f})"

        # Plot
        fig, axes = plt.subplots(1, 3, figsize = (15, 5))
        axes[0].imshow(original)
        axes[0].set_title('original patch', fontsize = 10)
        axes[0].axis('off')

        axes[1].imshow(heatmap_rgb)
        axes[1].set_title('shap magnitude', fontsize = 10)
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title('overlay', fontsize = 10)
        axes[2].axis('off')

        fig.suptitle(title, fontsize = 11, y = 1.01)
        plt.tight_layout()

        save_name = f"{case_id}_{slice_name}.png"
        plt.savefig(str(output_dir / save_name), bbox_inches = 'tight', dpi = 100)
        plt.close()

    print(f"  {len(test_tensors)} visualizations saved to {output_dir}")


def compute_patch_metrics(shap_values: np.ndarray,
                          metadata   : list,
                          predictions: list) -> pd.DataFrame:
    """
    Compute per patch shap magnitude metrics for chapter 4 reporting.

    Metrics per patch:
      mean_abs_shap   average absolute shap value across all pixels and channels
      max_abs_shap    maximum absolute shap value
      total_pos_shap  sum of positive shap values (push toward malignant)
      total_neg_shap  sum of negative shap values (push toward benign)
    """
    rows = []
    for i in range(len(shap_values)):
        sv = shap_values[i]  # (3, 224, 224)
        rows.append({
            'case_id'       : metadata[i]['case_id'],
            'slice_name'    : metadata[i]['slice_name'],
            'pred_label'    : predictions[i]['pred_label'],
            'prob_malignant': predictions[i]['prob_malignant'],
            'mean_abs_shap' : round(float(np.mean(np.abs(sv))), 6),
            'max_abs_shap'  : round(float(np.max(np.abs(sv))),  6),
            'total_pos_shap': round(float(sv[sv > 0].sum()),     6),
            'total_neg_shap': round(float(sv[sv < 0].sum()),     6),
        })
    return pd.DataFrame(rows)


def main():
    config_path = "/content/kidney-tumour-detection/configs/config.yaml"
    config      = load_config(config_path)

    print("Phase 8 - SHAP explainability for EfficientNet-B0")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # paths from config
    results_dir  = Path(config['paths']['results_dir'])
    splits_dir   = Path(config['paths']['splits_dir'])

    effnet_path  = results_dir / "phase7_efficientnet/weights/best.pt"
    patches_dir  = str(results_dir / "phase9_pipeline/patches")
    train_csv    = str(splits_dir  / "efficientnet_train.csv")

    output_dir   = results_dir / "phase8_shap/efficientnet"
    vis_dir      = output_dir  / "visualizations"
    shap_dir     = output_dir  / "shap_values"
    output_dir.mkdir(parents = True, exist_ok = True)
    vis_dir.mkdir(parents = True,    exist_ok = True)
    shap_dir.mkdir(parents = True,   exist_ok = True)

    # settings
    img_size   = config['preprocessing']['efficientnet_input_size']
    threshold  = config['efficientnet']['classification_threshold']
    seed       = config['dataset']['random_seed']

    # Cases to sample from and how many patches each
    # Selected based on dice score and contour patch count from phase 9
    case_samples = {
        'case_00088': 10,   # dice 0.6820 - best segmentation result
        'case_00293': 10,   # dice 0.2183
        'case_00292': 10,   # dice 0.1060
        'case_00146': 10,   # dice 0.1155
        'case_00049':  5,   # dice 0.0000 - contrast case
        'case_00001':  5,   # dice 0.0000 - contrast case
    }

    transform = get_transform(img_size)

    # load model
    print("\nLoading model")
    model   = load_efficientnet(str(effnet_path), device)
    wrapper = EfficientNetWrapper(model).to(device)
    wrapper.eval()

    # load background patches
    print("\nLoading background patches")
    background = sample_background_patches(train_csv, n = 50,
                                           seed = seed, transform = transform)

    # load test patches
    print("\nLoading test patches")
    test_tensors, metadata = sample_test_patches(patches_dir, case_samples, transform)
    print(f"  Total test patches loaded: {len(test_tensors)}")

    # run inference on test patches
    print("\n Running inference")
    predictions = run_inference(model, test_tensors, device, threshold)
    malignant_n = sum(1 for p in predictions if p['pred_label'] == 'malignant')
    print(f"  Malignant predictions: {malignant_n} / {len(predictions)}")

    # compute shap values
    print("\n Computing shap values")
    shap_values = compute_shap_values(wrapper, background, test_tensors,
                                      device, batch_size = 5)
    print(f"  Shap values shape: {shap_values.shape}")

    # save raw shap values and metadata
    np.save(str(shap_dir / "shap_values.npy"), shap_values)

    meta_out = []
    for i, m in enumerate(metadata):
        meta_out.append({**m, **predictions[i]})
    with open(str(shap_dir / "shap_metadata.json"), 'w') as f:
        json.dump(meta_out, f, indent = 2)
    print(f"Raw shap values saved to {shap_dir}")

    # generate visualizations
    print("\nGenerating visualizations")
    visualize_shap(test_tensors, shap_values, metadata, predictions, vis_dir)

    # compute and save metrics
    print("\nComputing patch metrics")
    patch_df = compute_patch_metrics(shap_values, metadata, predictions)
    patch_df.to_csv(str(output_dir / "shap_per_patch.csv"), index = False)

    # case level summary
    case_summary = patch_df.groupby('case_id').agg(
        n_patches     = ('slice_name',    'count'),
        mean_abs_shap = ('mean_abs_shap', 'mean'),
        max_abs_shap  = ('max_abs_shap',  'max'),
        total_pos_shap= ('total_pos_shap','sum'),
        total_neg_shap= ('total_neg_shap','sum'),
    ).reset_index().round(6)
    case_summary.to_csv(str(output_dir / "shap_case_summary.csv"), index = False)

    print(f"\n  Shap_per_patch.csv   : {len(patch_df)} rows")
    print(f"  Shap_case_summary.csv: {len(case_summary)} rows")

    print("Phase 8 complete")
    print(f"\nOutput directory: {output_dir}")
    print(f"Visualizations  : {vis_dir}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()