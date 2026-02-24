# Combined SHAP visualisation for all three models
#
#   This script generates SHAP attribution maps for yolo, unet and efficientnet
#   on two selected test cases to show which input pixels influenced
#   each model's decision at each pipeline stage.
#
# cases selected:
#   case_00088  dice 0.6820  best segmentation result
#   case_00001  dice 0.0000  contrast case (poor segmentation)
#
# model input reconstruction:
#   YOLO        512x512 slice from test slices directory
#   U-Net        256x256 crop using yolo predicted box on same slice
#   EfficientNet 224x224 patch from phase9_pipeline/patches
#
# SHAP approach:
#   YOLO        gradient explainer on yolo.model internal pytorch module
#               scalar output: max confidence score across all anchors
#   U-Net        gradient explainer on unet
#               scalar output: mean pixel probability across 256x256 map
#   EfficientNet gradient explainer on efficientnet wrapper
#               scalar output: sigmoid malignant probability
#
# output structure:
#   results/phase8_shap/combined/
#   ├── case_00088_slice_xxxx_combined.png
#   └── case_00001_slice_xxxx_combined.png
#       each figure has 3 rows (one per model) x 3 columns
#       (original input / SHAP magnitude / overlay)
#
# Execution: Google Colab


import os
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
import shap
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

from pathlib import Path
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO
import segmentation_models_pytorch as smp


def load_config(config_path: str) -> dict:
    """
    load the config.yaml file.
    all paths and settings come from here.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Model loaders

def load_yolo(path: str):
    """
    Load trained yolov8 model and return both the ultralytics wrapper
    and the internal pytorch detection model for SHAP.
    """
    yolo    = YOLO(path)
    internal = yolo.model
    internal.eval()
    for module in internal.modules():
        if hasattr(module, 'inplace'):
            module.inplace = False
    return yolo, internal


def load_unet(path: str, device: torch.device) -> nn.Module:
    """
    Rebuild unet with resnet50 encoder exactly as trained in phase 6,
    then load saved weights.
    """
    model = smp.Unet(
        encoder_name    = "resnet50",
        encoder_weights = None,
        in_channels     = 3,
        classes         = 1,
    )
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    for module in model.modules():
        if hasattr(module, 'inplace'):
            module.inplace = False
    return model


def load_efficientnet(path: str, device: torch.device) -> nn.Module:
    """
    Rebuild efficientnet b0 with same classifier head as phase 7,
    then load saved weights.
    """
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 1)
    )
    checkpoint = torch.load(path, map_location = device, weights_only = False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    for module in model.modules():
        if hasattr(module, 'inplace'):
            module.inplace = False
    return model


# Model wrappers for SHAP
# each wrapper reduces model output to a single scalar value
class YOLOShapWrapper(nn.Module):
    """
    Wraps yolo internal detection model to output a single scalar.
    Scalar: max confidence score across all 5376 anchor predictions.
    This tells SHAP which pixels most influenced the detection confidence.
    out[0] shape is (batch, 5, 5376) where index 4 is confidence.
    """
    def __init__(self, internal_model: nn.Module):
        super().__init__()
        self.model = internal_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        # out is a tuple, out[0] shape: (batch, 5, 5376)
        # index 4 across dim 1 is confidence score
        conf = out[0][:, 4, :]   # (batch, 5376)
        return conf.max(dim = 1, keepdim = True)[0]  # (batch, 1)


class UNetShapWrapper(nn.Module):
    """
    Wraps unet to output a single scalar per image.
    Scalar: mean pixel probability across the 256x256 output map.
    This tells SHAP which input pixels most influenced the overall
    segmentation response, summarised across all output pixels.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out  = self.model(x)               # (batch, 1, 256, 256)
        prob = torch.sigmoid(out)
        return prob.mean(dim = (1, 2, 3), keepdim = False).unsqueeze(1)  # (batch, 1)


class EfficientNetShapWrapper(nn.Module):
    """
    Wraps efficientnet b0 to output sigmoid malignant probability.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


# Transforms
def get_yolo_transform(size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std  = [0.229, 0.224, 0.225])
    ])


def get_unet_transform(size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std  = [0.229, 0.224, 0.225])
    ])


def get_effnet_transform(size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std  = [0.229, 0.224, 0.225])
    ])


# Helper functions
def expand_box(x1, y1, x2, y2, margin, img_w, img_h):
    """Expand bounding box by fractional margin, clamped to image boundary."""
    bw = x2 - x1
    bh = y2 - y1
    x1 = max(0,     int(x1 - bw * margin))
    y1 = max(0,     int(y1 - bh * margin))
    x2 = min(img_w, int(x2 + bw * margin))
    y2 = min(img_h, int(y2 + bh * margin))
    return x1, y1, x2, y2


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalisation for display. returns uint8 (h, w, 3)."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.permute(1, 2, 0).cpu().numpy()
    img  = img * std + mean
    img  = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def make_heatmap(shap_vals: np.ndarray) -> np.ndarray:
    """
    Convert raw SHAP values to a jet colormap heatmap.
    Sums absolute values across rgb channels to get magnitude map,
    Normalizes to 0-255, applies jet colormap.
    
    returns uint8 rgb array (h, w, 3).
    """
    if shap_vals.ndim == 4:
        shap_vals = shap_vals.squeeze(-1)
    magnitude = np.sum(np.abs(shap_vals), axis=0)  # (h, w)
    if magnitude.max() > 0:
        magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
    else:
        magnitude = magnitude.astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(magnitude, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def compute_shap_for_input(wrapper: nn.Module, background: torch.Tensor,
                           test_input: torch.Tensor,
                           device: torch.device) -> np.ndarray:
    """
    Compute gradient explainer SHAP values for a single input tensor.
    returns numpy array of shape (3, h, w) or (3, h, w, 1).
    """
    background = background.to(device)
    test_input = test_input.to(device)
    explainer  = shap.GradientExplainer(wrapper, background)
    shap_vals  = explainer.shap_values(test_input)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    return np.squeeze(shap_vals)


def get_representative_slice(slices_dir: Path, yolo_model,
                             yolo_conf: float) -> tuple:
    """
    Find a representative slice for a case by picking the slice where
    YOLO detects a kidney with highest confidence. Falls back to the
    middle slice if no detection found.

    returns (slice_path, crop_box or none).
    """
    all_slices  = sorted(slices_dir.glob('*.png'))
    best_slice  = all_slices[len(all_slices) // 2]  # default: middle slice
    best_conf   = 0.0
    best_box    = None

    # sample every 10th slice to find best detection without running all 512
    for slice_path in all_slices[::10]:
        results = yolo_model.predict(str(slice_path), conf = yolo_conf, verbose = False)
        boxes   = results[0].boxes
        if len(boxes) > 0:
            idx  = boxes.conf.argmax().item()
            conf = boxes.conf[idx].item()
            if conf > best_conf:
                best_conf  = conf
                best_slice = slice_path
                x1, y1, x2, y2 = boxes.xyxy[idx].tolist()
                best_box   = (x1, y1, x2, y2)

    return best_slice, best_box


# Background builders
# Each model needs its own background set in the correct input space

def build_yolo_background(slices_dir: Path, transform,
                          n: int, seed: int) -> torch.Tensor:
    """sample N slices as YOLO background baseline."""
    np.random.seed(seed)
    all_slices = sorted(slices_dir.glob('*.png'))
    indices    = np.random.choice(len(all_slices), size = min(n, len(all_slices)),
                                  replace = False)
    tensors = []
    for i in indices:
        img = Image.open(str(all_slices[i])).convert('RGB')
        tensors.append(transform(img).unsqueeze(0))
    return torch.cat(tensors, dim = 0)


def build_unet_background(slices_dir: Path, yolo_model,
                          transform, yolo_conf: float,
                          yolo_margin: float, unet_size: int,
                          n: int, seed: int,
                          device: torch.device) -> torch.Tensor:
    """
    Sample N crops as unet background baseline.
    Uses YOLO to find kidney region on sampled slices.
    Falls back to full slice resize if YOLO finds nothing.
    """
    np.random.seed(seed)
    all_slices = sorted(slices_dir.glob('*.png'))
    indices    = np.random.choice(len(all_slices), size = min(n, len(all_slices)),
                                  replace = False)
    tensors = []
    for i in indices:
        slice_path = str(all_slices[i])
        img_pil    = Image.open(slice_path).convert('RGB')
        img_w, img_h = img_pil.size

        results = yolo_model.predict(slice_path, conf=yolo_conf, verbose=False)
        boxes   = results[0].boxes

        if len(boxes) > 0:
            idx             = boxes.conf.argmax().item()
            x1, y1, x2, y2 = boxes.xyxy[idx].tolist()
            x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, yolo_margin, img_w, img_h)
            crop = img_pil.crop((x1, y1, x2, y2))
        else:
            crop = img_pil

        crop = crop.resize((unet_size, unet_size), Image.BILINEAR)
        tensors.append(transform(crop).unsqueeze(0))

    return torch.cat(tensors, dim = 0)


def build_effnet_background(train_csv: str, transform,
                            n: int, seed: int) -> torch.Tensor:
    """sample N training patches as efficientnet background baseline."""
    df      = pd.read_csv(train_csv)
    sampled = df.sample(n = n, random_state = seed)
    tensors = []
    for path in sampled['patch_path']:
        if Path(path).exists():
            img = Image.open(path).convert('RGB')
            tensors.append(transform(img).unsqueeze(0))
    return torch.cat(tensors, dim = 0)


# Combined visualisation

def visualise_combined(case_id      : str,
                       slice_path   : Path,
                       yolo_tensor  : torch.Tensor,
                       unet_tensor  : torch.Tensor,
                       effnet_tensor: torch.Tensor,
                       yolo_shap    : np.ndarray,
                       unet_shap    : np.ndarray,
                       effnet_shap  : np.ndarray,
                       output_dir   : Path) -> None:
    """
    Generate a 3 row x 3 column combined figure.
    row 1  YOLO    (512x512 slice input)
    row 2  U-Net    (256x256 crop input)
    row 3  EfficientNet (224x224 patch input)

    columns: original / SHAP magnitude / overlay
    """
    rows = [
        ('yolo detection',       yolo_tensor,   yolo_shap),
        ('unet segmentation',    unet_tensor,   unet_shap),
        ('efficientnet classification', effnet_tensor, effnet_shap),
    ]

    fig = plt.figure(figsize = (15, 15))
    gs  = gridspec.GridSpec(3, 3, figure = fig, hspace = 0.35, wspace = 0.1)

    for row_idx, (label, tensor, shap_vals) in enumerate(rows):
        original = denormalize(tensor.squeeze(0))
        heatmap  = make_heatmap(shap_vals)

        # resize heatmap to match original for overlay
        h, w     = original.shape[:2]
        heatmap  = cv2.resize(heatmap, (w, h))
        overlay  = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

        for col_idx, (img, col_title) in enumerate([
            (original, 'Original input'),
            (heatmap,  'SHAP magnitude'),
            (overlay,  'Overlay'),
        ]):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(img)
            if row_idx == 0:
                ax.set_title(col_title, fontsize = 10, fontweight = 'bold')
            ax.set_ylabel(label, fontsize=9) if col_idx == 0 else None
            ax.axis('off')

    fig.suptitle(f'{case_id} | {slice_path.stem} | Combined SHAP attribution',
                 fontsize=12, y=0.98)

    output_dir.mkdir(parents = True, exist_ok = True)
    save_path = output_dir / f'{case_id}_{slice_path.stem}_combined.png'
    plt.savefig(str(save_path), bbox_inches = 'tight', dpi = 100)
    plt.close()
    print(f"  Saved: {save_path.name}")


# Main
def main():
    config_path = "/content/kidney-tumour-detection/configs/config.yaml"
    config      = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    results_dir = Path(config['paths']['results_dir'])
    splits_dir  = Path(config['paths']['splits_dir'])
    slices_dir  = Path(config['paths']['slices_dir']) / 'test'

    yolo_path    = results_dir / "phase5_yolo_retrain/yolov8s_retrain_run1/weights/best.pt"
    unet_path    = results_dir / "phase6_unet/weights/best.pt"
    effnet_path  = results_dir / "phase7_efficientnet/weights/best.pt"
    train_csv    = str(splits_dir / "efficientnet_train.csv")
    patches_dir  = results_dir / "phase9_pipeline/patches"
    output_dir   = results_dir / "phase8_shap/combined"

    yolo_conf   = 0.10
    yolo_margin = config['preprocessing']['bbox_margin']
    unet_size   = config['preprocessing']['unet_input_size']
    effnet_size = config['preprocessing']['efficientnet_input_size']
    seed        = config['dataset']['random_seed']
    n_bg        = 20   # background samples per model, kept small for t4

    # transforms
    yolo_tf   = get_yolo_transform(512)
    unet_tf   = get_unet_transform(unet_size)
    effnet_tf = get_effnet_transform(effnet_size)

    # load models
    print("Loading models")
    yolo_model, yolo_internal = load_yolo(str(yolo_path))
    unet_model                = load_unet(str(unet_path),    device)
    effnet_model              = load_efficientnet(str(effnet_path), device)

    # wrappers
    yolo_wrapper   = YOLOShapWrapper(yolo_internal).to(device)
    unet_wrapper   = UNetShapWrapper(unet_model).to(device)
    effnet_wrapper = EfficientNetShapWrapper(effnet_model).to(device)

    for w in [yolo_wrapper, unet_wrapper, effnet_wrapper]:
        w.eval()

    # cases to process
    cases = {
        'case_00088': {'Dice': 0.6820, 'Note': 'best segmentation result'},
        'case_00001': {'Dice': 0.0000, 'Note': 'contrast case poor segmentation'},
    }

    for case_id, info in cases.items():
        print(f"\nProcessing {case_id} (dice={info['dice']}, {info['note']})")

        case_slices_dir = slices_dir / case_id / 'images'
        case_patches_dir = patches_dir / case_id

        # find representative slice using yolo detection
        print("  Finding representative slice...")
        slice_path, best_box = get_representative_slice(
            case_slices_dir, yolo_model, yolo_conf)
        print(f"  Selected: {slice_path.name}")

        img_pil  = Image.open(str(slice_path)).convert('RGB')
        img_w, img_h = img_pil.size

        # YOLO input: full 512x512 slice
        yolo_tensor = yolo_tf(img_pil).unsqueeze(0)

        # U-Net input: cropped kidney region resized to 256x256
        if best_box is not None:
            x1, y1, x2, y2 = expand_box(*best_box, yolo_margin, img_w, img_h)
            crop_pil = img_pil.crop((x1, y1, x2, y2))
        else:
            crop_pil = img_pil
        crop_pil    = crop_pil.resize((unet_size, unet_size), Image.BILINEAR)
        unet_tensor = unet_tf(crop_pil).unsqueeze(0)

        # EfficientNet input: corresponding patch from phase9 patches
        # use the same slice name if available, else first patch
        patch_path = case_patches_dir / slice_path.name
        if not patch_path.exists():
            available = sorted(case_patches_dir.glob('*.png'))
            patch_path = available[len(available) // 2]
        effnet_tensor = effnet_tf(Image.open(str(patch_path)).convert('RGB')).unsqueeze(0)

        print(f"  efficientnet patch: {patch_path.name}")

        # build backgrounds
        print("  Building backgrounds...")
        yolo_bg   = build_yolo_background(case_slices_dir, yolo_tf, n_bg, seed)
        unet_bg   = build_unet_background(case_slices_dir, yolo_model, unet_tf,
                                          yolo_conf, yolo_margin, unet_size,
                                          n_bg, seed, device)
        effnet_bg = build_effnet_background(train_csv, effnet_tf, n_bg, seed)

        # Compute SHAP values for each model
        print("  computing YOLO SHAP...")
        yolo_shap = compute_shap_for_input(yolo_wrapper, yolo_bg,
                                           yolo_tensor, device)

        print("  computing U-Net SHAP...")
        unet_shap = compute_shap_for_input(unet_wrapper, unet_bg,
                                           unet_tensor, device)

        print("  Computing EfficientNet SHAP...")
        effnet_shap = compute_shap_for_input(effnet_wrapper, effnet_bg,
                                             effnet_tensor, device)

        # generate combined visualisation
        print("  Generating visualisation...")
        visualise_combined(
            case_id       = case_id,
            slice_path    = slice_path,
            yolo_tensor   = yolo_tensor,
            unet_tensor   = unet_tensor,
            effnet_tensor = effnet_tensor,
            yolo_shap     = yolo_shap,
            unet_shap     = unet_shap,
            effnet_shap   = effnet_shap,
            output_dir    = output_dir,
        )

    print(f"\nCombined SHAP complete")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()