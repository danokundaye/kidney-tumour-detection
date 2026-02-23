# Step 7.1 — EfficientNet Patch Preparation
#
# This script extracts 224x224 patches from abnormal regions across all
# 120 segmentation train cases using a hybrid approach:
#
#   33 cases with unet_crops:
#     - Run U-Net inference on each abnormal slice
#     - Dice >= dice_threshold → extract patch from U-Net predicted region
#     - Dice <  dice_threshold → extract patch from GT mask region
#
#   87 crop cases with no abnormal tissue visible:
#     - Load directly from slices/segmentation_train
#     - Extract patch from GT mask region
#
# Execution: Google Colab
#
# OUTPUT STRUCTURE:
#   processed/patches/
#   ├── case_00295/
#   │   └── malignant/
#   │       ├── slice_0160.png
#   │       └── slice_0161.png
#   ├── case_00010/
#   │   └── benign/
#   │       └── slice_0045.png
#   └── ...
#
#   processed/splits/patches_index.csv
#   columns: patch_path, case_id, malignant, slice_name, source, dice

import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm

# Load config.yaml
def load_config(config_path: str) -> dict:
    """
    Load the config.yaml file
    All paths and settings come from here
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Load U-Net helpers
def load_unet(checkpoint, device):
    model = smp.Unet(
        encoder_name    = "resnet50",
        encoder_weights = None,
        in_channels     = 3,
        classes         = 1,
        activation      = None
    )
    ckpt  = torch.load(checkpoint, map_location = device)
    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"U-Net loaded. Device: {device}")
    return model

def preprocess_for_unet(img_path, device):
    """
    Normalize to match Phase 6 training: (pixel/255 - (mean) 0.485) / (std) 0.229.
    """
    img       = Image.open(img_path).convert("L")
    img_float = np.array(img, dtype=np.float32) / 255.0
    img_norm  = (img_float - 0.485) / 0.229
    tensor    = torch.from_numpy(np.stack([img_norm]*3) ).unsqueeze(0).to(device)
    return tensor

@torch.no_grad()
def run_unet(model, img_path, device):
    """
    Run U-Net inference. Returns binary mask (256x256) uint8.
    """
    tensor = preprocess_for_unet(img_path, device)
    prob   = torch.sigmoid(model(tensor))
    return (prob.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

def compute_dice(pred, gt_binary):
    """
    Dice coefficient between prediction and binary GT mask
    """
    g = (gt_binary > 0).astype(np.uint8)
    s = pred.sum() + g.sum()
    return float(2 * (pred * g).sum() / s) if s > 0 else 0.0


# Patch extraction helpers
def get_bounding_box(mask, img_hw, box_margin):
    """
    Get bounding box of non-zero pixels in mask, expanded by box_margin.
    Returns (x1, y1, x2, y2) in original image coordinates, or None if empty.
    """
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)

    if not rows.any():
        return None

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    h, w   = mask.shape
    oh, ow = img_hw

    # Scale from mask coordinates to original image coordinates
    rmin = int(rmin * oh / h)
    rmax = int(rmax * oh / h)
    cmin = int(cmin * ow / w)
    cmax = int(cmax * ow / w)

    # Expand by margin
    h_margin = int((rmax - rmin) * box_margin)
    w_margin = int((cmax - cmin) * box_margin)

    rmin = max(0,  rmin - h_margin)
    rmax = min(oh, rmax + h_margin)
    cmin = max(0,  cmin - w_margin)
    cmax = min(ow, cmax + w_margin)

    if (rmax - rmin) < 10 or (cmax - cmin) < 10:
        return None

    return (cmin, rmin, cmax, rmax)  # (x1, y1, x2, y2) for PIL crop

def extract_patch(img_path, box, patch_size):
    """
    Crop region from CT image and resize to patch_size x patch_size.
    """
    img   = Image.open(img_path).convert("L")
    patch = img.crop(box)
    patch = patch.resize((patch_size, patch_size), Image.BILINEAR)
    return patch


# Case processors
def process_unet_crops_cases(
        model, 
        label_map, 
        crops_dir, 
        slices_dir,
        patches_dir, 
        unet_crop_case_ids,
        dice_threshold, 
        patch_size, 
        box_margin,
        device, 
        abnormal_val):
    """
    Process 33 cases with abnormal content in unet_crops.
    Uses hybrid: U-Net prediction if Dice >= dice_threshold, else GT mask.
    """
    records = []
    print(f"\n[unet_crops] Processing {len(unet_crop_case_ids)} cases...")

    # Loop through each case and assign a class
    for case_id in tqdm(sorted(unet_crop_case_ids), desc="unet_crops cases"):
        malignant = label_map.get(case_id)
        if malignant is None:
            continue

        case_dir        = crops_dir / case_id
        images_dir      = case_dir / "images"
        masks_dir       = case_dir / "masks"
        orig_images_dir = slices_dir / case_id / "images"
        class_name      = "malignant" if malignant else "benign"

        for mask_file in sorted(masks_dir.glob("*.png")):
            stem    = mask_file.stem
            gt_crop = np.array(Image.open(mask_file).convert("L"))

            if not np.any(gt_crop == abnormal_val):
                continue

            img_crop_path = images_dir / f"{stem}.png"
            if not img_crop_path.exists():
                continue

            # Run inference and compute Dice
            pred_mask = run_unet(model, img_crop_path, device)
            gt_binary = (gt_crop == abnormal_val).astype(np.uint8)
            dice      = compute_dice(pred_mask, gt_binary)

            # Choose source
            if dice >= dice_threshold:
                source   = 'unet'
                use_mask = pred_mask
            else:
                source   = 'gt'
                use_mask = gt_binary

            # Extract from original full-resolution slice if available
            orig_img_path = orig_images_dir / f"{stem}.png"
            if not orig_img_path.exists():
                orig_img_path = img_crop_path

            orig_img = Image.open(orig_img_path).convert("L")
            img_hw   = (orig_img.height, orig_img.width)
            box      = get_bounding_box(use_mask, img_hw, box_margin)

            if box is None:
                continue

            patch    = extract_patch(orig_img_path, box, patch_size)
            out_dir  = patches_dir / case_id / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{stem}.png"
            patch.save(out_path)

            records.append({
                'patch_path': str(out_path),
                'case_id'   : case_id,
                'malignant' : malignant,
                'slice_name': stem,
                'source'    : source,
                'dice'      : round(dice, 4),
            })

    return records


def process_empty_crop_cases(
        label_map, 
        slices_dir, 
        patches_dir,
        unet_crop_case_ids, 
        patch_size, 
        box_margin,
        tumour_val, 
        cyst_val):
    """
    Process 87 cases with unet_crops missing abnormal tissue.
    Uses GT masks from slices/segmentation_train directly.
    """
    records     = []
    empty_cases = [
        d.name for d in sorted(slices_dir.iterdir())
        if d.is_dir() and d.name not in unet_crop_case_ids
    ]
    print(f"\n[abnormal tissue missing crops] Processing {len(empty_cases)} cases from raw slices...")

    for case_id in tqdm(empty_cases, desc="missing abnormal crop cases"):
        malignant  = label_map.get(case_id)
        if malignant is None:
            continue

        images_dir = slices_dir / case_id / "images"
        masks_dir  = slices_dir / case_id / "masks"
        class_name = "malignant" if malignant else "benign"

        if not masks_dir.exists():
            continue

        for mask_file in sorted(masks_dir.glob("*.png")):
            stem    = mask_file.stem
            gt_mask = np.array(Image.open(mask_file).convert("L"))

            if not (np.any(gt_mask == tumour_val) or np.any(gt_mask == cyst_val)):
                continue

            gt_binary = np.zeros_like(gt_mask, dtype=np.uint8)
            gt_binary[(gt_mask == tumour_val) | (gt_mask == cyst_val)] = 1

            img_path = images_dir / f"{stem}.png"
            if not img_path.exists():
                continue

            orig_img = Image.open(img_path).convert("L")
            img_hw   = (orig_img.height, orig_img.width)
            box      = get_bounding_box(gt_binary, img_hw, box_margin)

            if box is None:
                continue

            patch    = extract_patch(img_path, box, patch_size)
            out_dir  = patches_dir / case_id / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{stem}.png"
            patch.save(out_path)

            records.append({
                'patch_path': str(out_path),
                'case_id'   : case_id,
                'malignant' : malignant,
                'slice_name': stem,
                'source'    : 'gt',
                'dice'      : None,
            })

    return records


# Main
def main():
    config_path = "/content/kidney-tumour-detection/configs/config.yaml"
    config      = load_config(config_path)

    # Paths
    crops_dir   = Path(config['paths']['unet_crops_dir'])
    slices_dir  = Path(config['paths']['slices_dir']) / "segmentation_train"
    patches_dir = Path(config['paths']['patches_dir'])
    splits_dir  = Path(config['paths']['splits_dir'])
    checkpoint  = Path(config['paths']['checkpoints_dir']) / "phase6_unet" / "best.pt"
    kits_json   = Path(config['paths']['kits_root']) / "kits.json"

    # Settings
    dice_threshold = config['classification']['dice_threshold']
    patch_size     = config['classification']['patch_size']
    box_margin     = config['classification']['box_margin']
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mask pixel values
    tumour_val   = 170
    cyst_val     = 255
    abnormal_val = 255  # unet_crops masks are binary — abnormal = 255

    print("Step 7.1 — EfficientNet Patch Preparation")

    # Load labels
    with open(kits_json) as f:
        kits = json.load(f)
    label_map = {c['case_id']: c.get('malignant', None) for c in kits}

    # Identify unet_crops cases with abnormal content
    unet_crop_case_ids = set()
    for case_dir in sorted(crops_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        has_abnormal = any(
            np.any(np.array(Image.open(f).convert("L")) == abnormal_val)
            for f in (case_dir / "masks").glob("*.png")
        )
        if has_abnormal:
            unet_crop_case_ids.add(case_dir.name)

    print(f"Cases with unet_crops content : {len(unet_crop_case_ids)}")
    print(f"Cases using raw slices        : estimated {120 - len(unet_crop_case_ids)}")

    # Load U-Net for hybrid processing
    model = load_unet(checkpoint, device)

    # Process both groups
    records  = process_unet_crops_cases(
                    model, label_map, crops_dir, slices_dir, patches_dir,
                    unet_crop_case_ids, dice_threshold, patch_size,
                    box_margin, device, abnormal_val)

    records += process_empty_crop_cases(
                    label_map, slices_dir, patches_dir, unet_crop_case_ids,
                    patch_size, box_margin, tumour_val, cyst_val)

    # Save index
    df      = pd.DataFrame(records)
    out_csv = splits_dir / "patches_index.csv"
    df.to_csv(out_csv, index=False)

    # Summary
    print("Patch Preparation Complete")
    print(f"Total patches : {len(df)}")
    print(f"\nBy class:")
    print(df['malignant'].value_counts().rename({True: 'malignant', False: 'benign'}))
    print(f"\nBy source:")
    print(df['source'].value_counts())
    unet_used = df[df['source'] == 'unet']
    if len(unet_used) > 0:
        print(f"\nMean Dice (U-Net sourced patches): {unet_used['dice'].mean():.4f}")
    print(f"\nIndex saved to: {out_csv}")
    print("\nDone.")

if __name__ == "__main__":
    main()