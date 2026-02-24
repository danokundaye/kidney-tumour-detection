# Phase 9 - End-to-End Inference Pipeline
#
#   Chains YOLO, U-Net, and EfficientNet on all 70 test cases.
#   For each case:
#     1. YOLO detects kidney region per slice at conf=0.10
#     2. U-Net segments tumour boundaries on 256x256 crop
#     3. EfficientNet classifies extracted patches at 224x224
#     4. Predicted 3D masks are saved as .nii.gz, patches as png, metrics to csv
#
# Patch extraction tiers:
#   contour         mask >= 100 pixels, bounding box from contour + 10% margin
#   full_crop_small mask < 100 pixels but > 0, full 256x256 crop used
#   full_crop_empty empty mask, full 256x256 crop used
# 
# Execution: google colab
#
# Output Structure:
#   results/phase9_pipeline/
#   ├── masks/
#   │   └── case_xxxxx_pred_mask.nii.gz
#   ├── patches/
#   │   └── case_xxxxx/
#   │       └── slice_xxxx.png
#   └── predictions.csv
#       columns: case_id, n_slices, n_detected, detection_rate, dice_3d,
#                iou_3d, n_patches, mean_prob, pred_label, confidence_flag,
#                patches_contour, patches_small, patches_empty, patches_no_det,
#                processing_time_s

import os
import sys
import time
import csv
import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
import cv2
import yaml

from pathlib import Path
from datetime import datetime
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO
import segmentation_models_pytorch as smp
import pandas as pd


def load_config(config_path: str) -> dict:
    """
    Load the config.yaml file.
    All paths and settings come from here.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_yolo(path: str):
    """
    Load trained yolov8 model.
    """
    print(f"  Loading YOLO from: {path}...")
    model = YOLO(path)
    print(f"  YOLOv8s loaded")
    return model


def load_unet(path: str, device: torch.device) -> nn.Module:
    """
    Rebuild unet with resnet50 encoder exactly as trained in phase 6,
    then load saved weights.
    """
    print(f"  Loading U-Net (ResNet50) from: {path}...")
    model = smp.Unet(
        encoder_name    = "resnet50",
        encoder_weights = None,
        in_channels     = 3,
        classes         = 1,
    )
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    print(f"  U-Net loaded")
    return model


def load_efficientnet(path: str, device: torch.device) -> nn.Module:
    """
    Rebuild EfficientNet-B0 with same classifier head as phase 7,
    then load saved weights.
    """
    print(f"  Loading efficientnet from: {path}...")
    model = models.efficientnet_b0(weights = None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p = 0.2),
        nn.Linear(in_features, 1)
    )
    checkpoint = torch.load(path, map_location = device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    print(f"  EfficientNet loaded")
    return model


def expand_box(x1, y1, x2, y2, margin, img_w, img_h):
    """
    Expand a bounding box by a fractional margin on all sides,
    Clamped to image dimensions.
    """
    bw   = x2 - x1
    bh   = y2 - y1
    x1   = max(0,     int(x1 - bw * margin))
    y1   = max(0,     int(y1 - bh * margin))
    x2   = min(img_w, int(x2 + bw * margin))
    y2   = min(img_h, int(y2 + bh * margin))
    return x1, y1, x2, y2


def get_nonzero_bbox(mask: np.ndarray, margin: float):
    """
    Given a binary mask (h x w), find the bounding box of non zero pixels
    using a row and column scan (np.any). 
    Expand by margin and return (x1, y1, x2, y2) clamped to mask dims. 
    Returns none if mask is empty.
    """
    rows = np.any(mask > 0, axis = 1)
    cols = np.any(mask > 0, axis = 0)
    if not rows.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    H, W   = mask.shape
    return expand_box(x1, y1, x2 + 1, y2 + 1, margin, W, H)


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute dice coefficient between two binary arrays.
    Returns 1.0 if both are empty (no tumour in gt or prediction).
    """
    pred         = (pred > 0).astype(np.float32)
    gt           = (gt   > 0).astype(np.float32)
    intersection = (pred * gt).sum()
    denominator  = pred.sum() + gt.sum()
    if denominator == 0:
        return 1.0
    return float(2.0 * intersection / denominator)


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute intersection over union between two binary arrays.
    Returns 1.0 if both are empty.
    """
    pred         = (pred > 0).astype(np.float32)
    gt           = (gt   > 0).astype(np.float32)
    intersection = (pred * gt).sum()
    union        = ((pred + gt) > 0).astype(np.float32).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def extract_tumour_binary(mask: np.ndarray) -> np.ndarray:
    """
    Convert kits21 mask encoding to binary tumour mask.
    values: 0 = background, 85 = kidney, 170 = tumour, 255 = cyst.
    tumour and cyst both treated as abnormal.
    """
    return ((mask == 170) | (mask == 255)).astype(np.uint8)

# Note: the 100 pixel threshold is appliedat inference time and not in patch preparation
def process_slice(
    slice_path      : str,
    yolo_model,
    unet_model      : nn.Module,
    effnet_model    : nn.Module,
    unet_transform,
    effnet_transform,
    device          : torch.device,
    yolo_conf       : float,
    yolo_margin     : float,
    unet_size       : int,
    unet_threshold  : float,
    min_mask_pixels : int,
    patch_margin    : float,
) -> dict:
    """
    Run full pipeline on a single CT slice.

    returns a dict with:
      pred_mask_256  np array (256 x 256) binary unet prediction
      patch_crop     pil image (224 x 224) efficientnet input
      patch_method   str, one of bbox_nonzero / full_crop_small / full_crop_empty / no_detection
      effnet_prob    float or none, malignant probability
      detected       bool, whether yolo found a kidney
    """
    result = {
        'pred_mask_256' : np.zeros((unet_size, unet_size), dtype=np.uint8),
        'patch_crop'    : None,
        'patch_method'  : 'no_detection',
        'effnet_prob'   : None,
        'detected'      : False,
    }

    # load slice
    img_bgr = cv2.imread(slice_path)
    if img_bgr is None:
        return result
    img_h, img_w = img_bgr.shape[:2]
    img_pil      = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    # yolo detection
    yolo_results = yolo_model.predict(slice_path, conf=yolo_conf, verbose=False)
    boxes        = yolo_results[0].boxes

    if len(boxes) == 0:
        # no kidney detected, use full image as fallback crop for unet
        crop_pil = img_pil.resize((unet_size, unet_size), Image.BILINEAR)
    else:
        # take highest confidence box and expand by yolo margin
        best_idx        = boxes.conf.argmax().item()
        x1, y1, x2, y2  = boxes.xyxy[best_idx].tolist()
        x1, y1, x2, y2  = expand_box(x1, y1, x2, y2, yolo_margin, img_w, img_h)
        crop_pil        = img_pil.crop((x1, y1, x2, y2))
        crop_pil        = crop_pil.resize((unet_size, unet_size), Image.BILINEAR)
        result['detected'] = True

    # unet segmentation
    # crop_pil is now 256 x 256 regardless of detection outcome
    unet_input = unet_transform(crop_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        unet_output = unet_model(unet_input)
        prob_map    = torch.sigmoid(unet_output).squeeze().cpu().numpy()

    pred_mask = (prob_map >= unet_threshold).astype(np.uint8)
    result['pred_mask_256'] = pred_mask

    # patch extraction using tiered approach
    n_pixels = int(pred_mask.sum())

    if n_pixels >= min_mask_pixels:
        # tier 1 - non-zero based bounding box from predicted mask
        box = get_nonzero_bbox(pred_mask, patch_margin)
        if box is not None:
            px1, py1, px2, py2 = box
            crop_arr  = np.array(crop_pil)
            patch_arr = crop_arr[py1:py2, px1:px2]
            result['patch_method'] = 'bbox_nonzero'
        else:
            # bbox failed despite enough pixels, fall back to full crop
            patch_arr = np.array(crop_pil)
            result['patch_method'] = 'full_crop_small'
    elif n_pixels > 0:
        # tier 2 - small mask, use full 256 x 256 crop
        patch_arr = np.array(crop_pil)
        result['patch_method'] = 'full_crop_small'
    else:
        # tier 3 - empty mask, full 256x256 kidney crop passed as fallback
        # an empty unet prediction means no abnormal region was found, so this
        # crop is outside efficientnet's training distribution. predictions on
        # these slices are expected to be unreliable and are flagged accordingly.
        patch_arr = np.array(crop_pil)
        result['patch_method'] = 'full_crop_empty'

    # convert patch array to pil for efficientnet transform
    patch_pil = Image.fromarray(patch_arr.astype(np.uint8))
    result['patch_crop'] = patch_pil

    # efficientnet classification
    effnet_input = effnet_transform(patch_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logit           = effnet_model(effnet_input)
        prob_malignant  = torch.sigmoid(logit).item()

    result['effnet_prob'] = prob_malignant

    return result


def process_case(
    case_id         : str,
    test_slices_dir : Path,
    raw_dir         : Path,
    output_masks_dir: Path,
    output_patches_dir: Path,
    yolo_model,
    unet_model      : nn.Module,
    effnet_model    : nn.Module,
    unet_transform,
    effnet_transform,
    device          : torch.device,
    config          : dict,
) -> dict:
    """
    Run the full pipeline on one test case.

    Load all slices, runs per slice inference, stacks predicted masks
    into a 3d volume, saves as .nii.gz, saves patches as png files,
    and returns a summary dict for the predictions csv.
    """
    case_start = time.time()

    # settings from config
    yolo_conf        = 0.10   # intentionally lower than training conf of 0.5
                               # to maximise kidney detection recall
    yolo_margin      = config['preprocessing']['bbox_margin']
    unet_size        = config['preprocessing']['unet_input_size']
    unet_threshold   = config['unet']['segmentation_threshold']
    effnet_threshold = config['efficientnet']['classification_threshold']
    min_mask_pixels  = config['preprocessing']['min_tumor_area']
    patch_margin     = config['classification']['box_margin']

    images_dir = test_slices_dir / case_id / "images"
    masks_dir  = test_slices_dir / case_id / "masks"

    slice_paths = sorted(images_dir.glob("*.png"))
    n_slices    = len(slice_paths)

    if n_slices == 0:
        print(f"  warning: no slices found for {case_id}")
        return None

    # arrays to accumulate per slice results
    pred_mask_stack = []   # list of 256 x 256 binary arrays
    gt_mask_stack   = []   # list of 256 x 256 binary arrays (for dice/iou)
    effnet_probs    = []   # list of floats
    patch_methods   = []   # list of strings
    n_detected      = 0    # slices where yolo found a kidney

    # output patch folder for this case
    case_patch_dir = output_patches_dir / case_id
    case_patch_dir.mkdir(parents = True, exist_ok = True)

    for slice_path in slice_paths:
        slice_name = slice_path.stem

        # run pipeline on this slice
        result = process_slice(
            slice_path       = str(slice_path),
            yolo_model       = yolo_model,
            unet_model       = unet_model,
            effnet_model     = effnet_model,
            unet_transform   = unet_transform,
            effnet_transform = effnet_transform,
            device           = device,
            yolo_conf        = yolo_conf,
            yolo_margin      = yolo_margin,
            unet_size        = unet_size,
            unet_threshold   = unet_threshold,
            min_mask_pixels  = min_mask_pixels,
            patch_margin     = patch_margin,
        )

        pred_mask_stack.append(result['pred_mask_256'])
        patch_methods.append(result['patch_method'])

        if result['detected']:
            n_detected += 1

        if result['effnet_prob'] is not None:
            effnet_probs.append(result['effnet_prob'])

        # save patch png if one was extracted
        if result['patch_crop'] is not None:
            patch_save_path = case_patch_dir / f"{slice_name}.png"
            result['patch_crop'].save(str(patch_save_path))

        # load corresponding gt mask and convert to binary tumour mask
        gt_mask_path = masks_dir / f"{slice_name}.png"
        if gt_mask_path.exists():
            gt_raw  = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
            gt_bin  = extract_tumour_binary(gt_raw)
            # resize gt to 256 x 256 to match predicted mask dimensions
            gt_bin  = cv2.resize(gt_bin, (unet_size, unet_size),
                                 interpolation = cv2.INTER_NEAREST)
            gt_mask_stack.append(gt_bin)
        else:
            gt_mask_stack.append(np.zeros((unet_size, unet_size), dtype=np.uint8))

    # stack 2d masks into 3d volume (n_slices x 256 x 256)
    pred_volume = np.stack(pred_mask_stack, axis = 0).astype(np.uint8)
    gt_volume   = np.stack(gt_mask_stack,   axis = 0).astype(np.uint8)

    # save predicted mask as .nii.gz using identity affine
    # identity affine means no physical space information
    affine      = np.eye(4)
    nifti_img   = nib.Nifti1Image(pred_volume, affine)
    mask_save_path = output_masks_dir / f"{case_id}_pred_mask.nii.gz"
    nib.save(nifti_img, str(mask_save_path))

    # compute segmentation metrics across 3d volumes
    dice_3d = compute_dice(pred_volume, gt_volume)
    iou_3d  = compute_iou(pred_volume,  gt_volume)

    # aggregate efficientnet predictions at case level
    if len(effnet_probs) > 0:
        mean_prob       = float(np.mean(effnet_probs))
        pred_label      = 'malignant' if mean_prob >= effnet_threshold else 'benign'

        # flag low confidence if majority of patches came from fallback methods
        fallback_count  = sum(1 for m in patch_methods
                              if m in ('full_crop_small', 'full_crop_empty'))
        confidence_flag = 'low_confidence' if fallback_count > len(patch_methods) / 2 \
                          else 'standard'
    else:
        mean_prob       = None
        pred_label      = 'no_prediction'
        confidence_flag = 'no_patches'

    processing_time = round(time.time() - case_start, 2)

    # count patch methods used
    method_counts = {
        'bbox_nonzero'         : patch_methods.count('bbox_nonzero'),
        'full_crop_small' : patch_methods.count('full_crop_small'),
        'full_crop_empty' : patch_methods.count('full_crop_empty'),
        'no_detection'    : patch_methods.count('no_detection'),
    }

    return {
        'case_id'          : case_id,
        'n_slices'         : n_slices,
        'n_detected'       : n_detected,
        'detection_rate'   : round(n_detected / n_slices, 4) if n_slices > 0 else 0.0,
        'dice_3d'          : round(dice_3d, 4),
        'iou_3d'           : round(iou_3d,  4),
        'n_patches'        : len(effnet_probs),
        'mean_prob'        : round(mean_prob, 4) if mean_prob is not None else None,
        'pred_label'       : pred_label,
        'confidence_flag'  : confidence_flag,
        'patches_contour'  : method_counts['bbox_nonzero'],
        'patches_small'    : method_counts['full_crop_small'],
        'patches_empty'    : method_counts['full_crop_empty'],
        'patches_no_det'   : method_counts['no_detection'],
        'processing_time_s': processing_time,
    }


def main():
    config_path = "/content/kidney-tumour-detection/configs/config.yaml"
    config      = load_config(config_path)

    print("Phase 9 - End-to-End pipeline inference")
    print(f"started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\ndevice: {device}")
    if device.type == "cuda":
        print(f"gpu   : {torch.cuda.get_device_name(0)}")

    # paths from config
    results_dir   = Path(config['paths']['results_dir'])
    splits_dir    = Path(config['paths']['splits_dir'])
    slices_dir    = Path(config['paths']['slices_dir'])

    # model paths constructed from results_dir
    yolo_path    = results_dir / "phase5_yolo_retrain/yolov8s_retrain_run1/weights/best.pt"
    unet_path    = results_dir / "phase6_unet/weights/best.pt"
    effnet_path  = results_dir / "phase7_efficientnet/weights/best.pt"

    # output directories
    output_dir         = results_dir / "phase9_pipeline"
    output_masks_dir   = output_dir  / "masks"
    output_patches_dir = output_dir  / "patches"
    output_masks_dir.mkdir(parents=True,   exist_ok=True)
    output_patches_dir.mkdir(parents=True, exist_ok=True)

    # test slices and case list
    test_slices_dir = slices_dir / "test"
    test_df         = pd.read_csv(splits_dir / "test.csv")
    test_cases      = test_df['case_id'].tolist()

    print(f"\nTest cases loaded : {len(test_cases)}")
    print(f"Output directory  : {output_dir}")

    # verify all model paths exist before loading
    print("\nVerifying model paths...")
    for name, path in [("yolo", yolo_path), ("unet", unet_path), ("efficientnet", effnet_path)]:
        if not path.exists():
            print(f"  model not found: {path}")
            sys.exit(1)
        print(f"  {name} ok: {path}")

    # load models
    print("\nLoading models...")
    yolo_model   = load_yolo(str(yolo_path))
    unet_model   = load_unet(str(unet_path),   device)
    effnet_model = load_efficientnet(str(effnet_path), device)

    # image transforms (must match phase 6 and phase 7 training exactly)
    unet_transform = transforms.Compose([
        transforms.Resize((config['preprocessing']['unet_input_size'],
                           config['preprocessing']['unet_input_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])

    effnet_transform = transforms.Compose([
        transforms.Resize((config['preprocessing']['efficientnet_input_size'],
                           config['preprocessing']['efficientnet_input_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])

    # run pipeline on all test cases
    print(f"\nRunning pipeline on {len(test_cases)} test cases\n")

    all_results  = []
    pipeline_start = time.time()

    for i, case_id in enumerate(test_cases):
        print(f"[{i+1:02d} / {len(test_cases)}] {case_id}", end=" ... ")
        sys.stdout.flush()

        result = process_case(
            case_id            = case_id,
            test_slices_dir    = test_slices_dir,
            raw_dir            = Path(config['paths']['dataset_root']),
            output_masks_dir   = output_masks_dir,
            output_patches_dir = output_patches_dir,
            yolo_model         = yolo_model,
            unet_model         = unet_model,
            effnet_model       = effnet_model,
            unet_transform     = unet_transform,
            effnet_transform   = effnet_transform,
            device             = device,
            config             = config,
        )

        if result is not None:
            all_results.append(result)
            print(f"Dice = {result['dice_3d']:.4f}  "
                  f"IoU  = {result['iou_3d']:.4f}  "
                  f"Pred = {result['pred_label']}  "
                  f"Time = {result['processing_time_s']}s")
        else:
            print("Skipped (no slices found)")

    # save predictions csv
    pred_csv_path = output_dir / "predictions.csv"
    df_results    = pd.DataFrame(all_results)
    df_results.to_csv(str(pred_csv_path), index=False)

    # print summary
    total_time = round(time.time() - pipeline_start, 2)

    print("\n Phase 9 complete")
    print(f"\nCases processed    : {len(all_results)}")
    print(f"Mean Dice (3D)     : {df_results['dice_3d'].mean():.4f}")
    print(f"Mean IoU (3D)      : {df_results['iou_3d'].mean():.4f}")
    print(f"Mean Processing    : {df_results['processing_time_s'].mean():.1f}s per case")
    print(f"Total Runtime      : {total_time:.1f}s")

    print(f"\nPredictions csv    : {pred_csv_path}")
    print(f"Predicted masks    : {output_masks_dir}")
    print(f"Patches            : {output_patches_dir}")
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()