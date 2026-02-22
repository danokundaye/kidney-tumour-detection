# Phase 6, Step 6.2 - U-Net Crop Preparation (256 x 256)
#
#   This script uses the YOLO-predicted bounding boxes to crop kidney regions
#   from the segmentation training dataset and resizes them to 256x256 for U-Net
#   training.
#
# Steps:
#   - For 119 identified cases: use predicted bounding boxes from JSON files
#   - For missed case (00152): generate boxes from ground truth mask (fallback for training)
#   - Binary masks: tumour (170px) + cysts (255px) = 1, everything else = 0
#   - Track region_types (tumour only, cyst only, both) for EfficientNet training
# 
# Execution: Google Colab
#
# Mask values:
#   0 = background, 85 = kidney, 170 = tumour, 255 = cyst
#
# Output Structure:
#   unet_crops/
#       case_00000/
#           images/slice_0045.png           ← cropped + resized to 256x256
#           masks/slice_0045.png            ← binary mask, same crop + resize
#           region_types/slice_0045.txt     ← tumour_only/cyst_only/both

import os
import shutil
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Load config.yaml
def load_config(config_path: str) -> dict:
    """
    Load the config.yaml file
    All paths and settings come from here
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Generate ground-truth bounding box (case_00152)
def gt_bbox(
        mask_array  : np.ndarray,
        margin      : float = 0.20,     
) -> list:
    """
    Generate a bounding box from ground-truth mask for fallback scenarios.
    Finds non-background pixels and draws a tight box around them, and applies
    a 20% margin.

    Args:
        mask_array : 2D numpy array of mask values
        margin     : fractional expansion (0.20 = 20%)

    Returns:
        [x1, y1, x2, y2] in pixel space, or None if no kidney found
    """
    # Find all kidney/tumour/cyst pixels 
    rows, cols = np.where(mask_array > 0)

    if len(rows) == 0:
        return None

    img_h, img_w = mask_array.shape

    y1 = int(np.min(rows))
    y2 = int(np.max(rows))
    x1 = int(np.min(cols))
    x2 = int(np.max(cols))

    # Apply 20% margin expansion
    box_h = y2 - y1
    box_w = x2 - x1

    pad_y = int(box_h * margin)
    pad_x = int(box_w * margin)

    # Clip expanded box
    y1 = max(0,     y1 - pad_y)
    y2 = min(img_h, y2 + pad_y)
    x1 = max(0,     x1 - pad_x)
    x2 = min(img_w, x2 + pad_x)

    return [x1, y1, x2, y2]

# Binary mask creation
def create_binary_mask(mask_array: np.ndarray) -> tuple:
    """
    Convert KiTS21 mask to binary mask and determine region type.

    KiTS21 values: 0 = background, 85 = kidney, 170 = tumour, 255 = cyst

    Binary mask: tumour(170) + cyst(255) = 1, everything else = 0

    Returns:
        binary_mask : 2D numpy array (0 or 1)
        region_type : string (tumour_only/cyst_only/both/none)
    """
    has_tumour = np.any(mask_array == 170)
    has_cyst   = np.any(mask_array == 255)

    # Determine region type
    if has_tumour and has_cyst:
        region_type = "both"
    elif has_tumour:
        region_type = "tumour_only"
    elif has_cyst:
        region_type = "cyst_only"
    else:
        region_type = "none"

    # Create binary mask
    binary_mask = np.zeros_like(mask_array, dtype = np.uint8)
    binary_mask[mask_array == 170] = 1      # tumour = 1
    binary_mask[mask_array == 255] = 1      # cyst   = 1

    return binary_mask, region_type

# Single case processing
def process_case(
        case_id     : str,
        slices_dir   : Path,
        boxes_dir   : Path,
        output_dir  : Path,
        unet_size   : int,
        margin      : float,
        is_fallback : bool = False
) -> dict:
    """
    Process all slices for one case.

    For normal cases: reads bounding boxes from JSON file
    For fallback case (case_00152): generates boxes from ground truth masks

    Returns dict with processing statistics.
    """
    images_dir = slices_dir / "segmentation_train" / case_id / "images"
    masks_dir  = slices_dir / "segmentation_train" / case_id / "masks"

    # Output directories for a case
    output_images_dir = output_dir / case_id / "images"
    output_masks_dir = output_dir / case_id / "masks"
    output_regions_dir = output_dir / case_id / "region_types"

    output_images_dir.mkdir(parents = True,  exist_ok = True)
    output_masks_dir.mkdir(parents = True,   exist_ok = True)
    output_regions_dir.mkdir(parents = True, exist_ok = True)

    # Load bounding boxes from JSON for regular cases
    boxes = {}
    if not is_fallback:
        json_path = boxes_dir / f"{case_id}.json"
        if json_path.exists():
            boxes = json.loads(json_path.read_text())
    
     # Get all slice paths
    slice_paths = sorted(images_dir.glob("*.png"))

    stats = {
        'case_id'         : case_id,
        'total_slices'    : len(slice_paths),
        'processed_slices': 0,
        'skipped_slices'  : 0,
        'tumour_only'     : 0,
        'cyst_only'       : 0,
        'both'            : 0,
        'none'            : 0,
        'fallback'        : is_fallback
    }

    for slice_path in slice_paths:
        slice_name = slice_path.name

        # Load image and mask
        image = np.array(Image.open(slice_path).convert('L'))  # grayscale
        mask_path = masks_dir / slice_name

        if not mask_path.exists():
            stats['skipped_slices'] += 1
            continue

        mask = np.array(Image.open(mask_path))

        # Get bounding box for fallback case
        if is_fallback:
            box = gt_bbox(mask, margin)
            if box is None:
                stats['skipped_slices'] += 1
                continue
        else:
            # Use YOLO predicted JSON info
            if slice_name not in boxes:
                # No detection for this slice
                stats['skipped_slices'] += 1
                continue
            box = boxes[slice_name]
        
        x1, y1, x2, y2 = box

        # Crop image and mask using identical coordinates
        cropped_image = image[y1:y2, x1:x2]
        cropped_mask  = mask[y1:y2, x1:x2]

        # Skip if crop is empty
        if cropped_image.size == 0 or cropped_mask.size == 0:
            stats['skipped_slices'] += 1
            continue
    
        # Resize to 256x256
        cropped_image_resized = np.array(Image.fromarray(cropped_image).resize(
            (unet_size, unet_size), Image.BILINEAR))
        
        cropped_mask_resized = np.array(Image.fromarray(cropped_mask).resize(
            (unet_size, unet_size), Image.NEAREST))
        
        # Create binary mask and determine region type
        binary_mask, region_type = create_binary_mask(cropped_mask_resized)

        # Save outputs
        out_image_path  = output_images_dir  /   slice_name
        out_mask_path   = output_masks_dir   /   slice_name
        out_region_path = output_regions_dir /   slice_name.replace(".png", ".txt")

        # Save image as grayscale
        Image.fromarray(cropped_image_resized).save(out_image_path)

        # Save binary mask
        # Multiply by 255 for PNG visibility
        # Store as 0/255 PNG but loaded as 0/1 during training
        Image.fromarray((binary_mask * 255).astype(np.uint8)).save(out_mask_path)

        # Save region type as .txt
        out_region_path.write_text(region_type)

        # Update stats
        stats['processed_slices'] += 1
        stats[region_type] += 1
    
    return stats

# Processing loop
def prepare_crop(
        splits_dir  : Path,
        slices_dir  : Path,
        boxes_dir   : Path,
        output_dir  : Path,
        unet_size   : int,
        margin      : float):
    """
    Process all 120 segmentation_train cases.
    Writes to local storage first, syncs each case to Drive after completion.
    """
    FALLBACK_CASE = "case_00152"

    # Local storage for fast writes
    local_slices_dir = Path("/content/local_data/slices")
    local_output_dir = Path("/content/local_data/unet_crops")
    local_slices_dir.mkdir(parents = True, exist_ok = True)
    local_output_dir.mkdir(parents = True, exist_ok = True)

    seg_csv  = splits_dir / "segmentation_train.csv"
    cases_df = pd.read_csv(seg_csv)
    case_ids = cases_df['case_id'].tolist()

    print(f"Total cases      : {len(case_ids)}")
    print(f"Fallback case    : {FALLBACK_CASE}")
    print(f"Local output     : {local_output_dir}")
    print(f"Drive output     : {output_dir}")
    print(f"U-Net input size : {unet_size}x{unet_size}")
    print()

    output_dir.mkdir(parents = True, exist_ok = True)

    # Copy segmentation_train slices to local storage to make processing faster
    local_seg_dir = local_slices_dir / "segmentation_train"

    if not local_seg_dir.exists():
        print(f"\nCopying segmentation_train to local storage (~862MB)...")
        src = str(slices_dir / "segmentation_train") + "/"
        dst = str(local_seg_dir) + "/"
        ret = os.system(f"rsync -a --info=progress2 '{src}' '{dst}'")
        if ret != 0:
            raise RuntimeError("rsync failed. Check Drive mount.")
        print("Source data copied to local storage.")
    else:
        print(f"\nLocal source data already exists, skipping copy.")


    # Process cases from local storage
    all_stats = []

    for case_id in tqdm(case_ids, desc="Preparing U-Net crops"):

        # Check if case has already been processed and synced
        drive_case_dir = output_dir / case_id
        if drive_case_dir.exists() and any(drive_case_dir.rglob("*.png")):
            existing = len(list((drive_case_dir / "images").glob("*.png")))
            all_stats.append({
                'case_id'         : case_id,
                'total_slices'    : existing,
                'processed_slices': existing,
                'skipped_slices'  : 0,
                'tumour_only'     : 0,
                'cyst_only'       : 0,
                'both'            : 0,
                'none'            : 0,
                'fallback'        : case_id == FALLBACK_CASE
            })
            continue

        is_fallback = (case_id == FALLBACK_CASE)

        # Process from local slices to local output first
        stats = process_case(
            case_id     = case_id,
            slices_dir  = local_slices_dir,
            boxes_dir   = boxes_dir,
            output_dir  = local_output_dir, 
            unet_size   = unet_size,
            margin      = margin,
            is_fallback = is_fallback
        )

        all_stats.append(stats)

    # Sync all completed crops from local to Drive immediately
    print(f"\nSyncing crops to Drive...")
    src = str(local_output_dir) + "/"
    dst = str(output_dir) + "/"
    ret = os.system(f"rsync -a --info=progress2 '{src}' '{dst}'")
    if ret != 0:
        print("WARNING: rsync to Drive failed. Crops are still in local storage.")
    else:
        print("Sync complete.")


    # Print summary
    total_processed = sum(s['processed_slices'] for s in all_stats)
    total_skipped   = sum(s['skipped_slices']   for s in all_stats)
    total_tumour    = sum(s['tumour_only']       for s in all_stats)
    total_cyst      = sum(s['cyst_only']         for s in all_stats)
    total_both      = sum(s['both']              for s in all_stats)
    total_none      = sum(s['none']              for s in all_stats)

    print(f"\n U-Net Preparation Complete")
    print(f"  Cases processed    : {len(all_stats)}")
    print(f"  Total crops saved  : {total_processed}")
    print(f"  Skipped slices     : {total_skipped}")
    print(f"\n  Region type breakdown:")
    print(f"    tumour_only      : {total_tumour}")
    print(f"    cyst_only        : {total_cyst}")
    print(f"    both             : {total_both}")
    print(f"    none (healthy)   : {total_none}")
    print(f"\n  Crops saved to     : {output_dir}")

# Main
def main():
    config_path = "/content/kidney-tumour-detection/configs/config.yaml"
    config      = load_config(config_path)

    slices_dir  = Path(config['paths']['slices_dir'])
    splits_dir  = Path(config['paths']['splits_dir'])
    boxes_dir   = Path(config['paths']['processed_root']) / "unet_boxes"
    output_dir  = Path(config['paths']['unet_crops_dir'])
    unet_size   = config['preprocessing']['unet_input_size']
    margin      = config['preprocessing']['bbox_margin']

    print(f"\n U-Net Crop Preparation")

    prepare_crop(
        splits_dir = splits_dir,
        slices_dir = slices_dir,
        boxes_dir  = boxes_dir,
        output_dir = output_dir,
        unet_size  = unet_size,
        margin     = margin
    )

if __name__ == "__main__":
    main()