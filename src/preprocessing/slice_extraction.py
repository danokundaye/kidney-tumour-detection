# Phase 4, Step 4.3 - CT Slice Extraction
# This script converts 3D NIfTI CT volumes into 2D PNG slices for model training. Each split is processed differently based on what the models actually need:
# Detection training (110 cases) → ALL slices
#     - YOLO needs both positive (kidney present) and negative (non-kidney) examples to learn to say "nothing here".
# Segmentation training (120 cases) → KIDNEY-CONTAINING slices only
#     - U-Net operates on pre-cropped kidney regions fed by YOLO output.
#     - No point saving slices with no kidney anatomy.
#
# Test (70 cases) → ALL slices
#     - Test must simulate real deployment where no masks exist.
#
#   CT Windowing:
#     Raw HU values range from -2048 to 3071. We clip to [-79, 304] HU (standard abdominal window) then normalize to [0, 255] for PNG.
#
#   Checkpoint/Resume:
#     If Colab disconnects mid-run, the script skips already-processed cases on restart. Do not delete the output folders mid-run.
#
# Execution: Google Colab (data lives on Drive)
#
# OUTPUT STRUCTURE:
#   processed/slices/
#   ├── detection_train/
#   │   └── case_00000/
#   │       ├── images/slice_0000.png ... slice_NNNN.png
#   │       └── masks/ slice_0000.png ... slice_NNNN.png
#   ├── segmentation_train/
#   │   └── case_00001/
#   │       ├── images/slice_XXXX.png
#   │       └── masks/ slice_XXXX.png
#   └── test/
#       └── case_00002/
#           ├── images/slice_0000.png ... slice_NNNN.png
#           └── masks/ slice_0000.png ... slice_NNNN.png

import yaml
import numpy as np
import nibabel as nib
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Load config.yaml
def load_config(config_path: str) -> dict:
    """
    Load the config.yaml file
    All paths and settings come from here
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# CT Windowing and Normalisation
def apply_window_normalise(
        slice_data: np.ndarray,
        window_min: float,
        window_max: float
) -> np.ndarray:
    """
    Apply CT windowing and normalize to [0, 255] for PNG saving.

    Steps:
    1. Clip HU values to [window_min, window_max]
       - Everything below window_min becomes window_min (e.g. air → -79)
       - Everything above window_max becomes window_max (e.g. bone → 304)
    2. Normalize clipped values to [0.0, 1.0]
    3. Scale to [0, 255] and convert to uint8 for PNG format

    Args:
        slice_data  : 2D numpy array of HU values
        window_min  : Lower HU bound (from config, -79)
        window_max  : Upper HU bound (from config, 304)

    Returns:
        2D numpy array of dtype uint8, values in [0, 255]
    """
    # Clip image to window range
    clipped = np.clip(slice_data, window_min, window_max)

    # Normalise to [0.0, 1.0]
    normalised = (clipped - window_min) / (window_max - window_min)

    # Scale to [0, 255] and convert to uint8
    return (normalised * 255).astype(np.uint8)

# Single Case Processing
def process_case(
        case_id: str,
        dataset_root: Path,
        output_dir: Path,
        window_min: float,
        window_max: float,
        kidney_only: bool
) -> dict:
    """
    Process one case: extract slices and save as PNG.

    Args:
        case_id      : e.g. 'case_00000'
        dataset_root : Path to raw KiTS21 data
        output_dir   : Where to save this case's slices
        window_min   : CT window lower bound
        window_max   : CT window upper bound
        kidney_only  : If True, only save slices containing kidney/tumour/cyst

    Returns:
        Dict with processing statistics for this case
    """
    case_dir = dataset_root / case_id
    img_path = case_dir / "imaging.nii.gz"
    seg_path = case_dir / "aggregated_MAJ_seg.nii.gz"

    # Create output subdirectories
    images_dir = output_dir / case_id / "images"
    masks_dir  = output_dir / case_id / "masks"
    images_dir.mkdir(parents = True, exist_ok = True)
    masks_dir.mkdir(parents = True, exist_ok = True)

    # Load volumes
    img_data = nib.load(str(img_path)).get_fdata()
    seg_data = np.round(nib.load(str(seg_path)).get_fdata()).astype(np.uint8)

    total_slices   = img_data.shape[2]
    saved_slices   = 0
    skipped_slices = 0

    for slice_idx in range(total_slices):
        img_slice = img_data[:, :, slice_idx]
        seg_slice = seg_data[:, :, slice_idx]

        # For segmentation: skip slices with no kidney anatomy
        if kidney_only:
            has_kidney = np.any(seg_slice > 0)  # any label > 0 means organ present
            if not has_kidney:
                skipped_slices += 1
                continue
        
        # Apply windowing and normalise image slice
        img_processed = apply_window_normalise(
            img_slice, window_min, window_max
        )

        # Save image slice as PNG
        slice_name = f"slice_{slice_idx:04d}.png"
        Image.fromarray(img_processed).save(images_dir / slice_name)

        # Save mask slice as PNG
        # Mask values: 0=background, 1=kidney, 2=tumor, 3=cyst
        # Scale by 85 so labels are visible: 0→0, 1→85, 2→170, 3→255
        mask_visible = (seg_slice * 85).astype(np.uint8)
        Image.fromarray(mask_visible).save(masks_dir / slice_name)

        saved_slices +=1
    
    # Free memory
    del img_data, seg_data

    return {
        'case_id'       : case_id,
        'total_slices'  : total_slices,
        'saved_slices'  : saved_slices,
        'skipped_slices': skipped_slices
    }

# Process a Split
def process_split(
        split_name: str,
        split_csv: str,
        dataset_root: Path,
        slices_dir: Path,
        window_min: float,
        window_max: float,
        kidney_only: bool
) -> None:
    """
    Process all cases in one split (detection, segmentation, or test).

    Checkpoint/resume logic:
        A case is considered complete if its images/ folder exists and
        contains at least one PNG. Completed cases are skipped on restart.

    Args:
        split_name  : 'detection_train', 'segmentation_train', or 'test'
        split_csv   : Path to the split CSV file
        dataset_root: Path to raw KiTS21 data
        slices_dir  : Root output directory for slices
        window_min  : CT window lower bound
        window_max  : CT window upper bound
        kidney_only : Whether to skip non-kidney slices
    """
    print(f"\nProcessing:       {split_name.capitalize()}")
    print(f"Kidney-only filter: {kidney_only}")

    # Load case list for split
    cases_df = pd.read_csv(split_csv)
    case_ids = cases_df['case_id'].tolist()
    output_dir = slices_dir / split_name

    # Checkpoint for already completed cases
    completed = set()
    for case_id in case_ids:
        images_dir = output_dir / case_id / "images"
        if images_dir.exists() and any(images_dir.glob("*.png")):
            completed.add(case_id)

    remaining = [c for c in case_ids if c not in completed]

    print(f"Total cases    : {len(case_ids)}")
    print(f"Already done   : {len(completed)}")
    print(f"To process     : {len(remaining)}")

    if not remaining:
        print("All cases already processed. Skipping.")
        return
    
    # Process remaining cases
    stats = []
    for case_id in tqdm(remaining, desc=f"Extracting {split_name}"):
        result = process_case(
            case_id      = case_id,
            dataset_root = dataset_root,
            output_dir   = output_dir,
            window_min   = window_min,
            window_max   = window_max,
            kidney_only  = kidney_only
        )
        stats.append(result)

    # Print summary for this split
    if stats:
        stats_df = pd.DataFrame(stats)
        print(f"\n{split_name} summary:")
        print(f"  Cases processed  : {len(stats_df)}")
        print(f"  Total slices     : {stats_df['total_slices'].sum()}")
        print(f"  Saved slices     : {stats_df['saved_slices'].sum()}")
        print(f"  Skipped slices   : {stats_df['skipped_slices'].sum()}")
        print(f"  Avg saved/case   : {stats_df['saved_slices'].mean():.1f}")

# Main
def main():
    config_path = "/content/kidney-tumour-detection/configs/config.yaml"
    config = load_config(config_path)

    dataset_root = Path(config['paths']['dataset_root'])
    splits_dir   = Path(config['paths']['splits_dir'])
    slices_dir   = Path(config['paths']['slices_dir'])
    window_min   = config['preprocessing']['ct_window_min']
    window_max   = config['preprocessing']['ct_window_max']

    print("\nSlice Extraction")
    print(f"Dataset root : {dataset_root}")
    print(f"Output dir   : {slices_dir}")
    print(f"CT window    : [{window_min}, {window_max}] HU")

    # Process each split with the appropriate filters
    process_split(
        split_name   = "detection_train",
        split_csv    = str(splits_dir / "detection_train.csv"),
        dataset_root = dataset_root,
        slices_dir   = slices_dir,
        window_min   = window_min,
        window_max   = window_max,
        kidney_only  = False   # YOLO needs all slices including negatives
    )

    process_split(
        split_name   = "segmentation_train",
        split_csv    = str(splits_dir / "segmentation_train.csv"),
        dataset_root = dataset_root,
        slices_dir   = slices_dir,
        window_min   = window_min,
        window_max   = window_max,
        kidney_only  = True    # U-Net only needs kidney-containing slices
    )

    process_split(
        split_name   = "test",
        split_csv    = str(splits_dir / "test.csv"),
        dataset_root = dataset_root,
        slices_dir   = slices_dir,
        window_min   = window_min,
        window_max   = window_max,
        kidney_only  = False   # Test simulates real deployment, all slices
    )

    print("\n Slice extraction complete")
    print("Check your Drive for processed/slices/")
    print("Verify a few slices visually.")

if __name__ == "__main__":
    main()