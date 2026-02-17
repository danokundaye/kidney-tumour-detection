# Phase 4, Step 4.4 — YOLO Label Generation
#
# PURPOSE:
#   Generate YOLO-format bounding box label files from segmentation masks.
#   One .txt label file is created per image slice.
#
#   The bounding box covers the entire kidney region including any tumour
#   or cyst within it.
#
#   Label format (one line per kidney per slice):
#       class_id centre_x centre_y width height
#   All values normalized to [0.0, 1.0] relative to image dimensions.
#   class_id is always 0 (kidney).
#
#   Multiple kidneys per slice:
#       Connected components analysis separates left and right kidneys.
#       Each gets its own line in the label file.
#
#   Empty label files:
#       Slices with no kidney also get a .txt file, but it is empty.
#       YOLO requires every image to have a corresponding label file,
#       even if it is empty.
#
# Execution: Google Colab
#
# OUTPUT STRUCTURE:
#   processed/slice/
#   └── detection_train/
#       └── case_00000/
#           |- images
#           |- masks
#           |- labels
#              ├── slice_0000.txt  (empty — no kidney)
#              ├── slice_0139.txt  (one or two kidney boxes)
#              └── ...

import yaml
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage

# Load config.yaml
def load_config(config_path: str) -> dict:
    """
    Load the config.yaml file
    All paths and settings come from here
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Generate bounding boxes from masks
def mask_to_yolo(
        mask: np.ndarray,
        img_height: int,
        img_width: int,
        min_area: int = 100
) -> list:
    """
    Convert a segmentation mask to YOLO bounding box annotations.

    Steps:
    1. Create binary mask of all organ pixels (kidney + tumour + cyst)
       In our scaled masks: 85=kidney, 170=tumour, 255=cyst
       Any value > 0 means organ is present
    2. Use connected components to separate left and right kidneys
    3. For each component above min_area, compute bounding box
    4. Normalize coordinates to [0, 1]

    Args:
        mask       : 2D numpy array (scaled mask PNG, values 0/85/170/255)
        img_height : Image height in pixels (512)
        img_width  : Image width in pixels (512)
        min_area   : Minimum component size in pixels to avoid noise

    Returns:
        List of [class_id, cx, cy, w, h] for each kidney found.
        Empty list if no kidney present.
    """
    # Implement Binary Mask - Organ (1), Background (0)
    binary_mask = (mask > 0).astype(np.uint8)

    if binary_mask.sum() == 0:
        return []                   # No organ present
    
    # Check connected components and separate left/right kidneys
    # Each distinct group of organ pixels gets a unique label
    labelled_array, num_components = ndimage.label(binary_mask)

    boxes = []

    for component_id in range(1, num_components + 1):
        # Isolate component
        component = (labelled_array == component_id)

        # Skip tiny components which are likely annotation noise from CT protocols
        if component.sum() < min_area:
            continue
        
        # Compute bounding boxes from component pixel positions
        # np.where returns (row_indices, col_indices) of True pixels
        rows, cols = np.where(component)

        row_min, row_max = rows.min(), rows.max()
        col_min, col_max = cols.min(), cols.max()

        # Convert to YOLO format (centre_x, centre_y, width, height)
        cx = (col_min + col_max) / 2.0 / img_width
        cy = (row_min + row_max) / 2.0 / img_height
        w  = (col_max - col_min) / img_width
        h  = (row_max - row_min) / img_height

        # Clamp to [0, 1] to handle any edge cases
        cx = np.clip(cx, 0.0, 1.0)
        cy = np.clip(cy, 0.0, 1.0)
        w  = np.clip(w,  0.0, 1.0)
        h  = np.clip(h,  0.0, 1.0)

        boxes.append([0, cx, cy, w, h])  # class_id 0 = kidney

    return boxes

# Single case Pprocessing
def process_case(
        case_id: str,
        masks_dir: Path,
        labels_dir: Path,
        img_height: int,
        img_width: int,
        min_area: int
) -> dict:
    """
    Generate YOLO label files for all slices of one case.

    Args:
        case_id    : e.g. 'case_00000'
        masks_dir  : Path to this case's mask PNGs
        img_height : Image height (512)
        img_width  : Image width (512)
        min_area   : Minimum organ area to keep

    Returns:
        Dict with processing statistics for this case
    """
    labels_dir.mkdir(parents=True, exist_ok=True)
    mask_files = sorted(masks_dir.glob("*.png"))

    total_slices   = len(mask_files)
    positive_slices = 0   # slices with at least one kidney box
    total_boxes    = 0    # total kidney boxes generated


    for mask_file in mask_files:
        mask = np.array(Image.open(mask_file))

        # Generate bounding boxes from this mask
        boxes = mask_to_yolo(mask, img_height, img_width, min_area)

        # Save label file — same name as image but .txt extension
        label_file = labels_dir / mask_file.with_suffix('.txt').name

        if boxes:
            # Write one line per box
            with open(label_file, 'w') as f:
                for box in boxes:
                    # Format: class_id, cx cy w h (6 decimal places)
                    f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} "
                            f"{box[3]:.6f} {box[4]:.6f}\n")
            positive_slices += 1
            total_boxes     += len(boxes)
        else:
            # Empty file — negative example for YOLO
            label_file.touch()

    return {
        'case_id'        : case_id,
        'total_slices'   : total_slices,
        'positive_slices': positive_slices,
        'empty_slices'   : total_slices - positive_slices,
        'total_boxes'    : total_boxes,
        'avg_boxes'      : total_boxes / positive_slices if positive_slices > 0 else 0
    }

# Process all detection training cases
def generate_labels(
        split_csv: str,
        slices_dir: Path,
        img_height: int,
        img_width: int,
        min_area: int
) -> None:
    """
    Generate YOLO labels for all cases in the detection training split.

    Only detection_train needs YOLO labels — segmentation and test splits
    use the masks directly for U-Net training and final evaluation.

    Checkpoint:
        A case is considered complete if its label folder exists and
        contains at least one .txt file.
    """
    cases_df = pd.read_csv(split_csv)
    case_ids = cases_df['case_id'].tolist()

    # Checkpoint for already completed cases
    completed = set()
    for case_id in case_ids:
        case_labels_dir = slices_dir / "detection_train" / case_id / "labels"
        if case_labels_dir.exists() and any(case_labels_dir.glob("*.txt")):
            completed.add(case_id)

    remaining = [c for c in case_ids if c not in completed]

    print(f"Total cases    : {len(case_ids)}")
    print(f"Already done   : {len(completed)}")
    print(f"To process     : {len(remaining)}")

    if not remaining:
        print("All cases already processed. Skipping.")
        return
    
    stats = []

    for case_id in tqdm(remaining, desc="Generating YOLO labels"):
        masks_dir_case  = slices_dir / "detection_train" / case_id / "masks"
        labels_dir_case = slices_dir / "detection_train" / case_id / "labels"

        result = process_case(
            case_id    = case_id,
            masks_dir  = masks_dir_case,
            labels_dir = labels_dir_case,
            img_height = img_height,
            img_width  = img_width,
            min_area   = min_area
        )
        stats.append(result)

    # Print summary
    if stats:
        stats_df = pd.DataFrame(stats)
        print(f"\nYOLO Label Generation Summary:")
        print(f"  Cases processed   : {len(stats_df)}")
        print(f"  Total slices      : {stats_df['total_slices'].sum()}")
        print(f"  Positive slices   : {stats_df['positive_slices'].sum()}")
        print(f"  Empty slices      : {stats_df['empty_slices'].sum()}")
        print(f"  Total boxes       : {stats_df['total_boxes'].sum()}")
        print(f"  Avg boxes/positive: {stats_df['avg_boxes'].mean():.2f}")

# Main
def main():
    config_path = "/content/kidney-tumour-detection/configs/config.yaml"
    config      = load_config(config_path)

    slices_dir  = Path(config['paths']['slices_dir'])
    # Labels saved inside each case folder (slices_dir/case_id/labels/)
    # YOLO finds them automatically by replacing 'images' with 'labels' in path
    splits_dir  = Path(config['paths']['splits_dir'])
    img_height  = config['preprocessing']['slice_size']
    img_width   = config['preprocessing']['slice_size']
    min_area    = config['preprocessing']['min_tumor_area']

    # YOLO label generation
    print("YOLO label generation")
    print("=" * 50)
    print(f"Slices dir  : {slices_dir}")
    print(f"Image size  : {img_height} x {img_width}")
    print(f"Min area    : {min_area} pixels\n")

    generate_labels(
        split_csv  = str(splits_dir / "detection_train.csv"),
        slices_dir = slices_dir,
        img_height = img_height,
        img_width  = img_width,
        min_area   = min_area
    )

    print("Label generation complete")
    print("Verify a few label files manually:")
    print("  - Positive slices should have 1-2 lines")
    print("  - All values should be between 0 and 1")
    print("  - Empty slices should have 0-byte .txt files")


if __name__ == "__main__":
    main()
