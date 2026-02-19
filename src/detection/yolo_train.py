# YOLO retraining
# This aims to train YOLOv8s for kidney detection on the KiTS21 detection_train cases.
#
# Key lessons from Phase 5 initial runs applied here:
#   - NO large-slice filter
#     The original filter caused 27.1% detection rate on segmentation_train
#     because YOLO never learned to detect boundary/partial kidney slices.
#   - 1:1 positive to background ratio (reduces class imbalance)
#   - patience=0
#   - conf=0.10 at inference
#
# Training set composition (approximate):
#   - All 20,651 positive slices (kidney present)
#   - 20,651 background slices (randomly sampled, 1:1 ratio)
#   - Total: ~41,302 training images
#
# Output:
#   Results saved to:
#   .../results/phase5_yolo_retrain/
# =============================================================================

import os
import yaml
import random
import shutil
import pandas as pd
from pathlib import Path
from ultralytics import YOLO


# Load config.yaml
def load_config(config_path: str) -> dict:
    """
    Load the config.yaml file
    All paths and settings come from here
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Dataset Prep
def build_filtered_dataset(splits_dir: Path,
                            slices_dir: Path,
                            output_dir: Path,
                            seed: int = 42) -> tuple:
    """
    Build filtered train/val path lists with 1:1 positive:background ratio.
    No large-slice filter applied — all positive slices are included.

    Returns:
        Tuple of (train_txt_path, val_txt_path, data_yaml_path)
    """
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing train/val txt files from Phase 4
    train_txt = splits_dir / "yolo_train.txt"
    val_txt   = splits_dir / "yolo_val.txt"

    if not train_txt.exists() or not val_txt.exists():
        raise FileNotFoundError(
            f"Phase 4 path lists not found in {splits_dir}. "
            "Ensure Phase 4 preprocessing is complete."
        )

    train_paths = train_txt.read_text().strip().splitlines()
    val_paths   = val_txt.read_text().strip().splitlines()

    print(f"Original train paths : {len(train_paths)}")
    print(f"Original val paths   : {len(val_paths)}")

    # Filter training set: separate positives and backgrounds
    # A slice is positive if its label file is non-empty
    print("\nFiltering training set...")

    train_positives   = []
    train_backgrounds = []

    for img_path in train_paths:
        # Derive label path: replace 'images' folder with 'labels', .png -> .txt
        label_path = Path(img_path.replace("/images/", "/labels/")).with_suffix(".txt")

        if label_path.exists() and label_path.stat().st_size > 0:
            train_positives.append(img_path)
        else:
            train_backgrounds.append(img_path)

    print(f"  Positive slices    : {len(train_positives)}")
    print(f"  Background slices  : {len(train_backgrounds)}")

    # Sample backgrounds to match positive count (1:1 ratio)
    n_bg = len(train_positives)
    if len(train_backgrounds) >= n_bg:
        sampled_backgrounds = random.sample(train_backgrounds, n_bg)
    else:
        # Shouldn't happen but handle gracefully
        sampled_backgrounds = train_backgrounds
        print(f"  WARNING: fewer backgrounds than positives, using all {len(train_backgrounds)}")

    filtered_train = train_positives + sampled_backgrounds
    random.shuffle(filtered_train)

    print(f"  Filtered train set : {len(filtered_train)} images")
    print(f"  Ratio              : 1:1 positive:background")

    # Filter validation set: same 1:1 ratio approach
    print("\nFiltering validation set...")

    val_positives   = []
    val_backgrounds = []

    for img_path in val_paths:
        label_path = Path(img_path.replace("/images/", "/labels/")).with_suffix(".txt")

        if label_path.exists() and label_path.stat().st_size > 0:
            val_positives.append(img_path)
        else:
            val_backgrounds.append(img_path)

    print(f"  Positive slices    : {len(val_positives)}")
    print(f"  Background slices  : {len(val_backgrounds)}")

    n_val_bg = len(val_positives)
    if len(val_backgrounds) >= n_val_bg:
        sampled_val_backgrounds = random.sample(val_backgrounds, n_val_bg)
    else:
        sampled_val_backgrounds = val_backgrounds

    filtered_val = val_positives + sampled_val_backgrounds
    random.shuffle(filtered_val)

    print(f"  Filtered val set   : {len(filtered_val)} images")

    # Write filtered path lists
    retrain_train_txt = output_dir / "retrain_train.txt"
    retrain_val_txt   = output_dir / "retrain_val.txt"

    retrain_train_txt.write_text("\n".join(filtered_train))
    retrain_val_txt.write_text("\n".join(filtered_val))

    print(f"\nWritten: {retrain_train_txt}")
    print(f"Written: {retrain_val_txt}")

    # Write data.yaml for this retrain
    data_yaml = {
        'train': str(retrain_train_txt),
        'val'  : str(retrain_val_txt),
        'nc'   : 1,
        'names': ['kidney']
    }

    retrain_yaml = output_dir / "retrain_data.yaml"
    with open(retrain_yaml, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"Written: {retrain_yaml}")

    return str(retrain_train_txt), str(retrain_val_txt), str(retrain_yaml)

# Copy dataset to local
def copy_dataset_to_local(splits_dir: Path,
                           slices_dir: Path,
                           local_dir: Path,
                           filtered_train_txt: str,
                           filtered_val_txt: str) -> tuple:
    """
    Copy detection_train cases to local Colab storage using rsync.
    Copies entire case directories at once — much faster than file-by-file.
    """
    print("\nCopying dataset to local storage...")
    print("Using rsync for fast directory-level copying.")

    local_dir.mkdir(parents=True, exist_ok=True)

    # Copy entire detection_train folder structure to local
    src = str(slices_dir / "detection_train") + "/"
    dst = str(local_dir / "detection_train") + "/"

    print(f"  Source : {src}")
    print(f"  Dest   : {dst}")

    ret = os.system(f"rsync -a --info=progress2 '{src}' '{dst}'")

    if ret != 0:
        raise RuntimeError("rsync failed. Check Drive mount and source path.")

    print("rsync complete.")

    # Now rebuild path lists pointing to local storage
    train_paths = Path(filtered_train_txt).read_text().strip().splitlines()
    val_paths   = Path(filtered_val_txt).read_text().strip().splitlines()

    new_train_paths = []
    new_val_paths   = []

    for img_path in train_paths:
        # Replace Drive path prefix with local path
        # e.g. .../slices/detection_train/case_00000/images/slice_0000.png
        #   -> /content/yolo_retrain_data/detection_train/case_00000/images/slice_0000.png
        rel  = Path(img_path).relative_to(slices_dir)
        new_train_paths.append(str(local_dir / rel))

    for img_path in val_paths:
        rel  = Path(img_path).relative_to(slices_dir)
        new_val_paths.append(str(local_dir / rel))

    # Write updated path lists
    local_train_txt = local_dir / "local_train.txt"
    local_val_txt   = local_dir / "local_val.txt"

    local_train_txt.write_text("\n".join(new_train_paths))
    local_val_txt.write_text("\n".join(new_val_paths))

    # Write updated data.yaml
    local_yaml = local_dir / "local_data.yaml"
    data_yaml  = {
        'train': str(local_train_txt),
        'val'  : str(local_val_txt),
        'nc'   : 1,
        'names': ['kidney']
    }
    with open(local_yaml, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"\nLocal dataset ready:")
    print(f"  Train images : {len(new_train_paths)}")
    print(f"  Val images   : {len(new_val_paths)}")
    print(f"  Data yaml    : {local_yaml}")

    return str(local_train_txt), str(local_val_txt), str(local_yaml)

# Model training
def train_yolo(data_yaml: str,
               results_dir: Path,
               run_name: str = "yolov8s_retrain"):
    """
    Train YOLOv8s with settings validated in Phase 5.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolov8s.pt")

    print(f"\nStarting training...")
    print(f"  Data yaml    : {data_yaml}")
    print(f"  Results dir  : {results_dir}")
    print(f"  Run name     : {run_name}")

    model.train(
        data       = data_yaml,
        epochs     = 100,
        imgsz      = 512,
        batch      = 16,
        optimizer  = "Adam",
        lr0        = 0.001,
        patience   = 0,          # Disable early stopping — val metrics too noisy
        project    = str(results_dir),
        name       = run_name,
        exist_ok   = True,
        verbose    = True,
        device     = 0,          # GPU
        workers    = 4,
    )

    # Report best model location
    best_pt = results_dir / run_name / "weights" / "best.pt"
    print(f"\nTraining complete.")
    print(f"Best model: {best_pt}")
    print(f"Exists    : {best_pt.exists()}")


# Main
def main():
    config_path = "/content/kidney-tumour-detection/configs/config.yaml"
    config      = load_config(config_path)

    splits_dir  = Path(config['paths']['splits_dir'])
    slices_dir  = Path(config['paths']['slices_dir'])
    results_dir = Path("/content/drive/MyDrive/kidney-tumour-detection/results/phase5_yolo_retrain")
    seed        = config['dataset']['random_seed']

    # Temp dir for filtered path lists (on Drive for persistence)
    filter_output_dir = Path(config['paths']['processed_root']) / "retrain_dataset"

    # Local Colab storage for fast training
    local_dir = Path("/content/yolo_retrain_data")

    print("YOLO Retrain")
    print("Key change from Phase 5: NO large-slice filter")
    print("All positive kidney slices included in training")

    # Step 1: Build filtered dataset (all positives + 1:1 background)
    _, _, data_yaml = build_filtered_dataset(
        splits_dir  = splits_dir,
        slices_dir  = slices_dir,
        output_dir  = filter_output_dir,
        seed        = seed
    )

    # Step 2: Copy to local storage for fast read speeds
    _, _, local_yaml = copy_dataset_to_local(
        splits_dir         = splits_dir,
        slices_dir         = slices_dir,
        local_dir          = local_dir,
        filtered_train_txt = str(filter_output_dir / "retrain_train.txt"),
        filtered_val_txt   = str(filter_output_dir / "retrain_val.txt")
    )

    # Step 3: Train
    train_yolo(
        data_yaml   = local_yaml,
        results_dir = results_dir,
        run_name    = "yolov8s_retrain_run1"
    )


if __name__ == "__main__":
    main()