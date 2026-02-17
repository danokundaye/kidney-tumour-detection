# Phase 4, Step 4.5 — YOLO Dataset Structure
#   This aims to prepare the dataset structure YOLO needs for training.
#       1. train.txt  — absolute paths to all images in the 100 train cases
#       2. val.txt    — absolute paths to all images in the 10 val cases
#       3. data.yaml  — YOLO config: class names, nc, paths to train/val txt files
#
#   Steps:
#       - Split 110 detection train cases into 100 train / 10 validation
#          at patient level (stratified by malignancy)
#       - Generate train.txt and val.txt which contains one image path per line
#       - Generate data.yaml to tell YOLO where data lives and class info
#
#
# Execution: Google Colab
#
# OUTPUT STRUCTURE:
#   processed/splits/
#   ├── yolo_train.txt         (paths to ~51,400 training images)
#   ├── yolo_val.txt           (paths to ~5,200 validation images)
#   └── yolo_data.yaml         (YOLO configuration file)
#
# NOTE:
#   Labels are located alongside images (case_id/labels/) so YOLO finds
#   them automatically by replacing 'images' with 'labels' in each path.

import json
import yaml
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Load config.yaml
def load_config(config_path: str) -> dict:
    """
    Load the config.yaml file
    All paths and settings come from here
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# Split into training and validation datasets
def split_detection_cases(
        detection_csv: str,
        kits_json: str,
        val_size: int = 10,
        random_state: int = 42
) -> tuple:
    """
    Split 110 detection train cases into train and val sets.
    Stratified by malignancy to maintain class proportions in both sets.

    Args:
        case_ids : Full list of detection_train case IDs
        val_size : Number of cases to reserve for validation
        seed     : Random seed for to guarantee the same 10 val cases

    Returns:
        train_cases, val_cases
    """
     # Load the 110 detection train cases
    cases_df = pd.read_csv(detection_csv)

    # Load malignancy labels from kits.json
    # kits.json contains one entry per case with clinical metadata
    with open(kits_json, 'r') as f:
        kits_data = json.load(f)

    # Lookup malignancy -> True/False
    # convert integer case_id in kits.json to string
    malignancy_map = {}
    for entry in kits_data:
        case_id = f"case_{entry['case_id']:05d}"
        # Default to malignant if field is missing
        malignancy_map[case_id] = entry.get('malignant', True)

    # Add malignancy column to cases dataframe for stratification
    cases_df['malignant'] = cases_df['case_id'].map(malignancy_map)

    # Report class distribution before splitting
    n_malignant = cases_df['malignant'].sum()
    n_benign    = len(cases_df) - n_malignant
    print(f"  Malignant cases : {n_malignant}")
    print(f"  Benign cases    : {n_benign}")

    # Stratified split — val_size cases reserved for validation
    # 'stratify =' ensures malignant/benign ratio is preserved in both sets
    val_fraction = val_size / len(cases_df)
    train_cases, val_cases = train_test_split(
        cases_df['case_id'].tolist(),
        test_size    = val_fraction,
        stratify     = cases_df['malignant'].tolist(),
        random_state = random_state
    )

    return train_cases, val_cases

# Arranging image paths
def collect_image_paths(cases: list, detection_dir: Path) -> list:
    """
    YOLO reads train.txt and val.txt line by line during training.
    Each line must be the absolute path to one image file.
    YOLO then automatically locates the label file by replacing
    'images' with 'labels' in that path — no explicit label path needed.

    Args:
        cases         : List of case ID strings e.g. ['case_00000', ...]
        detection_dir : Path to the detection_train folder on Drive

    Returns:
        Sorted list of absolute image path strings
    """
    paths = []

    for case_id in sorted(cases):
        images_dir = detection_dir / case_id / "images"

        # Print warning if a folder is missing
        if not images_dir.exists():
            print(f"  WARNING: images folder missing for {case_id}")
            continue

        # Collect all PNG slices for this case
        case_images = sorted(images_dir.glob("*.png"))
        paths.extend([str(p) for p in case_images])

    return paths

# Writing info to files
def write_txt(paths: list, output_path: Path) -> None:
    """
    Write one image path per line to a .txt file.
    This is the format YOLO expects for its train and val file lists.

    Args:
        paths       : List of absolute image path strings
        output_path : Where to save the .txt file
    """
    with open(output_path, 'w') as f:
        for path in paths:
            f.write(path + '\n')
    print(f"  Written : {output_path} ({len(paths)} images)")


def write_data_yaml(
        train_txt: Path,
        val_txt: Path,
        output_path: Path
) -> None:
    """
    Write yolo_data.yaml — the configuration file YOLO reads before training.

    Required fields:
        nc    : Number of classes. There is only 1 (kidney only).
        names : List of class names. Index = class_id in label files.
                Index 0 = 'kidney', matching class_id 0 in .txt labels.
        train : Absolute path to yolo_train.txt
        val   : Absolute path to yolo_val.txt

    Args:
        train_txt   : Path to the generated yolo_train.txt file
        val_txt     : Path to the generated yolo_val.txt file
        output_path : Where to save yolo_data.yaml
    """
    data = {
        'nc'   : 1,              
        'names': ['kidney'],     
        'train': str(train_txt),
        'val'  : str(val_txt)
    }

    with open(output_path, 'w') as f:
        # 'sort_keys = False' preserves the field order above for readability
        yaml.dump(data, f, default_flow_style = False, sort_keys = False)

    print(f"  Written : {output_path}")

# Main
def main():
    config_path = "/content/kidney-tumour-detection/configs/config.yaml"
    config      = load_config(config_path)

    slices_dir = Path(config['paths']['slices_dir'])
    splits_dir = Path(config['paths']['splits_dir'])
    seed       = config['dataset']['random_seed']

    # dataset_root holds kits.json alongside the case folders
    kits_root  = Path(config['paths']['dataset_root'])
    kits_json     = kits_root / "kits.json"

    # detection_train folder contains all 110 cases with images/masks/labels
    detection_dir = slices_dir / "detection_train"

    # Output file paths — saved to splits_dir alongside the split CSVs
    train_txt = splits_dir / "yolo_train.txt"
    val_txt   = splits_dir / "yolo_val.txt"
    data_yaml = splits_dir / "yolo_data.yaml"

    # Split 110 detection cases into 100 train / 10 val
    print("\nYOLO Dataset Structure")
    print("\nStep 1: Splitting detection cases...")
    train_cases, val_cases = split_detection_cases(
        detection_csv = str(splits_dir / "detection_train.csv"),
        kits_json     = str(kits_json),
        val_size      = 10,
        random_state  = seed
    )
    print(f"  Train cases : {len(train_cases)}")
    print(f"  Val cases   : {len(val_cases)}")
    print(f"  Val set     : {sorted(val_cases)}")

    # Collect absolute image paths for each split
    print("\nStep 2: Collecting image paths...")
    train_paths = collect_image_paths(train_cases, detection_dir)
    val_paths   = collect_image_paths(val_cases,   detection_dir)
    print(f"  Train images : {len(train_paths)}")
    print(f"  Val images   : {len(val_paths)}")

    # Write the three output files YOLO needs
    print("\nStep 3: Writing output files...")
    splits_dir.mkdir(parents = True, exist_ok = True)
    write_txt(train_paths, train_txt)
    write_txt(val_paths,   val_txt)
    write_data_yaml(train_txt, val_txt, data_yaml)

    # Verification checks
    print("\nStep 4: Verification checks...")

    # Total image count should match Step 4.4 output (56,604 slices)
    total = len(train_paths) + len(val_paths)
    assert total == 56604, f"Image count mismatch: expected 56604, got {total}"
    print(f"  Total images (train + val) : {total} — matches Step 4.4? YES")

    # Confirm a sample path actually exists on Drive
    # Failure here means Drive is not mounted or paths are wrong
    sample = Path(train_paths[0])
    print(f"  Sample path exists         : {'YES' if sample.exists() else 'NO — CHECK DRIVE MOUNT'}")
    print(f"  Sample                     : {train_paths[0]}")

    print("\nStep 4.5 complete.")
    print("Next: Phase 5 — YOLOv8 Training")


if __name__ == "__main__":
    main()