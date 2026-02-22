# Step 6.3 — U-Net Data Splitting
# 
#   THis script splits 120 segmentation_train cases into 108 train / 12 validation at
#   patient level. It is stratified by abnormality, so both sets have proportional
#   representation of tumour, cyst, and healthy-only cases.
#
# Input:
#   - splits/segmentation_train.csv  (120 case IDs)
#   - dataset/kits.json              (histology labels per case)
#   - unet_crops/                    (region_types to determine abnormality)
#
# Execution: Google Colab
#
# Output:
#   - splits/unet_train.csv          (108 cases)
#   - splits/unet_val.csv            (12 cases)

import json
import yaml
import numpy as np
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
        return yaml.safe_load(f)

# Loop region_type file to obtain labels
def get_case_abnormality(case_id: str, crops_dir: Path) -> str:
    """
    Determine the abnormality type for a case by reading its region_type files.

    Scans all region_type txt files for a case and returns the highest
    priority abnormality found:
        - 'tumour'  : case contains at least one tumour slice
        - 'cyst'    : case contains cyst but no tumour
        - 'healthy' : case contains only healthy kidney slices

    Args:
        case_id   : e.g. 'case_00002'
        crops_dir : path to unet_crops directory

    Returns:
        string: 'tumour', 'cyst', or 'healthy'
    """
    regions_dir = crops_dir / case_id / "region_types"

    if not regions_dir.exists():
        return 'healthy'

    region_files = list(regions_dir.glob("*.txt"))

    if not region_files:
        return 'healthy'

    has_tumour = False
    has_cyst   = False

    # Check each file and update boolean values to corresponding labels
    for rf in region_files:
        region_type = rf.read_text().strip()
        if region_type in ("tumour_only", "both"):
            has_tumour = True
        elif region_type == "cyst_only":
            has_cyst = True

    if has_tumour:
        return 'tumour'
    elif has_cyst:
        return 'cyst'
    else:
        return 'healthy'

# Main
def main():
    config_path = "/content/kidney-tumour-detection/configs/config.yaml"
    config      = load_config(config_path)

    splits_dir  = Path(config['paths']['splits_dir'])
    crops_dir   = Path(config['paths']['unet_crops_dir'])
    kits_json   = Path(config['paths']['kits_root']) / "kits.json"

    print("Step 6.3 — U-Net Data Splitting")

    # Load 120 segmentation_train cases
    seg_csv  = splits_dir / "segmentation_train.csv"
    cases_df = pd.read_csv(seg_csv)
    case_ids = cases_df['case_id'].tolist()
    print(f"Segmentation train cases : {len(case_ids)}")

    # Load kits.json for histology labels
    with open(kits_json, 'r') as f:
        kits_data = json.load(f)

    # Build a lookup: case_id and its corresponding histology label
    # kits.json is a list of dicts with 'case_id' and 'tumor_histologic_subtype'
    histology_lookup = {}
    for entry in kits_data:
        cid   = entry.get('case_id', '')
        label = entry.get('tumor_histologic_subtype', 'unknown')
        histology_lookup[cid] = label

    # Determine abnormality type for each case from region_types
    print("\nAnalysing region types per case...")
    records = []
    for case_id in case_ids:
        abnormality = get_case_abnormality(case_id, crops_dir)
        histology   = histology_lookup.get(case_id, 'unknown')
        records.append({
            'case_id'    : case_id,
            'abnormality': abnormality,
            'histology'  : histology
        })

    df = pd.DataFrame(records)

    # Print distribution before split
    print(f"\nAbnormality distribution (120 cases):")
    print(df['abnormality'].value_counts().to_string())

    # Stratify by abnormality type
    # This ensures both train and val sets get proportional tumour/cyst/healthy cases
    train_ids, val_ids = train_test_split(
        df['case_id'].tolist(),
        test_size    = 24,          # exactly 12 val cases
        random_state = 42,
        stratify     = df['abnormality'].tolist()
    )

    print(f"\nSplit result:")
    print(f"  Train : {len(train_ids)} cases")
    print(f"  Val   : {len(val_ids)} cases")

    # Print stratification check
    train_df = df[df['case_id'].isin(train_ids)]
    val_df   = df[df['case_id'].isin(val_ids)]

    print(f"\nTrain abnormality distribution:")
    print(train_df['abnormality'].value_counts().to_string())

    print(f"\nVal abnormality distribution:")
    print(val_df['abnormality'].value_counts().to_string())

    # Save CSVs
    train_out = splits_dir / "unet_train.csv"
    val_out   = splits_dir / "unet_val.csv"

    train_df[['case_id', 'abnormality', 'histology']].to_csv(train_out, index = False)
    val_df[['case_id', 'abnormality', 'histology']].to_csv(val_out,   index = False)

    print(f"\nSaved:")
    print(f"  {train_out}")
    print(f"  {val_out}")
    print("\nStep 6.3 complete.")


if __name__ == "__main__":
    main()