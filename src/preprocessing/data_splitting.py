# Phase 4, Step 4.2 — Patient-Level Data Splitting
#   Divide all 300 KiTS21 cases into three non-overlapping groups at the patient level. This prevents data leakage — all slices from one patient stay in the same group.

#   Split:
#     - 110 cases → YOLOv8 detection training
#     - 120 cases → U-Net segmentation training
#     -  70 cases → Final testing

# Stratification:
#     The dataset has 275 malignant and 25 benign cases (11:1 imbalance).
#     Use stratified splitting to ensure each group receives a proportional share of benign cases — preventing all benign cases from landing in one group by chance.

# OUTPUT:
#   - splits/detection_train.csv    (110 cases)
#   - splits/segmentation_train.csv (120 cases)
#   - splits/test.csv               ( 70 cases)
#   - splits/split_summary.txt      (human-readable summary)

# Execution: Google Colab

import json
import yaml
import pandas as pd
import numpy as np
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


# Load case labels
def load_case_labels(kits_json_path: str) -> pd.DataFrame:
    """
    Load kits.json and extract case_id and malignant label for each case.

    Returns a DataFrame with columns:
        - case_id   : e.g. 'case_00000'
        - malignant : True (malignant) or False (benign)
        - label_str : 'malignant' or 'benign' (human-readable)
    """
    with open(kits_json_path, 'r') as f:
        kits_data = json.load(f)

    rows = [
        {
            'case_id'   : entry['case_id'],
            'malignant' : entry.get('malignant', None),
            'label_str' : 'malignant' if entry.get('malignant') else 'benign'
        }
        for entry in kits_data
    ]

    df = pd.DataFrame(rows)
    print(f"Total cases loaded: {len(df)}")
    print(f" Malignant: {df['malignant'].sum()}")
    print(f" Benign:    {(~df['malignant']).sum()}")

    return df


# Stratified Split
def stratified_split(
        cases_df: pd.DataFrame,
        detection_n: int,
        segmentation_n: int,
        test_n: int,
        random_seed: int
) -> tuple:
    """
    Split cases into three non-overlapping stratified groups.

    Strategy:
        Step 1 — Split off test_n cases, stratified by malignant label
        Step 2 — From the remaining cases, split off detection_n cases,
                 stratified by malignant label
        Step 3 — Whatever remains becomes the segmentation training set

    Args:
        cases_df      : DataFrame with case_id and malignant columns
        detection_n   : Number of cases for YOLOv8 training (110)
        segmentation_n: Number of cases for U-Net training (120)
        test_n        : Number of cases for final testing (70)
        random_seed   : Fixed seed for reproducibility
    
        Returns:
        Tuple of three DataFrames: (detection_df, segmentation_df, test_df)
    """
    total = len(cases_df)
    assert detection_n + segmentation_n + test_n == total
    print(f"\n Splits must sum to {total}. {detection_n + segmentation_n + test_n} splits confirmed.")

    # Separate test case
    test_fraction = test_n/total                            # 70/300

    remaining_df, test_df = train_test_split(
        cases_df,
        test_size = test_fraction,
        stratify = cases_df['malignant'],                   # Ensures both groups have the same proportion of
                                                            # benign cases
        random_state = random_seed
    )

    # Separate detection training set from remaining 230 cases
    detection_fraction = detection_n/len(remaining_df)      # 110/230

    segmentation_df, detection_df = train_test_split(
        remaining_df,
        test_size = detection_fraction,
        stratify = remaining_df['malignant'],
        random_state = random_seed
    )

    # Whatever is left is the segmentation training set

    return detection_df, segmentation_df, test_df


# Verify and Print Split Summary
def print_split_summary(
        detection_df: pd.DataFrame,
        segmentation_df: pd.DataFrame,
        test_df: pd.DataFrame
) -> str:
    """
    Print a summary of the split results.
    """
    lines = []
    lines.append("="*50)
    lines.append("Patient-level Split Summary")

    for name, df in [("Detection Train", detection_df),
                     ("Segmentation Train", segmentation_df),
                     ("Test", test_df)]:
        malignant_n = df['malignant'].sum()
        benign_n    = (~df['malignant'].sum())
        total_n     = len(df)
        lines.append(f"\n{name} ({total_n} cases):")
        lines.append(f"   Malignant : {malignant_n / total_n * 100:.1f}")
        lines.append(f"   Benign    : {benign_n / total_n * 100:.1f}")

    # Confirm none of the cases appears in more than one split
    all_ids = (
        set(detection_df['case_id']) |
        set(segmentation_df['case_id']) |
        set(test_df['case_id'])
    )

    total_unique = len(all_ids)
    total_assigned = len(detection_df) + len(segmentation_df) + len(test_df)

    lines.append(f"\n Uniqueness Check")
    lines.append(f"  Total cases assigned : {total_assigned}")
    lines.append(f"  Unique case IDs      : {total_unique}")

    if total_unique == total_assigned == 300:
        lines.append(f"  No overlaps detected : PASS")
    else:
        lines.append(f"  WARNING: Overlap or missing cases detected!")

    summary = "\n".join(lines)
    print(summary)
    return summary


# Save Splits to Drive
def save_splits(
        splits_dir: str,
        detection_df: pd.DataFrame,
        segmentation_df: pd.DataFrame,
        test_df: pd.DataFrame,
        summary: str
) -> None:
    """
    Save each split as a CSV and write the summary to a text file
    """
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents = True, exist_ok = True)

    detection_df.to_csv(splits_dir / "detection_train.csv", index = False)
    segmentation_df.to_csv(splits_dir / "segmentation_train.csv", index = False)
    test_df.to_csv(splits_dir / "test.csv", index = False)

    with open(splits_dir / "split_summary.txt", 'w') as f:
        f.write(summary)

    print(f"\nSplits saved to: {splits_dir}")
    print(f"  detection_train.csv    ({len(detection_df)} cases)")
    print(f"  segmentation_train.csv ({len(segmentation_df)} cases)")
    print(f"  test.csv               ({len(test_df)} cases)")
    print(f"  split_summary.txt")

# Main
def main():
    config_path = "/content/kidney-tumour-detection/configs/config.yaml"
    config = load_config(config_path)

    kits_json_path = (
        Path(config['paths']['dataset_root']).parent / "kits.json"
    )
    splits_dir  = config['paths']['splits_dir']
    random_seed = config['dataset']['random_seed']

    detection_n    = config['dataset']['detection_train_cases']
    segmentation_n = config['dataset']['segmentation_train_cases']
    test_n         = config['dataset']['test_cases']

    print("\n Patient-level Data Splitting")
    print(f"\nStrategy : Stratified by malignant label")
    print(f"Seed     : {random_seed}")
    print(f"Split    : {detection_n} detection / "
          f"{segmentation_n} segmentation / {test_n} test\n")
    
    # Load labels
    cases_df = load_case_labels(kits_json_path)

    detection_df, segmentation_df, test_df = stratified_split(
        cases_df,
        detection_n = detection_n,
        segmentation_n = segmentation_n,
        test_n = test_n,
        random_seed = random_seed
    )

    # Print summary
    summary = print_split_summary(detection_df, segmentation_df, test_df)

    # Save to Drive
    save_splits(splits_dir, detection_df, segmentation_df, test_df, summary)

    print("Splitting Complete \n")
    print("Do NOT modify the split CSVs manually after this point.")
    print("All subsequent scripts read from these files.")

if __name__ == "__main__":
    main()