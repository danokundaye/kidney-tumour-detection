# This script audits all 300 KiTS21 cases and produces a comprehensive report covering:
#     - File integrity (are all cases present and loadable?)
#     - Image dimensions and slice counts
#     - CT intensity ranges (informs our normalization strategy)
#     - Segmentation label distribution (kidney, tumor, cyst presence)
#     - Histology label availability (determines EfficientNet training size)
#
# Execution: Google Colab (data lives on Drive)
# Output: Reports saved to Drive for reference during documentation

import os
import json
import yaml
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Load config.yaml
def load_config(config_path: str) -> dict:
    """
    Load the config.yaml file
    All paths and settings come from here
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# File integrity check
def check_integrity(dataset_root: str) -> pd.DataFrame:
    """
    For every case in the dataset, check:
    - Does the case folder exist?
    - Does imaging.nii.gz exist?
    - Does segmentation.nii.gz exist?
    - Can both files actually be loaded without errors?

    Returns a DataFrame with one row per case summarising findings.
    """
    print("\n" + "="*50)
    print("Step 1: File Integrity Check")

    dataset_root = Path(dataset_root)
    results = []

    # KiTS21 cases are named case_00000 to case_00299
    case_ids = [f"case_{str(i).zfill(5)}" for i in range(300)]

    for case_id in tqdm(case_ids, desc="Checking cases..."):
        case_dir = dataset_root / case_id
        imaging_path = case_dir / "imaging.nii.gz"
        seg_path = case_dir / "segmentation.nii.gz"

        row = {
            "case id": case_id,
            "folder exists": case_dir.exists(),
            "imaging exists": imaging_path.exists(),
            "segmentation exists": seg_path.exists(),
            "imaging_loadble": False,
            "segmentation_loadble": False,
            "num_slices": None,
            "height": None,
            "width": None,
            "error": None
        }

        # Only attempt to load if file exists
        if imaging_path.exists() and seg_path.exists():
            try:
                img = nib.load(str(imaging_path))
                seg = nib.load(str(seg_path))


                img_shape = img.header.get_data_shape()
                seg_shape = seg.header.get_data_shape()

                row["imaging_loadable"] = True
                row["seg_loadable"] = True


                # Get NIfTI shape
                row["height"] = img_shape[0]
                row["width"] = img_shape[1]
                row["num_slices"] = img_shape[2]

                # Flag imaging and segmentation mismatch
                if img_shape != seg_shape:
                    row["error"] = f"Shape mismatch: img={img_shape}, seg={seg_shape}"

            except Exception as e:
                row["error"] = str(e)

        results.append(row)

    df = pd.DataFrame(results)

    # Print summary
    print(f"\nTotal cases expected:        300")
    print(f"Folders found:               {df['folder_exists'].sum()}")
    print(f"Imaging files found:         {df['imaging_exists'].sum()}")
    print(f"Segmentation files found:    {df['segmentation_exists'].sum()}")
    print(f"Successfully loaded (both):  {df['imaging_loadable'].sum()}")

    # Report problematic cases
    problem_cases = df[
        ~df['folder_exists'] |
        ~df['imaging_exists'] |
        ~df['segmentation_exists'] |
        df['error'].notna()
    ]

    if len(problem_cases) > 0:
        print(f"\nProblem Cases Found: {len(problem_cases)}")
        print(problem_cases[['case_id', 
                             'folder_exists',
                             'imaging_exists', 
                             'segmentation_exists',
                             'error']].to_string())
    else:
        print(f"\nAll cases passed integrity check")

    return df

# Slice Count Analysis
def analyse_slice_distribution(integrity_df: pd.DataFrame) -> None:
    """
    Analyze and visualize the distribution of slice counts across all cases.
    This matters because:
    - Too few slices = risk of missing the tumor entirely
    - High variation = our pipeline must handle variable-length inputs
    """
    print("\n" + "="*50)
    print("Step 2: Slice Count Distribution")

    slices = integrity_df['num_slices'].dropna()

    print(f"Min slices:    {int(slices.min())}")
    print(f"Max slices:    {int(slices.max())}")
    print(f"Mean slices:   {slices.mean():.2f}")
    print(f"Median slices: {slices.median():.2f}")
    print(f"Std deviation: {slices.std():.2f}")

    # Cases below 50 slices could be problematic for 3D reconstruction
    low_slice_cases = integrity_df[integrity_df['num_slices'] < 50]
    if len(low_slice_cases) > 0:
        print(f"\nCases with fewer than 50 slices: {len(low_slice_cases)}")
        print(low_slice_cases[['case_id', 'num_slices']].to_string())
    else:
        print(f"\nAll cases have 50+ slices")

# Intensity Statistics
def analyse_intensity(
        dataset_root: str,
        integrity_df: pd.DataFrame,
        sample_size: int = 20,
        random_seed: int = 42
) -> dict:
    """
    Load a random sample of cases and compute CT intensity statistics.

    Why sample and not all 300?
    Loading all 300 NIfTI volumes into memory would take 30+ minutes
    and potentially crash Colab. 20 cases gives us a reliable estimate
    of the intensity range without the overhead.

    The statistics here directly inform our CT windowing values in config.yaml.
    """
    print("\n" + "="*50)
    print(f"Step 3: Intensity Statistics (sample of {sample_size} cases)")

    dataset_root = Path(dataset_root)
    np.random.seed(random_seed)

    # Only use samples from loaded cases
    valid_cases = integrity_df[integrity_df['imaging_loadable']]['case_id'].tolist()
    sampled_cases = np.random.choice(valid_cases, size=sample_size, replace=False)

    all_mins, all_maxs, all_means, all_stds = [], [], [], []


    for case_id in tqdm(sampled_cases, desc="Computing intensity stats..."):
        img_path = dataset_root / case_id / "imaging.nii.gz"
        img_data = nib.load(str(img_path)).get_fdata()

        all_mins.append(img_data.min())
        all_maxs.append(img_data.max())
        all_means.append(img_data.mean())
        all_stds.append(img_data.std())

        # Free memory immediately after use
        del img_data

        stats = {
        "global_min": np.min(all_mins),
        "global_max": np.max(all_maxs),
        "mean_of_means": np.mean(all_means),
        "mean_of_stds": np.mean(all_stds),
        "p5": np.percentile(all_mins, 5),
        "p95": np.percentile(all_maxs, 95)
    }
    
    print(f"\nGlobal intensity min:  {stats['global_min']:.1f} HU")
    print(f"Global intensity max:  {stats['global_max']:.1f} HU")
    print(f"Mean of case means:    {stats['mean_of_means']:.1f} HU")
    print(f"Mean of case stds:     {stats['mean_of_stds']:.1f} HU")
    print(f"5th percentile min:    {stats['p5']:.1f} HU")
    print(f"95th percentile max:   {stats['p95']:.1f} HU")
    print(f"\nConfig window range:   -79 to 304 HU")
    print(f"→ Review if global min/max suggests a different window is needed")

    return stats

# Segmentation Label Analysis

def analyse_seg_labels(
        dataset_root: str,
        integrity_df: pd.DataFrame,
        sample_size: int = 50,
        random_seed: int = 42
) -> pd.DataFrame:
    """
    For a sample of cases, check which segmentation labels are present.
    KiTS21 labels: 0=background, 1=kidney, 2=tumor, 3=cyst

    This tells us:
    - What fraction of cases have tumors (label 2)
    - What fraction have cysts (label 3)
    - Are there any cases with no kidney label at all? (would be anomalous)
    """
    print("\n" + "="*50)
    print(f"Step 4: Segmentation Label Analysis (sample of {sample_size} cases)")

    dataset_root = Path(dataset_root)
    np.random.seed(random_seed)

    # Only use samples from loaded cases
    valid_cases = integrity_df[integrity_df['segmentation_loadable']]['case_id'].tolist()
    sampled_cases = np.random.choice(valid_cases, size=min(sample_size, len(valid_cases)), replace=False)

    label_results = []

    for case_id in tqdm(sampled_cases, desc="Analysing labels..."):
        img_path = dataset_root / case_id / "segmentation.nii.gz"
        seg_data = nib.load(str(img_path)).get_fdata().astype(np.uint8)

        unique_labels = np.unique(seg_data)

        label_results.append({
            "case_id": case_id,
            "has_kidney": 1 in unique_labels,
            "has_tumor": 2 in unique_labels,
            "has_cyst": 3 in unique_labels,
            "unique_labels": str(unique_labels.tolist())
        })

        del seg_data

    label_df = pd.DataFrame(label_results)

    print(f"\nOut of {len(sampled_cases)} sampled cases:")
    print(f"Has kidney label (1): {label_df['has_kidney'].sum()} "
          f"({label_df['has_kidney'].mean()*100:.1f}%)")
    print(f"Has tumor label  (2): {label_df['has_tumor'].sum()} "
          f"({label_df['has_tumor'].mean()*100:.1f}%)")
    print(f"Has cyst label   (3): {label_df['has_cyst'].sum()} "
          f"({label_df['has_cyst'].mean()*100:.1f}%)")
    
    # Flag cases missing kidney label    
    no_kidney = label_df[~label_df['has_kidney']]
    if len(no_kidney) > 0:
        print(f"\nCases with NO kidney label: {len(no_kidney)}")
        print(no_kidney[['case_id']].to_string())
    else:
        print(f"\nAll sampled cases have kidney label")

    return label_df

# Histology Label Analysis
def analyse_histology(dataset_root: str) -> pd.DataFrame:
    """
    KiTS21 includes a JSON metadata file with clinical information per case.
    This function reads it to find how many cases have confirmed histology
    labels (benign vs malignant) — the training data for EfficientNet.

    We don't know the exact number until we read this file.
    """
    print("\n" + "="*50)
    print("Step 5: Histology Label Analysis")

    dataset_root = Path(dataset_root)

    # KiTS21 metadata JSON — check both possible locations
    possible_json_paths = [
        dataset_root / "kits21.json",
        dataset_root / "data" / "kits21.json",
        dataset_root.parent / "kits21.json",
    ]

    json_path = None
    for path in possible_json_paths:
        if path.exists():
            json_path = path
            break

    if json_path is None:
        print("KiTS21.json not found. Searched:")
        for p in possible_json_paths:
            print(f"{p}")
        print("\nThis file contains histology labels for EfficientNet training.")
        print("Check your Drive for where this file is located and update paths.")
        return pd.DataFrame()
    
    print(f"Found metadata at: {json_path}")

    with open(json_path, 'r') as f:
        metadata = json.load(f)

    print(f"\nTotal entries in JSON: {len(metadata)}")
    print(f"\nSample entry keys: {list(metadata[0].keys()) if metadata else 'empty'}")

    # Build DataFrame from metadata
    rows = []
    for entry in metadata:
        row = {"case_id": entry.get("case_id", "unknown")}

        # Look for histology-related fields
        # We print all keys first so we can identify the right field name
        for key, value in entry.items():
            row[key] = value
        rows.append(row)

    meta_df = pd.DataFrame(rows)

    # Find columns that likely contain histology information
    histology_cols = [
        col for col in meta_df.columns
        if any(term in col.lower() for term in
               ['histol', 
                'malig', 
                'benign', 
                'tumor_type',
                'pathol', 
                'subtype', 
                'label'])
    ]

    print(f"\nPotential histology-related columns found:")
    if histology_cols:
        for col in histology_cols:
            print(f"\nColumn: '{col}'")
            print(f"Unique values: {meta_df[col].unique()}")
            print(f"Non-null count: {meta_df[col].notna().sum()} / {len(meta_df)}")
    else:
        print("None identified automatically.")
        print("All available columns:")
        for col in meta_df.columns:
            print(f"  - {col}: {meta_df[col].nunique()} unique values")

    return meta_df

# Save Reports
def save_reports(logs_dir: str,
        integrity_df: pd.DataFrame,
        label_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        intensity_stats: dict
) -> None:
    """
    Save all exploration results to Drive as CSV and text files.
    These become reference material for documentation.
    """
    print("\n" + "="*50)
    print("Step 6: Saving Report")

    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents = True, exist_ok = True)

    # Save integrity report
    integrity_path = logs_dir / "exploration_integrity.csv"
    integrity_df.to_csv(integrity_path, index = False)
    print(f"Integrity report saved: {integrity_path}")

     # Save label analysis
    if len(label_df) > 0:
        label_path = logs_dir / "exploration_labels.csv"
        label_df.to_csv(label_path, index = False)
        print(f"Label report saved: {label_path}")
    
    # Save metadata/histology
    if len(meta_df) > 0:
        meta_path = logs_dir / "exploration_metadata.csv"
        meta_df.to_csv(meta_path, index=False)
        print(f"Metadata saved:  {meta_path}")

    # Save intensity stats as text
    stats_path = logs_dir / "exploration_intensity_stats.txt"
    with open(stats_path, 'w') as f:
        f.write("KiTS21 Intensity Statistics (sampled)\n")
        f.write("="*40 + "\n")
        for key, value in intensity_stats.items():
            f.write(f"{key}: {value:.2f}\n")
    print(f"Intensity stats saved:  {stats_path}")

    print(f"\nAll reports saved to: {logs_dir}")

# Main
def main():
    # Load config
    # Update this path if your repo is cloned to a different location on Colab

    config_path = "/content/kidney-tumour-detection/configs/config.yaml"
    config = load_config(config_path)

    dataset_root = config['paths']['dataset_root']
    logs_dir = config['paths']['logs_dir']
    random_seed = config['dataset']['random_seed']

    print("\nKiTS21 DATASET EXPLORATION")
    print("\n" + "="*50)
    print(f"Dataset root: {dataset_root}")
    print(f"Logs output:  {logs_dir}")

    integrity_df = check_integrity(dataset_root)
    analyse_slice_distribution(integrity_df)

    intensity_stats = analyse_intensity(
        dataset_root, integrity_df,
        sample_size=20,
        random_seed=random_seed
    )

    label_df = analyse_seg_labels(
        dataset_root, integrity_df,
        sample_size=50,
        random_seed=random_seed
    )

    meta_df = analyse_histology(dataset_root)

    save_reports(logs_dir, integrity_df, label_df, meta_df, intensity_stats)

    print("\n" + "="*50)
    print("EXPLORATION COMPLETE")
    print("="*50)
    print("Review the output above before proceeding to Step 4.2")
    print("Key things to confirm:")
    print("  1. All 300 cases loaded successfully")
    print("  2. Intensity range aligns with config window (-79 to 304 HU)")
    print("  3. Note how many cases have tumor labels (label 2)")
    print("  4. Note how many cases have confirmed histology labels")


if __name__ == "__main__":
    main()