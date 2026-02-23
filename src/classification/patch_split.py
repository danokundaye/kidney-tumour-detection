# Step 7.2 — EfficientNet Train/Val Split
#
# This script splits 98 patch cases into train and val sets at patient level.
#
# Benign (5 cases) — manual assignment:
#   - case_00188 (1 patch) → train (too few patches to be useful in val)
#   - Remaining 4 benign cases → 3 train, 1 val (highest patch count to val)
#
# Malignant (93 cases) — 80/20 stratified random split: 74 train, 19 val
#
# Execution: Google Colab
#
# OUTPUT STRUCTURE:
#   processed/splits/
#   ├── efficientnet_train.csv
#   └── efficientnet_val.csv
#   columns: patch_path, case_id, malignant, slice_name, source, dice

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_config(config_path: str) -> dict:
    """
    Load the config.yaml file
    All paths and settings come from here
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    config_path = "/content/kidney-tumour-detection/configs/config.yaml"
    config      = load_config(config_path)

    splits_dir = Path(config['paths']['splits_dir'])
    index_csv  = splits_dir / "patches_index.csv"

    seed = config['dataset']['random_seed']

    print("Step 7.2 — EfficientNet Train/Val Split")

    df = pd.read_csv(index_csv)

    #Benign split
    benign_df    = df[df['malignant'] == False]
    benign_cases = benign_df.groupby('case_id')['slice_name'].count().reset_index()
    benign_cases = benign_cases.rename(columns = {'slice_name': 'patch_count'})
    benign_cases = benign_cases.sort_values('patch_count', ascending = False)

    # case_00188 forced to train — only 1 patch
    forced_train = ['case_00188']

    # Remaining 4 benign cases — highest patch count goes to val
    remaining_benign = benign_cases[~benign_cases['case_id'].isin(forced_train)]
    benign_val_case  = [remaining_benign.iloc[0]['case_id']]  # highest patch count
    benign_train_cases = forced_train + remaining_benign.iloc[1:]['case_id'].tolist()

    print(f"\nBenign split:")
    print(f"  Train cases : {benign_train_cases}")
    print(f"  Val cases   : {benign_val_case}")

    # Malignant split
    malignant_df    = df[df['malignant'] == True]
    malignant_cases = malignant_df['case_id'].unique()

    mal_train_cases, mal_val_cases = train_test_split(
        malignant_cases,
        test_size   = 0.20,
        random_state= seed
    )

    print(f"\nMalignant split:")
    print(f"  Train cases : {len(mal_train_cases)}")
    print(f"  Val cases   : {len(mal_val_cases)}")

    # Build dataframes
    all_train_cases = set(benign_train_cases) | set(mal_train_cases)
    all_val_cases   = set(benign_val_case)    | set(mal_val_cases)

    train_df = df[df['case_id'].isin(all_train_cases)].reset_index(drop = True)
    val_df   = df[df['case_id'].isin(all_val_cases)].reset_index(drop = True)

    # Save
    train_df.to_csv(splits_dir / "efficientnet_train.csv", index = False)
    val_df.to_csv(  splits_dir / "efficientnet_val.csv",   index = False)

    # Summary
    print(f"\nTrain set:")
    print(f"  Total patches  : {len(train_df)}")
    print(f"  Malignant      : {train_df['malignant'].sum()}")
    print(f"  Benign         : {(~train_df['malignant']).sum()}")
    print(f"  Benign cases   : {train_df[~train_df['malignant']]['case_id'].nunique()}")

    print(f"\nVal set:")
    print(f"  Total patches  : {len(val_df)}")
    print(f"  Malignant      : {val_df['malignant'].sum()}")
    print(f"  Benign         : {(~val_df['malignant']).sum()}")
    print(f"  Benign cases   : {val_df[~val_df['malignant']]['case_id'].nunique()}")

    print(f"\nSplits saved to: {splits_dir}")
    print("\nDone.")

if __name__ == "__main__":
    main()