# utils/create_subset.py
# Create a smaller, balanced subset CSV from the full metadata for quick experiments.

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import random

from pathlib import Path
import pandas as pd

def export_WSI_paths(subset_csv, output_dir):
    """Extract the FILE_LOCATION column from a subset CSV and export it as a simple CSV file."""
    
    subset_csv = Path(subset_csv)
    subset_df = pd.read_csv(subset_csv)

    # check whether it has a wsi path
    if "FILE_LOCATION" not in subset_df.columns:
        raise ValueError(f"{subset_csv} must contain a 'FILE_LOCATION' column")

    # Extract only the file paths
    wsi_paths_df = subset_df[["FILE_LOCATION"]].dropna().astype(str)

    # Optional sanity check
    if len(wsi_paths_df) != len(subset_df):
        print|("check if any slide IDs are missing in extraction of wsi paths")

    # Save next to input CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{subset_csv.stem}_paths.csv"
    wsi_paths_df.to_csv(output_path, index=False, header=True)

    print(f" Exported WSI paths to: {output_path}")
    return 


def create_balanced_subset(config):
    """Create a balanced subset of hypertrophy and non-hypertrophy slides from the validation or training set."""

    # Load config parameters
    subset_type = config["subset_creation"]["target_split"]  # "train" or "val"
    if subset_type == "train":
        subset_csv = config["datasets"]["train"]
    elif subset_type == "val":
        subset_csv = config["datasets"]["val"]
    else:
        raise ValueError("subset_creation.target_split must be 'train' or 'val'")

    # load dataset
    df = pd.read_csv(subset_csv)
    print(f"[INFO] Loaded {len(df)} entries from {subset_csv}")

    # check whether the format is correctly by having compounds and labels
    if "HasHypertrophy" not in df.columns or "COMPOUND_NAME" not in df.columns:
        raise ValueError(f"{subset_csv} must contain 'HasHypertrophy' and 'COMPOUND_NAME' columns")

    # seperate positives from negatives
    pos_df = df[df["HasHypertrophy"] == 1]
    neg_df = df[df["HasHypertrophy"] == 0]

    n_pos, n_neg = len(pos_df), len(neg_df)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Need both hypertrophy and non-hypertrophy slides in {subset_type} set")

    print(f"[INFO] Found {n_pos} hypertrophy and {n_neg} non-hypertrophy slides in {subset_type} set")

    # count number of unique compounds fior both classes (hypertrophy vs non-hypertrophy)
    n_pos_compounds = pos_df["COMPOUND_NAME"].nunique()
    n_neg_compounds = neg_df["COMPOUND_NAME"].nunique()
    print(f"[INFO] Hypertrophy compounds: {n_pos_compounds}, Non-hypertrophy compounds: {n_neg_compounds}")

    # get a fair distribution of number of slides per compound for negative class (non-hypertrophy)
    base_num_slides_per_compound = n_pos // n_neg_compounds
    remainder = n_pos % n_neg_compounds # how many slides are left 
    print(
        f"[INFO] We have {n_pos} negatives over {n_neg_compounds} compounds "
        f"\nleading to {base_num_slides_per_compound} negatives per compound and a remainder of {remainder}"
    )
    # randomly assign the compounds to cover up the remainder
    random.seed(42)
    unique_compounds = neg_df["COMPOUND_NAME"].unique()
    extra_compounds = set(random.sample(list(unique_compounds), k=min(remainder, len(unique_compounds))))
    
    # sample from each compound
    sampled_groups = []
    for compound, group in neg_df.groupby("COMPOUND_NAME"):
        n_to_take = base_num_slides_per_compound + (1 if compound in extra_compounds else 0)
        n_to_take = min(len(group), n_to_take)  # avoid oversampling
        sampled_groups.append(group.sample(n=n_to_take, random_state=42)) # creates all df and put them into the list
    # concatenate all in sampled_negatives_df in a df
    sampled_negatives_df = pd.concat(sampled_groups, ignore_index=True)

    # if still there are misbalance in the number of hypertrophy and non-hypterophy slides
    n_missing = n_pos - len(sampled_negatives_df)
    if n_missing > 0:
        print(f"[INFO] Need {n_missing} more negatives to perfectly match positives")
        remaining_neg_df = neg_df[~neg_df.index.isin(sampled_negatives_df.index)] # ensure no duplicates will occur
        if len(remaining_neg_df) == 0:
            print(" No negatives are left ")
            remaining_neg_df = neg_df
        additional_negatives_df = remaining_neg_df.sample(n=n_missing, replace=True, random_state=42)
        sampled_negatives_df = pd.concat([sampled_negatives_df, additional_negatives_df], ignore_index=True)

    # if we collected to many negative samples, use randomly sample over these again
    if len(sampled_negatives_df) > n_pos:
        sampled_negatives_df = sampled_negatives_df.sample(n=n_pos, random_state=42)

    print(f"[INFO] Sampled {len(sampled_negatives_df)} non-hypertrophy slides after balancing by compound")
    print("[INFO] Sampled non-hypertrophy compound distribution:")
    print(sampled_negatives_df["COMPOUND_NAME"].value_counts())

    # combine both positive and negative samples and finally shuffle them with frac=1 in sample
    balanced_df = (pd.concat([pos_df, sampled_negatives_df]).sample(frac=1, random_state=42).reset_index(drop=True))

    print(f"[INFO] Final balanced subset has {len(balanced_df)} slides ({len(pos_df)} pos / {len(sampled_negatives_df)} neg)")
    print(f"[INFO] Final compound distribution:\n{balanced_df['COMPOUND_NAME'].value_counts().sort_index()}")

    # saving the subset
    output_dir = Path(config["data"]["root"]) / "Subsets"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{subset_type}_balanced_subset.csv"
    balanced_df.to_csv(output_path, index=False)

    # export the WSI paths needed for the Slide2Vec
    export_WSI_paths(output_path, output_dir)

    # summary
    summary_path = output_dir / f"{subset_type}_balanced_summary.csv"
    summary_df = (
        balanced_df.groupby(["HasHypertrophy", "COMPOUND_NAME"])
        .size()
        .reset_index(name="count")
        .pivot(index="COMPOUND_NAME", columns="HasHypertrophy", values="count")
        .fillna(0)
        .rename(columns={0: "Non-Hypertrophy", 1: "Hypertrophy"})
        .astype(int)
    )
    summary_df.to_csv(summary_path)
    print(f"[INFO] Saved compound summary: {summary_path}")

    print(f" Saved subset: {output_path}")
    return 


if __name__ == "__main__":
    with open("configs/configs.yaml", "r") as f:
        config = yaml.safe_load(f)
    create_balanced_subset(config)
