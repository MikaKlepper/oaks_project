# utils/create_subset.py
# Create a smaller, balanced subset CSV from the full metadata for quick experiments.

import random
from pathlib import Path

import pandas as pd
import yaml


def _id_series(df: pd.DataFrame) -> pd.Series:
    for column in ("subject_organ_UID", "animal_number", "slide_id", "slide_filename"):
        if column in df.columns:
            return df[column].astype(str)
    raise ValueError("Subset CSV must contain a supported ID column.")


def create_seeded_holdout_subsets(
    source_csv,
    *,
    sample_size: int,
    seed: int,
    train_csv,
    test_csv,
    label_column: str | None = None,
    positive_value=None,
    available_ids: set[str] | None = None,
):
    source_csv = Path(source_csv)
    train_csv = Path(train_csv)
    test_csv = Path(test_csv)

    if train_csv.exists() and test_csv.exists():
        return train_csv, test_csv

    df = pd.read_csv(source_csv)
    if available_ids is not None:
        df = df[_id_series(df).isin(available_ids)].reset_index(drop=True)
    if sample_size <= 0 or sample_size >= len(df):
        raise ValueError(
            f"sample_size must be between 1 and {len(df) - 1}, got {sample_size}"
        )

    train_csv.parent.mkdir(parents=True, exist_ok=True)

    subset_df = None
    if label_column and label_column in df.columns and sample_size > 1:
        labels = _coerce_binary_labels(df[label_column], positive_value)
        pos_df = df[labels == 1]
        neg_df = df[labels == 0]
        if not pos_df.empty and not neg_df.empty:
            rng = random.Random(seed)
            pos_rows = pos_df.sample(frac=1, random_state=seed).to_dict("records")
            neg_rows = neg_df.sample(frac=1, random_state=seed).to_dict("records")
            pos_i = neg_i = 0
            picked = []
            while len(picked) < sample_size and (pos_i < len(pos_rows) or neg_i < len(neg_rows)):
                take_pos = len(picked) % 2 == 0
                if take_pos and pos_i < len(pos_rows):
                    picked.append(pos_rows[pos_i])
                    pos_i += 1
                elif not take_pos and neg_i < len(neg_rows):
                    picked.append(neg_rows[neg_i])
                    neg_i += 1
                elif pos_i < len(pos_rows):
                    picked.append(pos_rows[pos_i])
                    pos_i += 1
                elif neg_i < len(neg_rows):
                    picked.append(neg_rows[neg_i])
                    neg_i += 1

            subset_df = pd.DataFrame(picked)
            subset_df = subset_df.sample(frac=1, random_state=seed).reset_index(drop=True)
            if len(subset_df) < sample_size:
                print(
                    "[WARN] Could not reach requested sample size with balanced alternation; "
                    f"available pos={len(pos_df)} neg={len(neg_df)}."
                )

    if subset_df is None:
        subset_df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    subset_ids = set(_id_series(subset_df))
    holdout_df = df[~_id_series(df).isin(subset_ids)].reset_index(drop=True)

    subset_df.to_csv(train_csv, index=False)
    holdout_df.to_csv(test_csv, index=False)
    return train_csv, test_csv


def _coerce_binary_labels(series: pd.Series, positive_value=None) -> pd.Series:
    positive_bool = bool(positive_value) if positive_value is not None else True
    return series.fillna(False).astype(bool).eq(positive_bool).astype(int)

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
        print("check if any slide IDs are missing in extraction of wsi paths")

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
    seed = config.get("runtime", {}).get("seed", 42)
    random.seed(seed)
    unique_compounds = neg_df["COMPOUND_NAME"].unique()
    extra_compounds = set(random.sample(list(unique_compounds), k=min(remainder, len(unique_compounds))))
    
    # sample from each compound
    sampled_groups = []
    for compound, group in neg_df.groupby("COMPOUND_NAME"):
        n_to_take = base_num_slides_per_compound + (1 if compound in extra_compounds else 0)
        n_to_take = min(len(group), n_to_take)  # avoid oversampling
        sampled_groups.append(group.sample(n=n_to_take, random_state=seed)) # creates all df and put them into the list
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
        additional_negatives_df = remaining_neg_df.sample(n=n_missing, replace=True, random_state=seed)
        sampled_negatives_df = pd.concat([sampled_negatives_df, additional_negatives_df], ignore_index=True)

    # if we collected to many negative samples, use randomly sample over these again
    if len(sampled_negatives_df) > n_pos:
        sampled_negatives_df = sampled_negatives_df.sample(n=n_pos, random_state=seed)

    print(f"[INFO] Sampled {len(sampled_negatives_df)} non-hypertrophy slides after balancing by compound")
    print("[INFO] Sampled non-hypertrophy compound distribution:")
    print(sampled_negatives_df["COMPOUND_NAME"].value_counts())

    # combine both positive and negative samples and finally shuffle them with frac=1 in sample
    balanced_df = (pd.concat([pos_df, sampled_negatives_df]).sample(frac=1, random_state=seed).reset_index(drop=True))

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
