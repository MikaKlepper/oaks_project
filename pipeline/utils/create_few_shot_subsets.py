import pandas as pd
import random
from pathlib import Path
import yaml


def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def export_WSI_paths(subset_csv, output_dir):
    """Extract FILE_LOCATION and save as *_paths.csv"""
    subset_csv = Path(subset_csv)
    df = pd.read_csv(subset_csv)

    if "FILE_LOCATION" not in df.columns:
        raise ValueError(f"{subset_csv} must contain FILE_LOCATION")

    wsi_paths = df[["FILE_LOCATION"]].dropna().astype(str)

    output_path = output_dir / f"{subset_csv.stem}_paths.csv"
    wsi_paths.to_csv(output_path, index=False)
    print(f" Exported WSI paths --> {output_path}")


def sample_class_with_compound_distribution(df, k, seed=42):
    """
    Sample k rows from df balancing by compound distribution.
    - proportional sampling
    - robust to small compounds
    """
    random.seed(seed)
    compounds = df["COMPOUND_NAME"].unique()
    n_compounds = len(compounds)

    # Case 1: very small k → pick one per compound
    if k <= n_compounds:
        selected_compounds = random.sample(list(compounds), k)
        sampled = []
        for comp in selected_compounds:
            group = df[df["COMPOUND_NAME"] == comp]
            sampled.append(group.sample(n=1, random_state=seed))
        return pd.concat(sampled)

    # Case 2: proportional allocation
    base = k // n_compounds
    remainder = k % n_compounds

    extra_compounds = set(random.sample(list(compounds), remainder))

    sampled = []
    for comp in compounds:
        group = df[df["COMPOUND_NAME"] == comp]
        n_take = base + (1 if comp in extra_compounds else 0)
        n_take = min(len(group), n_take)
        sampled.append(group.sample(n=n_take, random_state=seed))

    result = pd.concat(sampled)

    # top up if undersampled
    if len(result) < k:
        missing = k - len(result)
        remaining = df[~df.index.isin(result.index)]
        extra = remaining.sample(n=missing, replace=True, random_state=seed)
        result = pd.concat([result, extra])

    # if overshoot
    if len(result) > k:
        result = result.sample(n=k, random_state=seed)

    return result.reset_index(drop=True)


def create_fewshot_compound_subsets(config_path, ks=[5, 10, 20, 50, 100]):
    # load dataset
    config = load_yaml_config(config_path)
    train_csv = config["datasets"]["train"]
    df = pd.read_csv(train_csv)

    required_cols = ["slide_id", "subject_UID", "HasHypertrophy", "COMPOUND_NAME", "FILE_LOCATION"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    pos_df = df[df["HasHypertrophy"] == 1]
    neg_df = df[df["HasHypertrophy"] == 0]

    out_dir = Path(config["data"]["root"]) / "FewShotCompoundAware"
    out_dir.mkdir(parents=True, exist_ok=True)

    for k in ks:
        print(f"\n=== Creating k={k} few-shot compound-aware subset ===")

        few_pos = sample_class_with_compound_distribution(pos_df, k)
        few_neg = sample_class_with_compound_distribution(neg_df, k)

        combined = pd.concat([few_pos, few_neg]).sample(frac=1, random_state=42)
        output_csv = out_dir / f"train_fewshot_compound_{k}.csv"
        combined.to_csv(output_csv, index=False)

        print(f"Saved few-shot CSV → {output_csv}")
        print(f"Positives={len(few_pos)}, Negatives={len(few_neg)}")

        # export WSI paths
        export_WSI_paths(output_csv, out_dir)

        # save summary distribution
        summary_path = out_dir / f"train_fewshot_compound_{k}_summary.csv"
        summary_df = (
            combined.groupby(["HasHypertrophy", "COMPOUND_NAME"])
            .size()
            .reset_index(name="count")
            .pivot(index="COMPOUND_NAME", columns="HasHypertrophy", values="count")
            .fillna(0)
            .rename(columns={0: "Non-Hypertrophy", 1: "Hypertrophy"})
            .astype(int)
        )
        summary_df.to_csv(summary_path)

        print(f"Saved summary -> {summary_path}")

    print("\n All few-shot compound-aware subsets were created successfully !!")


if __name__ == "__main__":
    cfg_path = "/data/temporary/mika/repos/oaks_project/pipeline/configs/configs.yaml"
    create_fewshot_compound_subsets(cfg_path)
