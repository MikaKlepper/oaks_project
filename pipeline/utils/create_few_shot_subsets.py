import pandas as pd
import random
from pathlib import Path
import yaml


# helper functions

DEFAULT_SEED = 42

# 1: load yaml
def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# 2: export WSI paths

def export_WSI_paths(subset_csv, output_dir):
    subset_csv = Path(subset_csv)
    df = pd.read_csv(subset_csv)

    if "FILE_LOCATION" not in df.columns:
        raise ValueError(f"{subset_csv} must contain FILE_LOCATION")

    # rename and fix base paths
    df = df.rename(columns={"FILE_LOCATION": "wsi_path"})
    df["wsi_path"] = df["wsi_path"].str.replace("RBS_PA_CPGARCHIVE", "pa_cpgarchive", regex=False)

    # save each k separately with all paths
    output_path = output_dir / f"{subset_csv.stem}_paths.csv"
    df[["wsi_path"]].dropna().astype(str).to_csv(output_path, index=False)
    print(f"Exported WSI paths -> {output_path}")

    # only create the scratch version for the final (k=100) subset
    if subset_csv.stem.endswith("k100"):
        df_scratch = df.copy()
        df_scratch["wsi_path"] = df_scratch["wsi_path"].apply(
            lambda p: f"/scratch_mikaklepper_train/wsis/liver/train/{Path(p).name}"
        )

        scratch_path = Path(
            "/data/temporary/mika/repos/oaks_project/splitting_data/FewShotCompoundBalanced/train_balanced_fewshot_subset_paths_scratch.csv"
        )
        df_scratch[["wsi_path"]].to_csv(scratch_path, index=False)
        print(f" Exported scratch WSI paths -> {scratch_path}")


# 3: per-compound compute how many samples to take
def per_compound_targets(compounds, k, seed=DEFAULT_SEED):
    n = len(compounds)
    if n == 0:
        raise ValueError("No compounds found in dataset.")
    base, remainder = divmod(k, n)
    random.seed(seed)
    ordered = random.sample(sorted(compounds), len(compounds))
    targets = {c: base for c in ordered}
    for c in ordered[:remainder]:
        targets[c] += 1
    return targets


# 4: grow cumulative subset up to MAX k 
def grow_to_k(df_full, used_df, k, seed=DEFAULT_SEED):
    df_unique = df_full.drop_duplicates(subset=["subject_UID"])
    compounds = df_unique["COMPOUND_NAME"].unique()
    targets = per_compound_targets(compounds, k, seed)

    current_counts = used_df["COMPOUND_NAME"].value_counts().to_dict()
    new_samples = []

    for comp in compounds:
        group = df_unique[df_unique["COMPOUND_NAME"] == comp]
        current = current_counts.get(comp, 0)
        needed = max(0, targets[comp] - current)

        if needed > 0:
            available = group[~group["subject_UID"].isin(used_df["subject_UID"])]
            take = min(len(available), needed)
            if take > 0:
                new_samples.append(available.sample(n=take, random_state=seed))

    # Combine new samples
    if new_samples:
        added = pd.concat(new_samples, ignore_index=True)
        used_df = pd.concat([used_df, added], ignore_index=True).drop_duplicates(subset=["subject_UID"])

    # fill remaining slots if total < k (especially helps positives)
    if len(used_df) < k:
        remaining = df_unique[~df_unique["subject_UID"].isin(used_df["subject_UID"])]
        missing = k - len(used_df)
        if len(remaining) > 0:
            extra = remaining.sample(n=min(missing, len(remaining)), random_state=seed)
            used_df = pd.concat([used_df, extra], ignore_index=True)

    # shuffle the whole thing
    used_df = used_df.sample(n=min(k, len(used_df)), random_state=seed).reset_index(drop=True)

    # check if it works and k is reached
    if len(used_df) != k:
        print(f" Warning: Could not reach full k={k}. Got only {len(used_df)} samples. ")
    else:
        print(f" Verified: subset reached exactly k={k} samples.")
    return used_df


# 5: main few-shot subset creation
def create_fewshot_compound_balanced(config_path, ks=[5, 10, 20, 40, 80, 100]):
    config = load_yaml_config(config_path)
    seed = config.get("runtime", {}).get("seed", DEFAULT_SEED)
    train_csv = config["datasets"]["train"]
    df = pd.read_csv(train_csv)

    required_cols = ["slide_id", "subject_UID", "HasHypertrophy", "COMPOUND_NAME", "FILE_LOCATION"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    pos_df = df[df["HasHypertrophy"] == 1]
    neg_df = df[df["HasHypertrophy"] == 0]

    out_dir = Path(config["data"]["root"]) / "FewShotCompoundBalanced"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Found {len(pos_df)} positive and {len(neg_df)} negative samples.")
    print(f"[INFO] Positive compounds: {pos_df['COMPOUND_NAME'].nunique()}, Negative compounds: {neg_df['COMPOUND_NAME'].nunique()}")

    used_pos = pd.DataFrame(columns=df.columns)
    used_neg = pd.DataFrame(columns=df.columns)

    for k in ks:
        print(f"\n--- Creating cumulative few-shot subset for k={k} ---")

        used_pos = grow_to_k(pos_df, used_pos, k, seed=seed)
        used_neg = grow_to_k(neg_df, used_neg, k, seed=seed)

        combined = pd.concat([used_pos, used_neg]).sample(frac=1, random_state=seed)

        output_csv = out_dir / f"train_fewshot_k{k}.csv"
        combined.to_csv(output_csv, index=False)
        print(f"Saved cumulative few-shot subset -> {output_csv}")

        export_WSI_paths(output_csv, out_dir)

        # Per-compound summary (sorted + pretty)
        summary_path = out_dir / f"train_fewshot_k{k}_summary.csv"
        summary_df = (
            combined.groupby(["HasHypertrophy", "COMPOUND_NAME"])
            .size()
            .reset_index(name="count")
            .pivot(index="COMPOUND_NAME", columns="HasHypertrophy", values="count")
            .fillna(0)
            .rename(columns={0: "Non-Hypertrophy", 1: "Hypertrophy"})
            .astype(int)
            .sort_index()
        )
        summary_df.to_csv(summary_path)
        print(f"Saved summary -> {summary_path}")
        print(summary_df)

    print("\n All cumulative, compound-prioritized few-shot subsets created successfully.")


# Run
if __name__ == "__main__":
    cfg_path = "/data/temporary/mika/repos/oaks_project/pipeline/configs/configs.yaml"
    create_fewshot_compound_balanced(cfg_path, ks=[5, 10, 20, 40, 80, 100])
