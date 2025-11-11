import pandas as pd
from pathlib import Path
import torch
import yaml

def load_yaml_config(path):
    """Load configuration YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def resolve_feature_paths(cfg):
    encoder = cfg["features"]["encoder"]
    slide_dir = cfg["features"]["slide_dir"].format(encoder=encoder)
    animal_dir = cfg["features"]["animal_dir"].format(encoder=encoder)
    return Path(slide_dir), Path(animal_dir)

def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    if "slide_id" not in df.columns or "subject_UID" not in df.columns:
        raise ValueError(f"CSV must contain slide_id and subject_UID columns! File: {csv_path}")
    return df

def group_features_by_animal(cfg_path):
    cfg = load_yaml_config(cfg_path)
    datasets = cfg["datasets"]

    slide_dir, animal_dir = resolve_feature_paths(cfg)
    animal_dir.mkdir(parents=True, exist_ok=True)

    if datasets["use_subset"]:
        subset_type = datasets["subset_type"]
        print(f"Using only the {subset_type} subset")
        subset_df = load_csv(datasets["subset"])
        splits = {subset_type: subset_df}
    else:
        print("Using full train/val/test splits")
        splits = {
            "train": load_csv(datasets["train"]),
            "val":   load_csv(datasets["val"]),
            "test":  load_csv(datasets["test"]),
        }
    
    missing_files = []
    summary = {}

    for split_name, df in splits.items():
        print(f"Processing split: {split_name}, with {len(df)} slide_ids")

        grouped = df.groupby("subject_UID")["slide_id"].apply(list)
        split_info = {
                "num_animals": len(grouped),
                "animals": [],
            }

        for subject_id, slide_ids in grouped.items():
            out_file = animal_dir / f"{subject_id}.pt"

            split_info["animals"].append(
                {"subject": subject_id, "num_slides": len(slide_ids)}
            )

            if out_file.exists():
                continue

            tensors = []
            for slide_id in slide_ids:
                feat_path = slide_dir / f"{slide_id}.pt"
                if feat_path.exists():
                    tensors.append(torch.load(feat_path, map_location="cpu"))
                else:
                    missing_files.append(str(feat_path))

            if tensors:
                combined = torch.cat(tensors, dim=0)
                torch.save(combined, out_file)
            else:
                print(f"No valid features for animal {subject_id}")

        summary[split_name] = split_info
        print(f"FINISHED ANIMAL FEATURE AGGREGATION FOR {split_name}")

    if missing_files:
        print("\nMissing feature files:")
        print(f"  Total missing: {len(missing_files)}")
        for m in missing_files[:10]:
            print("   -", m)
        print("...")
    else:
        print("\nNo missing feature slides detected.")

    print("\nAnimal-level features updated successfully.")
    return summary



if __name__ == "__main__":
    cfg_path = "/data/temporary/mika/repos/oaks_project/pipeline/configs/configs.yaml"
    summary = group_features_by_animal(cfg_path)

    print("\n===== SUMMARY REPORT =====")
    print("\n===== Animals with multiple slides =====")
    for split, info in summary.items():
        print(f"\nSplit: {split}")
        multi = [a for a in info["animals"] if a["num_slides"] > 1]
        
        if not multi:
            print("  No animals with multiple slides ")
        else:
            print(f"  {len(multi)} animals have multiple slides:")
            for a in multi[:10]:  # show first 10
                print(f"    - subject {a['subject']} ({a['num_slides']} slides)")