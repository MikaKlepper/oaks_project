import pandas as pd
from pathlib import Path
import yaml

def load_yaml_config(path):
    """
    Load configuration file to the corresponding path where the config file is located
    """
    with open(path, "r") as f:
       return yaml.safe_load(f)

def load_ids(csv_path):
    """
    Load ids from the corresponding csv_paths for training, val, test csv
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"File not found in {csv_path}")
        return set()
    # if there is csv_path existing
    df= pd.read_csv(csv_path)
    ids= set(df["slide_id"].astype(str))
    print(f"Loaded {len(ids)} IDs from {csv_path.name}")
    return ids

def load_feature_ids(feature_dir):
    """
    Extract Slide IDs from .pt files.
    Ex: 41124.pt will be 41124
    """
    feature_dir= Path(feature_dir)
    if not feature_dir:
        print(f"Feature directory not found at {feature_dir}")
        return set()
    feature_ids = set()
    for file in feature_dir.glob("*.pt"):
        feature_ids.add(file.stem)
    print(f"Found {len(feature_ids)} .pt features in {feature_dir}")
    return feature_ids

def resolve_feature_paths(cfg):
    encoder = cfg["features"]["encoder"]
    slide_dir = cfg["features"]["slide_dir"].format(encoder=encoder)
    animal_dir = cfg["features"]["animal_dir"].format(encoder=encoder)
    return slide_dir, animal_dir


def check_subset_consistency(cfg_path):
    cfg = load_yaml_config(cfg_path)
    datasets = cfg["datasets"]

    # train all ids, creates all sets
    train_ids = load_ids(datasets["train"])
    val_ids   = load_ids(datasets["val"])
    test_ids  = load_ids(datasets["test"])

    # Step 1: Check global split consistency
    print("\n SPLIT check if any consistency is executed during splitting the data into train, test and val")
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    overlap_train_val = train_ids & val_ids
    overlap_train_test = train_ids & test_ids
    overlap_val_test = val_ids & test_ids

    if overlap_train_val or overlap_train_test or overlap_val_test:
        print(f" Train/Val overlap: {len(overlap_train_val)}")
        print(f" Train/Test overlap: {len(overlap_train_test)}")
        print(f" Val/Test overlap: {len(overlap_val_test)}")
    else:
        print(" !! No overlaps between train/val/test sets !! ")

    # Step 2: Check subset consistency + feature consistency
    if datasets["use_subset"]:
        subset_path = Path(datasets["subset"])
        subset_type = datasets["subset_type"]
        subset_ids  = load_ids(subset_path)

        slide_dir, animal_dir = resolve_feature_paths(cfg)
        features_ids = load_feature_ids(slide_dir)


        # Correct mapping
        if subset_type == "val":
            missing_ids = subset_ids - val_ids
        elif subset_type == "train":
            missing_ids = subset_ids - train_ids
        elif subset_type == "test":
            missing_ids = subset_ids - test_ids  
        else:
            print(f" Unknown subset type '{subset_type}'. Expected 'train','test','val'.")
            missing_ids = set()

        # Correct feature matching
        missing_feature_ids = subset_ids - features_ids 

        # Report
        print("\n--- Subset Consistency Check ---")
        print(f"Subset type: {subset_type}")
        print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)} | Subset: {len(subset_ids)}")

        if missing_ids:
            print(f" {len(missing_ids)} subset IDs not found in {subset_type} set!")
            print("Missing examples:", list(missing_ids)[:10], "...")
        else:
            print(f" All subset IDs are present in the {subset_type} set!")

        if missing_feature_ids:
            print(f" {len(missing_feature_ids)} subset slides missing feature files!")
            print("Missing features:", list(missing_feature_ids)[:10], "...")
        else:
            print(" All required feature files are present.")

    return train_ids, val_ids, test_ids, subset_ids, features_ids



if __name__ == "__main__":
    cfg_path = "/data/temporary/mika/repos/oaks_project/pipeline/configs/configs.yaml"
    check_subset_consistency(cfg_path)


