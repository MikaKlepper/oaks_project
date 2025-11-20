from argparser import get_args
from utils.config_loader import load_merged_config
from data.prepare_dataset import prepare_dataset_inputs
from data.features_per_animal import group_features_by_animal
from data.create_datasets import AnimalDataset
import pprint

def main():
    args = get_args()

    # 1) Load merged config (resolved)
    cfg, _ = load_merged_config(args.config, args)

    print("\n===== FINAL CONFIG =====")
    pprint.pp(cfg)

    # 2) First: create animal-level features
    print("\n===== BUILDING ANIMAL FEATURES =====")
    prepared = prepare_dataset_inputs(cfg)   # <-- prepares CSVs + dirs

    summary = group_features_by_animal(prepared)
    print("\n===== ANIMAL FEATURE SUMMARY =====")
    pprint.pp(summary)

    # 3) Now dataset loading is safe
    print("\n===== LOADING ANIMAL DATASET =====")
    ds = AnimalDataset(prepared)

    print(f"[INFO] Dataset size: {len(ds)}")
    feats, label = ds[0]
    print("[INFO] First sample:", feats.shape, label)


if __name__ == "__main__":
    main()
