from argparser import get_args
from utils.config_loader import load_merged_config
from data.prepare_dataset import prepare_dataset_inputs
from data.features_per_animal import group_features_by_animal
from data.dataset_check import check_subset_consistency
from data.create_datasets import ToxicologyDataset
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

    if prepared["data"]["features_type"] == "animal":
        summary = group_features_by_animal(prepared)
        print("\n===== ANIMAL FEATURE SUMMARY =====")
        #pprint.pp(summary)

    # Check subset consistency 
    print("\n===== CHECKING SUBSET CONSISTENCY =====")
    check_subset_consistency(prepared)

    # 3) Now dataset loading is safe
    print("\n===== LOADING DATASET =====")
    ds = ToxicologyDataset(prepared)

    print(f"[INFO] Dataset size: {len(ds)}")
    feats, label = ds[0]
    print("[INFO] First sample:", feats.shape, label)


if __name__ == "__main__":
    main()
