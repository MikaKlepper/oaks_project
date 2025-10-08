# main.py
# Orchestrates the whole pipeline: splitting -> training -> evaluation.

import argparse
import yaml
from pathlib import Path

from splitting import main as run_splitting
from train import run_training
from test_evaluate import run_evaluation
from argparser import get_args
from utils.create_subset import create_balanced_subset


def main(config_path, do_split=False, do_train=True, do_eval=True, do_subset=False):
    # load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"[INFO] Loaded config from {config_path}")

    # splitting only if requested or if CSVs don't exist
    # Use the paths defined in YAML instead of hardcoding train/val/test
    train_csv = Path(config["datasets"]["train"])
    val_csv   = Path(config["datasets"]["val"])
    test_csv  = Path(config["datasets"]["test"])

    if do_split or not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
        print("[INFO] Running data splitting...")
        run_splitting(config_path)
        return  # exit after splitting
    else:
        print("[INFO] Found existing splits, skipping splitting.")


    # creating subset if requested
    if do_subset or config.get("subset_creation", {}).get("enabled", False):
        create_balanced_subset(config)
        return
    
    # training
    if do_train:
        print("[INFO] Starting training...")
        run_training(config)

    # evaluation
    if do_eval:
        print("[INFO] Starting evaluation...")
        run_evaluation(config, model_path="outputs/slide_classifier.pth")


if __name__ == "__main__":
    args = get_args()

    main(
        config_path=args.config,
        do_split=args.split,
        do_train=not args.no_train,
        do_eval=not args.no_eval,
        do_subset=args.subset
    )
