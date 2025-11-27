# pipeline/main.py

import logging
from argparser import get_args
from utils.config_loader import load_merged_config
from logger import setup_logger
from train import run_train
from eval import run_eval
from copy import deepcopy


def main():
    # ------------- 1) Parse CLI Arguments --------------------
    args = get_args()

    # ------------- 2) Load merged config (config YAML + CLI overrides) -----
    cfg, _ = load_merged_config(args.config, args)

    # ------------- 3) Setup logging --------------------------
    setup_logger(cfg.experiment_root)
    logging.info("========== MAIN STAGE ==========")
    logging.info(f"[Main] Stage selected: {cfg.datasets.split}")

    # ------------- 4) Execute stage --------------------------
    stage = cfg.datasets.split.lower()

    if stage == "train":
        logging.info("[Main] Running TRAIN stage")
        run_train(cfg)

    elif stage == "val":
        logging.info("[Main] Running EVAL stage")
        run_eval(cfg)

    elif stage == "all":
        logging.info("[Main] Running: TRAIN → EVAL")

        # override split for TRAIN
        cfg_train = deepcopy(cfg)
        cfg_train.datasets.split = "train"
        run_train(cfg_train)

        # override split for EVAL (full validation)
        cfg_eval = deepcopy(cfg)
        cfg_eval.datasets.split = "val"
        run_eval(cfg_eval)

    else:
        raise ValueError(
            f"[Main] Unknown --stage '{stage}'. "
            f"Valid options: train, eval, all."
        )

    logging.info("========== MAIN DONE ==========")


if __name__ == "__main__":
    main()
