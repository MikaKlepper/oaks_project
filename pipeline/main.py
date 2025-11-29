# main.py

import subprocess
import sys
import logging
from pathlib import Path
from types import SimpleNamespace

from argparser import get_args
from utils.config_loader import load_merged_config
from logger import setup_logger
from omegaconf import OmegaConf


def write_config(cfg, out_dir: Path):
    """
    Write cfg to <out_dir>/config.yaml and return that path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "config.yaml"
    OmegaConf.save(cfg, out_file)
    return out_file


def run_subprocess(script: str, cfg_path: Path):
    """
    Run train.py or eval.py in a fresh Python process.
    """
    cmd = [sys.executable, script, "--config", str(cfg_path)]
    print(f"[MAIN] Launching: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    args = get_args()

    # 1) Load merged config ONCE with CLI args just to get experiment_root
    cfg_for_root = load_merged_config(args.config, args)
    exp_root = Path(cfg_for_root.experiment_root)

    setup_logger(exp_root)
    logging.info("========== MAIN ==========")
    logging.info(f"[Main] Requested stage: {args.stage}")

    # ---------------- TRAIN ONLY ----------------
    if args.stage == "train":
        args_train = SimpleNamespace(**vars(args))
        args_train.stage = "train"

        # This call builds dirs for SPLIT=train and applies CLI overrides
        cfg_train = load_merged_config(args.config, args_train)
        train_cfg_path = write_config(cfg_train, exp_root / "train")
        run_subprocess("train.py", train_cfg_path)
        return

    # ---------------- EVAL ONLY ----------------
    if args.stage == "eval":
        args_eval = SimpleNamespace(**vars(args))
        args_eval.stage = "eval"

        # This call builds dirs for SPLIT=val and applies CLI overrides
        cfg_eval = load_merged_config(args.config, args_eval)
        eval_cfg_path = write_config(cfg_eval, exp_root / "eval")
        run_subprocess("eval.py", eval_cfg_path)
        return

    # ---------------- TRAIN → EVAL ----------------
    if args.stage == "all":
        logging.info("[Main] Running TRAIN → EVAL (fully isolated)")

        # TRAIN CONFIG (split=train)
        args_train = SimpleNamespace(**vars(args))
        args_train.stage = "train"
        cfg_train = load_merged_config(args.config, args_train)
        train_cfg_path = write_config(cfg_train, exp_root / "train")
        run_subprocess("train.py", train_cfg_path)

        # EVAL CONFIG (split=val)
        args_eval = SimpleNamespace(**vars(args))
        args_eval.stage = "eval"
        cfg_eval = load_merged_config(args.config, args_eval)
        eval_cfg_path = write_config(cfg_eval, exp_root / "eval")
        run_subprocess("eval.py", eval_cfg_path)

        return

    # ---------------- UNKNOWN STAGE ----------------
    raise ValueError(f"Unknown stage={args.stage}")


if __name__ == "__main__":
    main()
