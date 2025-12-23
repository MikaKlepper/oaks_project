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


def run_stage(stage: str, args, exp_root: Path):
    """
    Run a single pipeline stage (train / eval / test) in isolation.
    """
    stage_args = SimpleNamespace(**vars(args))
    stage_args.stage = stage

    cfg = load_merged_config(args.config, stage_args)
    cfg_path = write_config(cfg, exp_root / stage)

    script = "train.py" if stage == "train" else "eval.py"
    run_subprocess(script, cfg_path)


def main():
    args = get_args()

    # Load once to determine experiment root
    cfg_for_root = load_merged_config(args.config, args)
    exp_root = Path(cfg_for_root.experiment_root)

    setup_logger(exp_root)
    logging.info("========== MAIN ==========")
    logging.info(f"[Main] Requested stage: {args.stage}")

    # ---------------- TRAIN ONLY ----------------
    if args.stage == "train":
        run_stage("train", args, exp_root)
        return

    # ---------------- EVAL ONLY (validation) ----------------
    if args.stage == "eval":
        run_stage("eval", args, exp_root)
        return

    # ---------------- TEST ONLY ----------------
    if args.stage == "test":
        run_stage("test", args, exp_root)
        return

    # ---------------- TRAIN → EVAL → TEST ----------------
    if args.stage == "all":
        logging.info("[Main] Running TRAIN → EVAL → TEST (fully isolated)")
        run_stage("train", args, exp_root)
        run_stage("eval", args, exp_root)
        run_stage("test", args, exp_root)
        return

    # ---------------- UNKNOWN STAGE ----------------
    raise ValueError(f"Unknown stage={args.stage}")


if __name__ == "__main__":
    main()
