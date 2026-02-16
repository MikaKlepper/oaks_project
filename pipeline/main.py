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


def write_config(cfg, out_dir: Path) -> Path:
    """
    Write the given config to a file in the specified directory.

    Args:
        cfg: The config to write
        out_dir: The directory to write the config to

    Returns:
        The path to the written config file
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "config.yaml"
    OmegaConf.save(cfg, path)
    return path


def run_stage(stage: str, args, exp_root: Path):
    """
    Run the specified stage of the pipeline.

    Args:
        stage: The stage to run (train, eval, test)
        args: The parsed command line arguments
        exp_root: The root directory of the experiment

    Runs the specified stage by generating a stage-specific config, writing it to a file, and then invoking the corresponding stage's script with the generated config.

    Note: This function will block until the specified stage has finished running. If the stage fails, this function will raise a CalledProcessError.
    """
    stage_args = SimpleNamespace(**{**vars(args), "stage": stage})
    cfg = load_merged_config(args.config, stage_args)

    cfg.stage = stage
    cfg_path = write_config(cfg, exp_root / stage)

    script = "train.py" if stage == "train" else "eval.py"
    cmd = [sys.executable, script, "--config", str(cfg_path)]

    logging.info(f"[MAIN] Running {stage.upper()}")
    subprocess.run(cmd, check=True)


def main():
    args = get_args()

    # Determine experiment root once
    cfg = load_merged_config(args.config, args)
    exp_root = Path(cfg.experiment_root)

    setup_logger(exp_root)
    logging.info("========== MAIN ==========")
    logging.info(f"[MAIN] Requested stage: {args.stage}")

    # Decide which stages to run
    if args.stage == "all":
        # stages = ["train", "eval", "test"]
        stages = ["train", "eval"]
    else:
        stages = [args.stage]

    valid = {"train", "eval", "test"}
    for stage in stages:
        if stage not in valid:
            raise ValueError(f"Unknown stage: {stage}")
        run_stage(stage, args, exp_root)


if __name__ == "__main__":
    main()
