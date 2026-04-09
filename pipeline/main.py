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


TEST_ONLY_DATASETS = {"ucb"}
REAL_STAGES = {"train", "eval", "test"}


def resolve_stages(requested_stage: str, dataset: str, calibration_enabled: bool = False):
    if requested_stage == "all":
        if dataset in TEST_ONLY_DATASETS:
            return ["train", "test"] if calibration_enabled else ["test"]
        return ["train", "eval"] # test will be done in benchmark.py, so we skip it here
    return [requested_stage]


def run_stage(stage: str, args, exp_root: Path):
    assert stage in REAL_STAGES

    # IMPORTANT: copy args first, then override stage
    stage_args = SimpleNamespace(**vars(args))
    stage_args.stage = stage

    cfg = load_merged_config(args.config, stage_args)

    cfg.stage = stage
    out_dir = exp_root / stage
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = out_dir / "config.yaml"
    OmegaConf.save(cfg, cfg_path)

    script = "train.py" if stage == "train" else "eval.py"
    cmd = [sys.executable, script, "--config", str(cfg_path)]

    logging.info(f"[MAIN] Running {stage.upper()}")
    subprocess.run(cmd, check=True)


def main():
    args = get_args()
    assert args.stage in REAL_STAGES | {"all"}

    # preview the stages that would be run for the given dataset and stage argument combination
    preview_stage = "test" if args.stage == "all" else args.stage
    preview_args = SimpleNamespace(**vars(args))
    preview_args.stage = preview_stage

    cfg_preview = load_merged_config(args.config, preview_args)
    dataset = cfg_preview.datasets.name
    exp_root = Path(cfg_preview.experiment_root)

    setup_logger(exp_root)

    stages = resolve_stages(
        args.stage,
        dataset,
        calibration_enabled=bool(cfg_preview.calibration.enabled),
    )

    logging.info("========== MAIN ==========")
    logging.info(f"[MAIN] Dataset: {dataset}")
    logging.info(f"[MAIN] Stages to run: {stages}")

    for stage in stages:
        run_stage(stage, args, exp_root)


if __name__ == "__main__":
    main()
