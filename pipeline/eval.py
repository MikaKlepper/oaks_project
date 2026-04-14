# pipeline/eval.py

import gc
import logging
from pathlib import Path

import numpy as np
import torch

from argparser import get_args
from utils.config_loader import load_merged_config
from data.prepare_dataset import prepare_dataset_inputs
from data.create_datasets import ToxicologyDataset
from data.dataset_check import check_subset_consistency

from data.create_dataset_MIL import ToxicologyMILDataset
from data.collate_MIL import collate_mil

from probes import build_probe, TorchProbe, default_probe_path
from metrics import compute_and_log_metrics
from logger import setup_logger
from log_benchmark import log_benchmark
from eval_analysis import run_misclassification_analysis
from utils.experiment_registry import append_experiment_row


def build_train_experiment_root(cfg):
    """
    Build the experiment root corresponding to the TRAINING dataset,
    keeping aggregation / encoder / probe / k identical.
    """
    if cfg.calibration.enabled:
        return Path(cfg.experiment_root)
    return Path(
        str(cfg.experiment_root).replace(
            f"/{cfg.datasets.name}/",
            f"/{cfg.datasets.train_name}/"
        )
    )


def _stage_base_dir(exp_root: Path, stage: str, dataset_name: str) -> Path:
    if stage == "test":
        return exp_root / "testing" / dataset_name
    return exp_root / "validation"


def run_eval(cfg):
    """
    Run evaluation on the dataset split specified by cfg.datasets.split.
    Used for both validation (stage=eval) and test (stage=test).
    """

    exp_root = Path(cfg.experiment_root)
    stage = cfg.stage  # "eval" or "test"
    dataset_name = cfg.datasets.name

    stage_dir = _stage_base_dir(exp_root, stage, dataset_name)
    stage_dir.mkdir(parents=True, exist_ok=True)

    setup_logger(exp_root=stage_dir)
    logging.info(f"========== {stage.upper()} ==========")
    logging.info(f"[Eval] Dataset: {dataset_name}")
    logging.info(f"[Eval] Split: {cfg.datasets.split}")
    logging.info(f"[Eval] Output dir: {stage_dir}")

    prepared = prepare_dataset_inputs(cfg)
    probe_type = prepared["probe"]["type"].lower()

    if probe_type not in {"abmil", "clam", "dsmil", "flow"}:
        # Keep this lightweight check for pooled probes.
        check_subset_consistency(prepared)

    # create dataset and collate_fn based on probe type
    if probe_type in {"abmil", "clam", "dsmil", "flow"}:
        dataset = ToxicologyMILDataset(prepared)
        collate_fn = collate_mil
    else:
        dataset = ToxicologyDataset(prepared)
        collate_fn = None

    data = prepared["data"]

    # build probe and load checkpoint
    probe = build_probe(
        prepared,
        input_dim=data["embed_dim"],
        num_classes=data["num_classes"],
    )


    train_exp_root = build_train_experiment_root(cfg)

    ckpt_path = default_probe_path(
        prepared,
        train_exp_root,
        is_torch=isinstance(probe, TorchProbe),
    )
    # ckpt_path = default_probe_path(
    #     prepared, exp_root, isinstance(probe, TorchProbe)
    # )
    logging.info(f"[Eval] Loading checkpoint from {ckpt_path}")
    probe.load(ckpt_path)

    # run predictions
    logging.info("[Eval] Running predictions…")
    y_pred = probe.predict(dataset, collate_fn=collate_fn)
    y_true = np.asarray(dataset.labels)

    # run misclassification analysis and log results
    run_misclassification_analysis(
        dataset=dataset,
        y_true=y_true,
        y_pred=y_pred,
        exp_root=stage_dir,
        stage=stage,
    )

    # optional probabilities
    try:
        y_proba = probe.predict_proba(dataset, collate_fn=collate_fn)
    except Exception:
        y_proba = None

    # compute and log metrics
    metrics = compute_and_log_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        exp_root=stage_dir,
        class_names=["No Hypertrophy", "Hypertrophy"],
    )

    registry_path = append_experiment_row(
        cfg,
        prepared,
        stage=stage,
        status="completed",
        exp_root=exp_root,
        metrics=metrics,
        checkpoint_path=ckpt_path,
        metrics_path=stage_dir / "metrics" / "metrics.json",
    )
    log_benchmark(cfg, metrics, registry_path=registry_path)
    logging.info(f"[Eval] Updated experiment registry -> {registry_path}")

    logging.info(f"[{stage.upper()}] Final metrics -> {metrics}")
    logging.info(f"========== {stage.upper()} DONE ==========")


if __name__ == "__main__":
    args = get_args()

    # args.stage is set by main.py
    cfg = load_merged_config(args.config, args)

    run_eval(cfg)

    torch.cuda.empty_cache()
    gc.collect()
    raise SystemExit(0)
