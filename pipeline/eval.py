# pipeline/eval.py

import gc
import logging
from pathlib import Path

import numpy as np
import torch

from argparser import get_args
from utils.config_loader import load_merged_config
from utils.feature_cache import ensure_cached_features
from data.prepare_dataset import prepare_dataset_inputs
from data.create_datasets import ToxicologyDataset
from data.dataset_check import check_subset_consistency

from data.create_dataset_MIL import ToxicologyMILDataset
from data.collate_MIL import collate_mil
from probes import MILTorchProbe

from probes import build_probe, TorchProbe, default_probe_path
from metrics import compute_and_log_metrics
from logger import setup_logger
from log_benchmark import log_benchmark

from eval_analysis import run_misclassification_analysis


def run_eval(cfg):
    """
    Run evaluation on the dataset split specified by cfg.datasets.split.
    This function is used for both validation (stage=eval) and test (stage=test).
    """
    exp_root = Path(cfg.experiment_root)
    stage = cfg.stage  # "eval" or "test"
    stage_dir = exp_root / stage

    setup_logger(exp_root)

    logging.info(f"========== {stage.upper()} ==========")
    logging.info(f"[Eval] Dataset split: {cfg.datasets.split}")

    #load and prepare dataset inputs (metadata, feature directories, IDs, labels, etc.)
    prepared = prepare_dataset_inputs(cfg)

    probe_type = prepared["probe"]["type"].lower()

    # only probes work on pooled features, need cached features + subset consistency checks
    if probe_type not in {"abmil", "clam", "dsmil"}:
        ensure_cached_features(prepared)
        check_subset_consistency(prepared)

    # create PyTorch dataset for evaluation (pooled vs MIL)
    if probe_type in {"abmil", "clam", "dsmil"}:
        dataset = ToxicologyMILDataset(prepared)
        collate_fn = collate_mil
    else:
        dataset = ToxicologyDataset(prepared)
        collate_fn = None

    data = prepared["data"]

    # Build probe model based on config and get checkpoint path
    probe = build_probe(
        prepared,
        input_dim=data["embed_dim"],
        num_classes=data["num_classes"],
    )

    ckpt_path = default_probe_path(
        prepared, exp_root, isinstance(probe, TorchProbe)
    )
    logging.info(f"[Eval] Loading checkpoint from {ckpt_path}")
    probe.load(ckpt_path)

    # Run predictions
    logging.info("[Eval] Running predictions…")
    y_pred = probe.predict(dataset, collate_fn=collate_fn)

    y_true = np.asarray(dataset.labels)

    # misclassification analysis: log and analyze misclassified samples,
    # save results to exp_root/eval/misclassification_analysis.csv (or test/)
    run_misclassification_analysis(
        dataset=dataset,
        y_true=y_true,
        y_pred=y_pred,
        exp_root=exp_root,
        stage=stage,
    )

    # optional: predict probabilities
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

    log_benchmark(cfg, metrics)

    logging.info(f"[{stage.upper()}] Final metrics -> {metrics}")
    logging.info(f"========== {stage.upper()} DONE ==========")


if __name__ == "__main__":
    args = get_args()

    # IMPORTANT:
    # args.stage is already set by main.py subprocess call
    cfg = load_merged_config(args.config, args)

    run_eval(cfg)

    torch.cuda.empty_cache()
    gc.collect()
    raise SystemExit(0)
