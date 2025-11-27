# pipeline/eval.py
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from argparser import get_args
from utils.config_loader import load_merged_config
from data.prepare_dataset import prepare_dataset_inputs
from data.create_datasets import ToxicologyDataset
from data.dataset_check import check_subset_consistency
from data.features_per_animal import group_features_by_animal

from probes import build_probe, TorchProbe, default_probe_path
from logger import setup_logger
from metrics import compute_and_log_metrics


def _load_full_dataset_as_tensors(dataset):
    """
    Load the entire dataset as one batch of tensors.
    """
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    for feats, labels in loader:
        return feats, labels
    raise RuntimeError("Dataset loader returned no batches.")


def run_eval(cfg):
    exp_root = Path(cfg.experiment_root)
    setup_logger(exp_root)
    logging.info("========== EVAL STAGE ==========")

    # 1) Prepare dataset
    prepared = prepare_dataset_inputs(cfg)

    if prepared["data"]["features_type"] == "animal":
        logging.info("[EVAL] Building animal-level features...")
        group_features_by_animal(prepared)

    check_subset_consistency(prepared)

    # 2) Build dataset
    ds = ToxicologyDataset(prepared)
    logging.info(f"[Eval] Dataset size: {len(ds)}")

    # 3) Load whole dataset as tensors
    X, y = _load_full_dataset_as_tensors(ds)

    input_dim = prepared["data"]["embed_dim"]
    num_classes = len(np.unique(y.numpy()))

    logging.info(f"[Eval] Input dim: {input_dim}, Num classes: {num_classes}")

    # 4) Build probe and load weights
    probe = build_probe(prepared, input_dim, num_classes)
    is_torch = isinstance(probe, TorchProbe)
    ckpt_path = default_probe_path(prepared, exp_root, is_torch)

    logging.info(f"[Eval] Loading checkpoint: {ckpt_path}")
    probe.load(ckpt_path)

    # 5) Predictions
    y_pred = probe.predict(X)

    # 6) Predict probabilities (if available)
    try:
        y_proba = probe.predict_proba(X)
    except NotImplementedError:
        y_proba = None

    # 7) Compute metrics
    metrics = compute_and_log_metrics(
        y_true=y.cpu().numpy(),
        y_pred=y_pred,
        y_proba=y_proba,
        exp_root=exp_root,
        class_names=["No Hypertrophy", "Hypertrophy"],
    )

    logging.info(f"[Eval] Metrics: {metrics}")
    logging.info("========== EVAL STAGE DONE ==========")

    return metrics


if __name__ == "__main__":
    args = get_args()
    cfg, _ = load_merged_config(args.config, args)
    run_eval(cfg)
