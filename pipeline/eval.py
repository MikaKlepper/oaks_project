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
from probes import build_probe, TorchProbe, default_probe_path
from logger import setup_logger
from metrics import compute_and_log_metrics


def _dataset_to_numpy(dataset, batch_size: int = 256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    feats_list, labels_list = [], []

    for feats, labels in loader:
        feats_list.append(feats)
        labels_list.append(labels)

    X = torch.cat(feats_list, dim=0).cpu().numpy()
    y = torch.cat(labels_list, dim=0).cpu().numpy()
    return X, y


def run_eval(cfg):
    exp_root = Path(cfg.experiment_root)
    setup_logger(exp_root)
    logging.info("========== EVAL STAGE ==========")

    prepared = prepare_dataset_inputs(cfg)
    check_subset_consistency(prepared)

    ds = ToxicologyDataset(prepared)
    logging.info(f"[Eval] Dataset size: {len(ds)}")

    X, y = _dataset_to_numpy(ds, batch_size=cfg.runtime.batch_size)
    input_dim = X.shape[1]
    num_classes = int(y.max() + 1)
    logging.info(f"[Eval] Input dim: {input_dim}, Num classes: {num_classes}")

    # Build and load probe
    probe = build_probe(cfg, input_dim, num_classes)
    is_torch = isinstance(probe, TorchProbe)
    ckpt_path = default_probe_path(cfg, exp_root, is_torch=is_torch)

    probe.load(ckpt_path)

    # Predictions
    y_pred = probe.predict(X)

    # Probabilities (if available)
    try:
        y_proba = probe.predict_proba(X)
    except NotImplementedError:
        y_proba = None

    metrics = compute_and_log_metrics(
        y_true=y,
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
