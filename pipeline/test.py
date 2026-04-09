# pipeline/test.py

import logging
from pathlib import Path

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
from utils.experiment_registry import append_experiment_row


def _dataset_to_numpy(dataset, batch_size: int = 256):
    feats_list, labels_list = [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for feats, labels in loader:
        feats_list.append(feats)
        labels_list.append(labels)
    X = torch.cat(feats_list, dim=0).cpu().numpy()
    y = torch.cat(labels_list, dim=0).cpu().numpy()
    return X, y


def run_test(cfg):
    exp_root = Path(cfg.experiment_root)
    setup_logger(exp_root)
    logging.info("========== TEST STAGE ==========")

    prepared = prepare_dataset_inputs(cfg)
    check_subset_consistency(prepared)

    ds = ToxicologyDataset(prepared)
    logging.info(f"[Test] Dataset size: {len(ds)}")

    X, y = _dataset_to_numpy(ds, batch_size=cfg.runtime.batch_size)
    input_dim = X.shape[1]
    num_classes = int(y.max() + 1)
    logging.info(f"[Test] Input dim: {input_dim}, Num classes: {num_classes}")

    probe = build_probe(cfg, input_dim, num_classes)
    is_torch = isinstance(probe, TorchProbe)
    ckpt_path = default_probe_path(cfg, exp_root, is_torch=is_torch)
    probe.load(ckpt_path)

    y_pred = probe.predict(X)

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
    registry_path = append_experiment_row(
        cfg,
        prepared,
        stage="test",
        status="completed",
        exp_root=exp_root,
        metrics=metrics,
        checkpoint_path=ckpt_path,
        metrics_path=exp_root / "metrics" / "metrics.json",
    )

    logging.info(f"[Test] Metrics: {metrics}")
    logging.info(f"[Test] Updated experiment registry -> {registry_path}")
    logging.info("========== TEST STAGE DONE ==========")
    return metrics


if __name__ == "__main__":
    args = get_args()
    cfg, _ = load_merged_config(args.config, args)
    run_test(cfg)
