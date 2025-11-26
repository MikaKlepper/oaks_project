# pipeline/train.py
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from argparser import get_args
from utils.config_loader import load_merged_config
from data.prepare_dataset import prepare_dataset_inputs
from data.features_per_animal import group_features_by_animal
from data.dataset_check import check_subset_consistency
from data.create_datasets import ToxicologyDataset
from probes import build_probe, TorchProbe, SklearnProbe, default_probe_path
from logger import setup_logger


def _dataset_to_numpy(dataset, batch_size: int = 256):
    """
    Convert ToxicologyDataset -> (X, y) numpy arrays.
    We only load tensors from .pt here, no gradient tracking.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    feats_list = []
    labels_list = []

    for feats, labels in loader:
        feats_list.append(feats)
        labels_list.append(labels)

    X = torch.cat(feats_list, dim=0).cpu().numpy()
    y = torch.cat(labels_list, dim=0).cpu().numpy()

    return X, y


def run_train(cfg):
    exp_root = Path(cfg.experiment_root)
    setup_logger(exp_root)
    logging.info("========== TRAIN STAGE ==========")

    # 1) Prepare dataset inputs
    prepared = prepare_dataset_inputs(cfg)

    # 2) Build animal features if needed
    if prepared["features_type"] == "animal":
        logging.info("[Train] Building animal-level features...")
        group_features_by_animal(prepared)

    # 3) Consistency checks
    check_subset_consistency(prepared)

    # 4) Build dataset
    ds = ToxicologyDataset(prepared)
    logging.info(f"[Train] Dataset size: {len(ds)}")

    # 5) Convert to numpy for probes
    X, y = _dataset_to_numpy(ds, batch_size=prepared.runtime.batch_size)
    input_dim = X.shape[1]
    num_classes = int(y.max() + 1)

    logging.info(f"[Train] Input dim: {input_dim}, Num classes: {num_classes}")

    # 6) Build & train probe
    probe = build_probe(prepared, input_dim, num_classes)
    probe.fit(X, y)

    # 7) Save checkpoint
    is_torch = isinstance(probe, TorchProbe)
    ckpt_path = default_probe_path(prepared, exp_root, is_torch=is_torch)
    probe.save(ckpt_path)

    logging.info("========== TRAIN STAGE DONE ==========")
    return probe, ckpt_path


if __name__ == "__main__":
    args = get_args()
    cfg, _ = load_merged_config(args.config, args)
    run_train(cfg)
