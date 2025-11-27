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
from probes import build_probe, TorchProbe, default_probe_path
from logger import setup_logger


def _load_full_dataset_as_tensors(dataset):
    """
    Load the entire ToxicologyDataset into two tensors (X, y)
    in ONE batch. No NumPy conversion — probes handle both.
    """
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    for feats, labels in loader:
        return feats, labels

    raise RuntimeError("Dataset loader returned no batches.")


def run_train(cfg):
    exp_root = Path(cfg.experiment_root)
    setup_logger(exp_root)
    logging.info("========== TRAIN STAGE ==========")

    # 1) Prepare dataset
    prepared = prepare_dataset_inputs(cfg)

    # 2) Build animal-level features if needed
    if prepared["data"]["features_type"] == "animal":
        logging.info("[Train] Building animal-level features...")
        group_features_by_animal(prepared)

    # 3) Validate split consistency
    check_subset_consistency(prepared)

    # 4) Dataset object
    ds = ToxicologyDataset(prepared)
    logging.info(f"[Train] Dataset size: {len(ds)}")

    # 5) Load full dataset (tensor X, tensor y)
    X, y = _load_full_dataset_as_tensors(ds)

    input_dim = prepared["data"]["embed_dim"]
    num_classes = int(prepared["data"]["num_classes"])

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
