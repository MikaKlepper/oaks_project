# pipeline/train.py

import logging
from pathlib import Path

from argparser import get_args
from utils.config_loader import load_merged_config
from data.prepare_dataset import prepare_dataset_inputs

from data.process_slide_features import process_slide_features
from data.features_per_animal import group_features_by_animal
from utils.feature_cache import ensure_cached_features

from data.create_datasets import ToxicologyDataset
from data.dataset_check import check_subset_consistency

from probes import build_probe, TorchProbe, default_probe_path
from logger import setup_logger



def run_train(cfg):
    exp_root = Path(cfg.experiment_root)
    setup_logger(exp_root)

    logging.info("========== TRAIN ==========")

    # ---------------- LOAD DATA ----------------
    prepared = prepare_dataset_inputs(cfg)

    # ---------------- FEATURE CACHING ----------------
    ensure_cached_features(prepared)

    # ---------------- SANITY CHECK ----------------
    check_subset_consistency(prepared)

    # ---------------- DATASET ----------------
    dataset = ToxicologyDataset(prepared)

    input_dim = prepared["data"]["embed_dim"]
    num_classes = prepared["data"]["num_classes"]

    probe = build_probe(prepared, input_dim, num_classes)
    ckpt_path = default_probe_path(prepared, exp_root, isinstance(probe, TorchProbe))

    # ---------------- TRAIN ----------------
    logging.info("[Train] Starting training…")
    probe.fit(dataset)
    probe.save(ckpt_path)

    logging.info(f"[Train] Saved checkpoint → {ckpt_path}")
    logging.info("========== TRAIN DONE ==========")


if __name__ == "__main__":
    args = get_args()
    cfg = load_merged_config(args.config, args=None)

    run_train(cfg)

    import torch, gc
    torch.cuda.empty_cache()
    gc.collect()

    raise SystemExit(0)
