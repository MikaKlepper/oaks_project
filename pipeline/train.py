# pipeline/train.py

import logging
from pathlib import Path

from argparser import get_args
from utils.config_loader import load_merged_config
from data.prepare_dataset import prepare_dataset_inputs
from utils.feature_cache import ensure_cached_features

from data.create_datasets import ToxicologyDataset
from data.dataset_check import check_subset_consistency

from probes import build_probe, TorchProbe, default_probe_path
from logger import setup_logger


def run_train(cfg):
    """
    Run the training stage of the pipeline.

    This function loads and prepares the dataset inputs (metadata, feature directories, IDs, labels, etc.),
    ensures all required features are computed and cached before training, checks subset consistency,
    builds the probe model, and trains it using the dataset.

    If the checkpoint already exists, training is skipped. Otherwise, the probe is fit to the dataset,
    saved to the checkpoint path, and the training is logged.

    :param cfg: The pipeline configuration to use for training.
    """
    exp_root = Path(cfg.experiment_root)
    setup_logger(exp_root)

    logging.info("========== TRAIN ==========")

    # load and prepare dataset inputs (metadata, feature directories, IDs, labels, etc.)
    prepared = prepare_dataset_inputs(cfg)

    # feature caching: ensure all required features are computed and cached before training
    ensure_cached_features(prepared)

    # check subset consistency: if using a subset for training, ensure no leakage and consistency with metadata and features
    check_subset_consistency(prepared)

    # create PyTorch dataset for training
    dataset = ToxicologyDataset(prepared)

    # build probe model based on config and get checkpoint path
    input_dim = prepared["data"]["embed_dim"]
    num_classes = prepared["data"]["num_classes"]

    probe = build_probe(prepared, input_dim, num_classes)
    ckpt_path = default_probe_path(prepared, exp_root, isinstance(probe, TorchProbe))

    # check if checkpoint already exists to skip training if so 
    if ckpt_path.exists():
        logging.info(f"[Train] Checkpoint already exists -> skipping training.")
        return
    
    # otherwise, fit the probe to the dataset and save the checkpoint
    logging.info("[Train] Starting training…")
    probe.fit(dataset)
    probe.save(ckpt_path)

    logging.info(f"[Train] Saved checkpoint -> {ckpt_path}")
    logging.info("========== TRAIN DONE ==========")


if __name__ == "__main__":
    args = get_args()
    cfg = load_merged_config(args.config, args=None)

    run_train(cfg)

    import torch, gc
    torch.cuda.empty_cache()
    gc.collect()

    raise SystemExit(0)
