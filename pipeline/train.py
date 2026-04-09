# pipeline/train.py

import logging
from pathlib import Path
import torch

from argparser import get_args
from utils.config_loader import load_merged_config
from data.prepare_dataset import prepare_dataset_inputs
from utils.feature_cache import ensure_cached_features

from data.create_datasets import ToxicologyDataset
from data.dataset_check import check_subset_consistency

from data.create_dataset_MIL import ToxicologyMILDataset
from data.collate_MIL import collate_mil


from probes import build_probe, TorchProbe, FlowProbe, default_probe_path
from logger import setup_logger
from utils.experiment_registry import append_experiment_row


def build_base_experiment_root(cfg):
    return Path(
        str(cfg.experiment_root)
        .replace(f"/{cfg.datasets.name}/", f"/{cfg.calibration.base_dataset}/")
        .replace(f"/{cfg.experiment.tag}", f"/{cfg.calibration.base_experiment_tag}")
    )


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

    probe_type = prepared["probe"]["type"].lower()

    # feature caching + subset consistency only needed for probes that don't use MIL directly
    if probe_type not in {"abmil", "clam", "dsmil", "flow"}:
        ensure_cached_features(prepared)
        check_subset_consistency(prepared)

    # create PyTorch dataset for training (pooled vs MIL) based on probe type
    if probe_type in {"abmil", "clam", "dsmil", "flow"}:
        dataset = ToxicologyMILDataset(prepared)
        collate_fn = collate_mil
    
        if probe_type == "flow":
            logging.info("[Train] Using FlowProbe, which only uses training NORMAL samples for fitting.")
            normal_indices = [i for i, label in enumerate(dataset.labels) if label == 0]
            dataset = torch.utils.data.Subset(dataset, normal_indices)
            logging.info(f"[Train] Subsetted to {len(dataset)} NORMAL samples for Flow Probe training.")
                     
    else:
        dataset = ToxicologyDataset(prepared)
        collate_fn = None
       
    # build probe model based on config and get checkpoint path
    input_dim = prepared["data"]["embed_dim"]

    num_classes = prepared["data"]["num_classes"]

    probe = build_probe(prepared, input_dim, num_classes)
    # ckpt_path = default_probe_path(prepared, exp_root, isinstance(probe, TorchProbe))
    ckpt_path = default_probe_path(prepared, exp_root, isinstance(probe, (TorchProbe, FlowProbe)))

    if cfg.calibration.enabled and cfg.calibration.init_from_base:
        base_exp_root = build_base_experiment_root(cfg)
        base_ckpt_path = default_probe_path(
            prepared,
            base_exp_root,
            isinstance(probe, (TorchProbe, FlowProbe)),
        )
        if base_ckpt_path.exists():
            logging.info(f"[Train] Calibration mode -> warm-starting from {base_ckpt_path}")
            probe.load(base_ckpt_path)
        else:
            logging.warning(
                f"[Train] Calibration requested, but base checkpoint was not found at {base_ckpt_path}. "
                "Training from scratch instead."
            )

    # check if checkpoint already exists to skip training if so
    if ckpt_path.exists():
        logging.info(f"[Train] Checkpoint already exists -> skipping training.")
        registry_path = append_experiment_row(
            cfg,
            prepared,
            stage="train",
            status="checkpoint_exists",
            exp_root=exp_root,
            checkpoint_path=ckpt_path,
        )
        logging.info(f"[Train] Updated experiment registry -> {registry_path}")
        return

    # otherwise, fit the probe to the dataset and save the checkpoint
    logging.info("[Train] Starting training…")
    probe.fit(dataset, collate_fn=collate_fn)
    probe.save(ckpt_path)
    registry_path = append_experiment_row(
        cfg,
        prepared,
        stage="train",
        status="completed",
        exp_root=exp_root,
        checkpoint_path=ckpt_path,
    )

    logging.info(f"[Train] Saved checkpoint -> {ckpt_path}")
    logging.info(f"[Train] Updated experiment registry -> {registry_path}")
    logging.info("========== TRAIN DONE ==========")


if __name__ == "__main__":
    args = get_args()
    cfg = load_merged_config(args.config, args=None)

    run_train(cfg)

    import torch, gc
    torch.cuda.empty_cache()
    gc.collect()

    raise SystemExit(0)
