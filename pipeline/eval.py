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

from probes import build_probe, TorchProbe, default_probe_path
from metrics import compute_and_log_metrics
from logger import setup_logger
from log_benchmark import log_benchmark
from eval_analysis import run_misclassification_analysis

def build_train_experiment_root(cfg):
    """
    Build the experiment root corresponding to the TRAINING dataset,
    keeping aggregation / encoder / probe / k identical.
    """
    return Path(
        str(cfg.experiment_root).replace(
            f"/{cfg.datasets.name}/",
            f"/{cfg.datasets.train_name}/"
        )
    )


def run_eval(cfg):
    """
    Run evaluation on the dataset split specified by cfg.datasets.split.
    Used for both validation (stage=eval) and test (stage=test).
    """

    exp_root = Path(cfg.experiment_root)
    stage = cfg.stage  # "eval" or "test"
    dataset_name = cfg.datasets.name

    # Separate output dir for test datasets
    # test can have multiple sets while eval is always on tggates val
    if stage == "test":
        stage_dir = exp_root / stage / dataset_name
    else:
        stage_dir = exp_root / stage

    stage_dir.mkdir(parents=True, exist_ok=True)

    setup_logger(exp_root=stage_dir)
    logging.info(f"========== {stage.upper()} ==========")
    logging.info(f"[Eval] Dataset: {dataset_name}")
    logging.info(f"[Eval] Split: {cfg.datasets.split}")

    prepared = prepare_dataset_inputs(cfg)
    probe_type = prepared["probe"]["type"].lower()

    # if probe_type not in {"abmil", "clam", "dsmil"}:
    #     if dataset_name == "tggates":
    #         ensure_cached_features(prepared)
    #         check_subset_consistency(prepared)
    #     else:
    #         logging.info(
    #             "[Eval] Test-only dataset detected -> skipping feature caching "
    #             "and subset consistency checks"
    #         )

    if probe_type not in {"abmil", "clam", "dsmil"}:
            ensure_cached_features(prepared)
            # check_subset_consistency(prepared)

   # create dataset and collate_fn based on probe type
    if probe_type in {"abmil", "clam", "dsmil"}:
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

    log_benchmark(cfg, metrics)

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
