# pipeline/eval.py

import logging
from pathlib import Path
import numpy as np

from argparser import get_args
from utils.config_loader import load_merged_config
from data.prepare_dataset import prepare_dataset_inputs
from data.create_datasets import ToxicologyDataset
from data.features_per_animal import group_features_by_animal
from data.process_slide_features import process_slide_features
from data.dataset_check import check_subset_consistency
from log_benchmark import log_benchmark

from probes import build_probe, TorchProbe, default_probe_path
from metrics import compute_and_log_metrics
from logger import setup_logger

def run_eval(cfg):
    exp_root = Path(cfg.experiment_root)
    setup_logger(exp_root)

    logging.info("========== EVAL ==========")

    # -----------------------------------------------------
    # PREPARE DATA
    # -----------------------------------------------------
    prepared = prepare_dataset_inputs(cfg)
    data = prepared["data"]

    # -----------------------------------------------------
    # Ensure required features exist
    # -----------------------------------------------------

    # 1) Slide-level mode → only processed slide embeddings needed
    if data["features_type"] == "slide":
        logging.info("[Eval] Ensuring processed slide features are available…")
        process_slide_features(prepared)

    # 2) Animal-level mode → need processed slides first, then animals
    elif data["features_type"] == "animal":
        logging.info("[Eval] Ensuring processed slide features are available for animal aggregation…")
        process_slide_features(prepared)

        logging.info("[Eval] Aggregating processed slides → animal features…")
        group_features_by_animal(prepared)

    # -----------------------------------------------------
    # CONSISTENCY CHECK
    # -----------------------------------------------------
    check_subset_consistency(prepared)

    # -----------------------------------------------------
    # LOAD DATASET
    # -----------------------------------------------------
    dataset = ToxicologyDataset(prepared)

    input_dim = data["embed_dim"]
    num_classes = data["num_classes"]

    probe = build_probe(prepared, input_dim, num_classes)

    # -----------------------------------------------------
    # LOAD CHECKPOINT
    # -----------------------------------------------------
    ckpt_path = default_probe_path(prepared, exp_root, isinstance(probe, TorchProbe))
    logging.info(f"[Eval] Loading checkpoint → {ckpt_path}")
    probe.load(ckpt_path)

    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------
    logging.info("[Eval] Running predictions…")
    y_pred = probe.predict(dataset)
    y_true = np.array(dataset.labels)

    try:
        y_proba = probe.predict_proba(dataset)
    except Exception:
        y_proba = None

    metrics = compute_and_log_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        exp_root=exp_root / "eval",
        class_names=["No Hypertrophy", "Hypertrophy"],
    )
    log_benchmark(cfg, metrics)
    logging.info(f"[Eval] Final metrics → {metrics}")
    logging.info("========== EVAL DONE ==========")


if __name__ == "__main__":
    args = get_args()

    # Load final config (subprocess mode)
    cfg = load_merged_config(args.config, args=None)

    run_eval(cfg)

    import torch, gc
    torch.cuda.empty_cache()
    gc.collect()

    raise SystemExit(0)
