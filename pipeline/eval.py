# pipeline/eval.py

import logging
from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt

from argparser import get_args
from utils.config_loader import load_merged_config
from data.prepare_dataset import prepare_dataset_inputs
from data.create_datasets import ToxicologyDataset
from utils.feature_cache import ensure_cached_features
from data.dataset_check import check_subset_consistency

from probes import build_probe, TorchProbe, default_probe_path
from metrics import compute_and_log_metrics
from logger import setup_logger
from log_benchmark import log_benchmark


def run_eval(cfg):
    exp_root = Path(cfg.experiment_root)
    setup_logger(exp_root)

    logging.info("========== EVAL ==========")

    # ---------------- PREPARE DATA ----------------
    prepared = prepare_dataset_inputs(cfg)
    data = prepared["data"]

    # ---------------- FEATURE CACHING ----------------
    ensure_cached_features(prepared)

    # ---------------- SANITY CHECK ----------------
    check_subset_consistency(prepared)

    # ---------------- LOAD DATASET ----------------
    dataset = ToxicologyDataset(prepared)

    input_dim = data["embed_dim"]
    num_classes = data["num_classes"]

    # ---------------- BUILD & LOAD PROBE ----------------
    probe = build_probe(prepared, input_dim, num_classes)
    ckpt_path = default_probe_path(prepared, exp_root, isinstance(probe, TorchProbe))

    logging.info(f"[Eval] Loading checkpoint → {ckpt_path}")
    probe.load(ckpt_path)

    # ---------------- PREDICTIONS ----------------
    logging.info("[Eval] Running predictions…")
    y_pred = probe.predict(dataset)
    y_true = np.array(dataset.labels)

    # =====================================================
    # MISCLASSIFIED SAMPLES
    # =====================================================
    wrong_idx = np.where(y_pred != y_true)[0]

    wrong_ids = [dataset.ids[i] for i in wrong_idx]
    wrong_severity = [dataset.severity[i] for i in wrong_idx]
    wrong_location = [dataset.location[i] for i in wrong_idx]

    metrics_dir = exp_root / "eval" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    mis_path = metrics_dir / "misclassified.json"

    with open(mis_path, "w") as f:
        json.dump({
            "total_misclassified": len(wrong_ids),
            "wrong_indices": wrong_idx.tolist(),
            "wrong_ids": wrong_ids,
            "y_true_wrong": y_true[wrong_idx].tolist(),
            "y_pred_wrong": y_pred[wrong_idx].tolist(),
            "severity_wrong": wrong_severity,
            "location_wrong": wrong_location
        }, f, indent=2)

    logging.info(f"[Eval] Saved misclassified WSIs → {mis_path}")

    # =====================================================
    # SIMPLE CLEAN HISTOGRAMS (NO NANS)
    # =====================================================

    # --------------------------- #
    #         Severity            #
    # --------------------------- #

    # Remove None
    clean_severity = [s for s in wrong_severity if s is not None]

    # Define order
    severity_order = ["minimal", "slight", "moderate", "severe"]

    # Filter values that match valid categories
    clean_severity = [s for s in clean_severity if s in severity_order]

    if clean_severity:
        values, counts = np.unique(clean_severity, return_counts=True)

        # Sort according to defined order
        sorted_pairs = [(sev, cnt) for sev, cnt in zip(values, counts)]
        sorted_pairs = sorted(sorted_pairs, key=lambda x: severity_order.index(x[0]))

        labels = [p[0] for p in sorted_pairs]
        height = [p[1] for p in sorted_pairs]

        plt.figure(figsize=(7, 4))
        plt.bar(labels, height)
        plt.title("Severity Distribution — Misclassified WSIs")
        plt.xlabel("Severity")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(metrics_dir / "severity_histogram.png", dpi=200)
        plt.close()

    # --------------------------- #
    #         Location            #
    # --------------------------- #
    clean_location = [loc for loc in wrong_location if loc is not None]

    if clean_location:
        values, counts = np.unique(clean_location, return_counts=True)

        plt.figure(figsize=(7, 4))
        plt.bar(values, counts)
        plt.title("Location Distribution — Misclassified WSIs")
        plt.xlabel("Location")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(metrics_dir / "location_histogram.png", dpi=200)
        plt.close()

    # =====================================================
    # FP / FN ANALYSIS
    # =====================================================

    fp_idx = [i for i in wrong_idx if y_pred[i] == 1 and y_true[i] == 0]  # Pred 1, True 0
    fn_idx = [i for i in wrong_idx if y_pred[i] == 0 and y_true[i] == 1]  # Pred 0, True 1

    def count_dist(values):
        values = [v for v in values if v is not None]  # drop None
        if not values:
            return {}
        u, c = np.unique(values, return_counts=True)
        return {str(k): int(v) for k, v in zip(u, c)}

    analysis = {
        "false_positives": {
            "description": "Predicted Hypertrophy (1) but actual No Hypertrophy (0)",
            "count": len(fp_idx),
            "ids": [dataset.ids[i] for i in fp_idx],
            "severity_distribution": count_dist([dataset.severity[i] for i in fp_idx]),
            "location_distribution": count_dist([dataset.location[i] for i in fp_idx])
        },
        "false_negatives": {
            "description": "Predicted No Hypertrophy (0) but actual Hypertrophy (1)",
            "count": len(fn_idx),
            "ids": [dataset.ids[i] for i in fn_idx],
            "severity_distribution": count_dist([dataset.severity[i] for i in fn_idx]),
            "location_distribution": count_dist([dataset.location[i] for i in fn_idx])
        }
    }

    analysis_path = metrics_dir / "misclassified_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)

    logging.info(f"[Eval] Saved FP/FN analysis → {analysis_path}")

    # =====================================================
    # PREDICT PROBA (IF AVAILABLE)
    # =====================================================
    try:
        y_proba = probe.predict_proba(dataset)
    except Exception:
        y_proba = None

    # =====================================================
    # METRICS
    # =====================================================
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
    cfg = load_merged_config(args.config, args=None)

    run_eval(cfg)

    import torch, gc
    torch.cuda.empty_cache()
    gc.collect()

    raise SystemExit(0)
