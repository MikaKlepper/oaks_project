# pipeline/eval_analysis.py

import json
import logging
from pathlib import Path

import numpy as np

from eval_plots import plot_severity_histogram, plot_location_histogram


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def count_distribution(values):
    values = [v for v in values if v is not None]
    if not values:
        return {}
    u, c = np.unique(values, return_counts=True)
    return {str(k): int(v) for k, v in zip(u, c)}


def run_misclassification_analysis(dataset, y_true, y_pred, exp_root):
    logging.info("[Eval] Running misclassification analysis")

    wrong_idx = np.where(y_pred != y_true)[0]

    metrics_dir = exp_root / "eval" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    wrong_ids = [dataset.ids[i] for i in wrong_idx]
    wrong_severity = [dataset.severity[i] for i in wrong_idx]
    wrong_location = [dataset.location[i] for i in wrong_idx]

    # ---------------- Raw dump ----------------
    save_json(
        metrics_dir / "misclassified.json",
        {
            "total_misclassified": len(wrong_idx),
            "indices": wrong_idx.tolist(),
            "ids": wrong_ids,
            "y_true": y_true[wrong_idx].tolist(),
            "y_pred": y_pred[wrong_idx].tolist(),
            "severity": wrong_severity,
            "location": wrong_location,
        },
    )

    # ---------------- FP / FN ----------------
    fp_idx = [i for i in wrong_idx if y_pred[i] == 1 and y_true[i] == 0]
    fn_idx = [i for i in wrong_idx if y_pred[i] == 0 and y_true[i] == 1]

    analysis = {
        "false_positives": {
            "count": len(fp_idx),
            "ids": [dataset.ids[i] for i in fp_idx],
            "severity_distribution": count_distribution(
                [dataset.severity[i] for i in fp_idx]
            ),
            "location_distribution": count_distribution(
                [dataset.location[i] for i in fp_idx]
            ),
        },
        "false_negatives": {
            "count": len(fn_idx),
            "ids": [dataset.ids[i] for i in fn_idx],
            "severity_distribution": count_distribution(
                [dataset.severity[i] for i in fn_idx]
            ),
            "location_distribution": count_distribution(
                [dataset.location[i] for i in fn_idx]
            ),
        },
    }

    save_json(metrics_dir / "misclassified_analysis.json", analysis)

    # ---------------- Plots ----------------
    plot_severity_histogram(wrong_severity, metrics_dir)
    plot_location_histogram(wrong_location, metrics_dir)

    logging.info("[Eval] Misclassification analysis saved")
