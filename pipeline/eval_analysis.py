# pipeline/eval_analysis.py

import json
import logging
from pathlib import Path

import numpy as np

from eval_plots import plot_severity_histogram, plot_location_histogram


def save_json(path: Path, payload: dict):
    """
    Saves a given dictionary as a JSON file to the given path.

    The directory of the given path is created recursively if it doesn't already exist.

    Parameters:
        path (Path): path to save the JSON file
        payload (dict): dictionary to save as JSON
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def count_distribution(values):
    """
    Count the distribution of values in the given list.

    Parameters:
        values (list): list of values to count

    Returns:
        dict: dictionary with keys as unique values and values as their respective counts
    """
    values = [v for v in values if v is not None]
    if not values:
        return {}
    u, c = np.unique(values, return_counts=True)
    return {str(k): int(v) for k, v in zip(u, c)}


def run_misclassification_analysis(dataset, y_true, y_pred, exp_root, stage):
    """
    Run a detailed misclassification analysis on the given dataset.

    This function runs predictions on the given dataset, computes the
    indices of misclassified samples, and saves a detailed analysis
    to exp_root/<stage>/metrics/misclassified.json and
    exp_root/<stage>/metrics/misclassified_analysis.json.

    The detailed analysis includes the total number of misclassified
    samples, their indices, IDs, true labels, predicted labels,
    severity distribution, and location distribution.

    The summary analysis includes the count distribution of severity
    and location for false positives and false negatives separately.

    Finally, this function plots the severity and location histograms
    of misclassified samples and saves them to exp_root/<stage>/metrics/.

    Parameters:
        dataset (ToxicologyDataset): dataset to run analysis on
        y_true (np.ndarray): true labels
        y_pred (np.ndarray): predicted labels
        exp_root (Path): experiment root directory
        stage (str): "eval" or "test"

    Returns:
        None
    """
    logging.info("[Eval] Running misclassification analysis")

    wrong_idx = np.where(y_pred != y_true)[0]

    metrics_dir = exp_root / stage/ "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    wrong_ids = [dataset.ids[i] for i in wrong_idx]
    wrong_severity = [dataset.severity[i] for i in wrong_idx]
    wrong_location = [dataset.location[i] for i in wrong_idx]

    # detailed misclassification info saved to misclassified.json
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

    # summary analysis: count distribution of severity and location for FP and FN separately, save to misclassified_analysis.json
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

    # plots
    plot_severity_histogram(wrong_severity, metrics_dir)
    plot_location_histogram(wrong_location, metrics_dir)

    logging.info("[Eval] Misclassification analysis saved")
