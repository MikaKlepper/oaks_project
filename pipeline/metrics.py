# pipeline/metrics.py
import json
import logging
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report,
    ConfusionMatrixDisplay
)

# helper to ensure directory exists
def _ensure_dir(path: Path) -> Path:
    """
    Ensures that a given directory exists by creating it recursively if it doesn't already exist.

    Args:
        path: The path to the directory to be ensured.

    Returns:
        The ensured path object.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


# plotting functions
def plot_confusion_matrix(y_true, y_pred, out_path: Path, class_names=None):
    """
    Saves a confusion matrix using sklearn's ConfusionMatrixDisplay.
    """
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=class_names,
        cmap="Blues",
        colorbar=True
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    logging.info(f"[Metrics] Saved confusion matrix → {out_path}")


# roc curve plotting for binary classification
def plot_roc_binary(y_true, y_score, out_path: Path):
    """
    Saves a binary ROC curve using pure matplotlib.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    logging.info(f"[Metrics] Saved ROC curve -> {out_path}")


# main function to compute and log all metrics
def compute_and_log_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        exp_root: Path,
        class_names: Optional[list] = None,
    ) -> Dict[str, float]:
    """
    Computes all classification metrics and saves:
    - confusion_matrix.png
    - roc_curve.png (binary only)
    - classification_report.txt
    - metrics.json
    """

    metrics_dir = _ensure_dir(Path(exp_root) / "metrics")

    # basic classification metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    logging.info(f"[Metrics] Accuracy:  {acc:.4f}")
    logging.info(f"[Metrics] Precision: {prec:.4f}")
    logging.info(f"[Metrics] Recall:    {rec:.4f}")
    logging.info(f"[Metrics] F1-score:  {f1:.4f}")

    
    cm_path = metrics_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, cm_path, class_names)

    # ROC AUC for binary classification only 
    roc_auc = None
    if y_proba is not None and len(np.unique(y_true)) == 2:

        # If shape (N,2), select prob of positive class
        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
            y_score = y_proba[:, 1]
        else:
            y_score = y_proba  # already (N,)

        roc_auc = roc_auc_score(y_true, y_score)
        roc_path = metrics_dir / "roc_curve.png"
        plot_roc_binary(y_true, y_score, roc_path)

        logging.info(f"[Metrics] ROC AUC: {roc_auc:.4f}")

    # classification report
    report_text = classification_report(y_true, y_pred, digits=4)
    with open(metrics_dir / "classification_report.txt", "w") as f:
        f.write(report_text)
    logging.info(f"[Metrics] Saved classification_report.txt")

    # metrics dict for JSON logging
    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
    }

    with open(metrics_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logging.info(f"[Metrics] Saved metrics.json")

    return metrics
