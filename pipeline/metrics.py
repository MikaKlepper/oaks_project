# pipeline/metrics.py

import logging
from pathlib import Path
from typing import Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    class_names: Optional[list] = None,
):
    cm = confusion_matrix(y_true, y_pred)
    if class_names is None:
        classes = [str(i) for i in range(cm.shape[0])]
    else:
        classes = class_names

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logging.info(f"[Metrics] Saved confusion matrix to {out_path}")


def plot_roc_binary(
    y_true: np.ndarray,
    y_proba_pos: np.ndarray,
    out_path: Path,
):
    fpr, tpr, _ = roc_curve(y_true, y_proba_pos)
    auc = roc_auc_score(y_true, y_proba_pos)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logging.info(f"[Metrics] Saved ROC curve to {out_path}")


def compute_and_log_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    exp_root: Path,
    class_names: Optional[list] = None,
) -> Dict[str, float]:
    """
    Compute standard metrics, log them, and save plots to <exp_root>/metrics.
    Assumes binary classification for ROC.
    """
    metrics_dir = _ensure_dir(Path(exp_root) / "metrics")

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary")

    logging.info(f"[Metrics] Accuracy: {acc:.4f}")
    logging.info(f"[Metrics] F1-score: {f1:.4f}")

    # Confusion matrix
    cm_path = metrics_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, cm_path, class_names)

    # ROC curve (binary) if probabilities provided
    roc_auc = None
    if y_proba is not None:
        # assume y_proba is shape (N, 2) or (N,)
        if y_proba.ndim == 2:
            # positive class is index 1
            y_pos = y_proba[:, 1]
        else:
            y_pos = y_proba

        roc_auc = roc_auc_score(y_true, y_pos)
        roc_path = metrics_dir / "roc_curve.png"
        plot_roc_binary(y_true, y_pos, roc_path)
        logging.info(f"[Metrics] ROC-AUC: {roc_auc:.4f}")

    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
    }
