# pipeline/metrics.py
import json
import logging
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,confusion_matrix, roc_auc_score, roc_curve,classification_report, ConfusionMatrixDisplay

# helper function to ensure directory exists
def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

# create and save confusion matrix plot
def plot_confusion_matrix(y_true, y_pred, out_path, class_names=None):
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

# roc curve for binary classification
def plot_roc_binary(y_true, y_score, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")

    ax.set(
        xlabel = "False Positive Rate",
        ylabel = "True Positive Rate",
        title = "ROC Curve"
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logging.info(f"[Metrics] Saved ROC curve → {out_path}")


# maini function to compute and log metrics
def compute_and_log_metrics(
        y_true:np.ndarray,
        y_pred:np.ndarray,
        y_proba:Optional[np.ndarray],
        exp_root:Path,
        class_names:Optional[list]=None,
    ) -> Dict[str, float]:
    
    metrics_dir = _ensure_dir(Path(exp_root) / "metrics")
    #Compute basic metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary")
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    logging.info(f"[Metrics] Accuracy:  {acc:.4f}")
    logging.info(f"[Metrics] Precision: {prec:.4f}")
    logging.info(f"[Metrics] Recall:    {rec:.4f}")
    logging.info(f"[Metrics] F1-score:  {f1:.4f}")

    #Confusion Matrix
    cm_path = metrics_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, cm_path, class_names)

    # roc auc
    roc_auc = None
    if y_proba is not None and len(np.unique(y_true)) == 2:
        # if y_proba is shape (N,2), take positive class column
        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
            y_score = y_proba[:, 1]
        else:
            y_score = y_proba  # assume already (N,)

        roc_auc = roc_auc_score(y_true, y_score)
        plot_roc_binary(y_true, y_score, metrics_dir / "roc_curve.png")
        logging.info(f"[Metrics] ROC AUC: {roc_auc:.4f}")

    #Classification Report
    report = classification_report(y_true, y_pred, digits=4)
    with open(metrics_dir / "classification_report.txt", "w") as f:
        f.write(report)
    logging.info(f"[Metrics] Saved classification report")

    #Store numeric metrics
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
