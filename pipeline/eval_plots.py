# pipeline/eval_plots.py

import matplotlib.pyplot as plt
import numpy as np


def plot_severity_histogram(summary, out_dir):
    """
    Plot misclassification rate by severity bucket, including a normal bucket.

    Severity is expected to be encoded as integers:
        0 = normal / no severity
        1 = minimal
        2 = slight
        3 = moderate
        4 = severe

    Parameters
    ----------
    summary : dict
        Mapping from severity code -> summary stats with misclassification_pct.
    out_dir : Path
        The directory to save the plot.

    Returns
    -------
    None
    """
    severity_map = {
        0: "normal",
        1: "minimal",
        2: "slight",
        3: "moderate",
        4: "severe",
    }

    valid_keys = [
        int(k) for k in summary.keys()
        if str(k).isdigit() and int(k) in severity_map
    ]
    if not valid_keys:
        return

    order = sorted(valid_keys)
    labels = [severity_map[v] for v in order]
    heights = [summary[str(v)]["misclassification_pct"] for v in order]

    plt.figure(figsize=(7, 4))
    plt.bar(labels, heights)
    plt.title("Misclassification Rate By Severity")
    plt.xlabel("Severity / Normal")
    plt.ylabel("Misclassified (%)")
    plt.ylim(0, max(heights) * 1.15 if heights else 1)
    plt.tight_layout()
    plt.savefig(out_dir / "severity_histogram.png", dpi=200)
    plt.close()


def plot_location_histogram(location, out_dir):
    """
    Plot a histogram of the distribution of location labels in a list of location strings.

    Parameters
    ----------
    location : list
        A list of location strings.
    out_dir : Path
        The directory to save the plot.

    Returns
    -------
    None
    """
    clean = [l for l in location if l is not None]
    if not clean:
        return

    values, counts = np.unique(clean, return_counts=True)

    plt.figure(figsize=(7, 4))
    plt.bar(values, counts)
    plt.title("Location Distribution — Misclassified WSIs")
    plt.xlabel("Location")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "location_histogram.png", dpi=200)
    plt.close()


def plot_severity_misclassification_rate(summary, out_dir):
    """
    Plot misclassification rate by severity as a percentage of all cases
    available at each severity level.
    """
    severity_map = {
        0: "normal",
        1: "minimal",
        2: "slight",
        3: "moderate",
        4: "severe",
    }

    valid_keys = [
        int(k) for k in summary.keys()
        if str(k).isdigit() and int(k) in severity_map
    ]
    if not valid_keys:
        return

    order = sorted(valid_keys)
    labels = [severity_map[k] for k in order]
    percentages = [summary[str(k)]["misclassification_pct"] for k in order]

    plt.figure(figsize=(7, 4))
    plt.bar(labels, percentages)
    plt.title("Misclassification Rate By Severity")
    plt.xlabel("Severity")
    plt.ylabel("Misclassified (%)")
    plt.ylim(0, max(percentages) * 1.15 if percentages else 1)
    plt.tight_layout()
    plt.savefig(out_dir / "severity_misclassification_rate.png", dpi=200)
    plt.close()
