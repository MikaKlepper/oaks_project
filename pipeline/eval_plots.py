# pipeline/eval_plots.py

import matplotlib.pyplot as plt
import numpy as np


def plot_severity_histogram(severity, out_dir):
    """
    Plot a histogram of the distribution of severity labels in a list of severity strings.

    Parameters
    ----------
    severity : list
        A list of severity strings.
    out_dir : Path
        The directory to save the plot.

    Returns
    -------
    None
    """
    severity_order = ["minimal", "slight", "moderate", "severe"]
    clean = [s for s in severity if s in severity_order]

    if not clean:
        return

    values, counts = np.unique(clean, return_counts=True)
    pairs = sorted(zip(values, counts), key=lambda x: severity_order.index(x[0]))
    labels, heights = zip(*pairs)

    plt.figure(figsize=(7, 4))
    plt.bar(labels, heights)
    plt.title("Severity Distribution — Misclassified WSIs")
    plt.xlabel("Severity")
    plt.ylabel("Count")
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
