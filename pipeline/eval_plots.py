# pipeline/eval_plots.py

import matplotlib.pyplot as plt
import numpy as np


def plot_severity_histogram(severity, out_dir):
    """
    Plot a histogram of the distribution of normalized severity labels.

    Severity is expected to be encoded as integers:
        1 = minimal
        2 = slight
        3 = moderate
        4 = severe

    Parameters
    ----------
    severity : list of int
        A list of normalized severity values.
    out_dir : Path
        The directory to save the plot.

    Returns
    -------
    None
    """
    severity_map = {
        1: "minimal",
        2: "slight",
        3: "moderate",
        4: "severe",
    }

    clean = [s for s in severity if s in severity_map]

    if not clean:
        return

    values, counts = np.unique(clean, return_counts=True)

    # enforce semantic ordering
    order = sorted(values)
    labels = [severity_map[v] for v in order]
    heights = [counts[list(values).index(v)] for v in order]

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
