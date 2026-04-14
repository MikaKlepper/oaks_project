import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


# encoders
ENCODER_ORDER = [
    "CONCH",
    "H_OPTIMUS_0",
    "H_OPTIMUS_1",
    "UNI",
    "UNI_2",
    "VIRCHOW2",
    "KAIKO",
    "PHIKON",
    "PHIKON_V2",
    "MIDNIGHT12K",
    "PRISM",
    "RESNET50",
    "HIBOU_B",
    "HIBOU_L",
    "PROV_GIGAPATH_224_SLIDE",
    "PROV_GIGAPATH_256_SLIDE",
    "PROV_GIGAPATH_224_TILE",
    "PROV_GIGAPATH_256_TILE",
]

# pretty names for encoders (for plots)
def prettify(name: str) -> str:
    mapping = {
        "H_OPTIMUS_1": "H-Optimus-1",
        "H_OPTIMUS_0": "H-Optimus-0",
        "UNI": "UNI",
        "UNI_2": "UNI-2",
        "CONCH": "Conch",
        "VIRCHOW2": "Virchow-2",
        "PHIKON_V2": "Phikon-V2",
        "KAIKO": "Kaiko",
        "MIDNIGHT12K": "Midnight-12K",
        "PRISM": "Prism",
        "RESNET50": "ResNet-50",
        "HIBOU_B": "Hibou-B",
        "HIBOU_L": "Hibou-L",
        "PROV_GIGAPATH_224_TILE": "GigaPath-224 (Tile)",
        "PROV_GIGAPATH_256_TILE": "GigaPath-256 (Tile)",
        "PROV_GIGAPATH_224_SLIDE": "GigaPath-224 (Slide)",
        "PROV_GIGAPATH_256_SLIDE": "GigaPath-256 (Slide)",
    }
    return mapping.get(name, name.replace("_", "-"))


# general style
sns.set_theme(style="whitegrid", font_scale=1.25)

mpl.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "grid.alpha": 0.2,
})

# heatmap style
HEATMAP_CMAP = sns.color_palette("Blues", as_cmap=True)
HEATMAP_VMIN = 0.50
HEATMAP_VMAX = 1.00



def _get_encoder_style_maps(encoders):
    """
    Returns a tuple of two dictionaries: color_map and marker_map.
    color_map maps each encoder to a color for plotting.
    marker_map maps each encoder to a marker style for plotting.
    """
    colors = (
        list(plt.get_cmap("tab20").colors)
        + list(plt.get_cmap("tab20b").colors)
        + list(plt.get_cmap("tab20c").colors)
    )
    markers = ["o", "s", "^", "D", "P", "X", "*", "v"]

    color_map = {}
    marker_map = {}

    for i, enc in enumerate(encoders):
        color_map[enc] = colors[i % len(colors)]
        marker_map[enc] = markers[i % len(markers)]

    return color_map, marker_map



def plot_learning_curve(df, out_dir: Path, agg: str):
    """
    Plots a learning curve for each probe in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the results of the experiment.
    out_dir : Path
        Output directory for the plot.
    agg : str
        String indicating the aggregation method used (e.g. "mean", "max","min", "mil").

    Notes
    ----
    Each probe is plotted in a separate figure. For each encoder, the ROC-AUC
    is plotted against the k-shot value. The best ROC-AUC across all encoders
    is also plotted as a black line.

    The x-axis is log-scaled.

    The plot is saved as a PNG file in the output directory.
    """
    df = df.sort_values("k_shot")
    df["pretty_encoder"] = df["encoder"].apply(prettify)

    encoders = [e for e in ENCODER_ORDER if e in df["encoder"].unique()]
    probes = sorted(df["probe"].unique())

    color_map, marker_map = _get_encoder_style_maps(encoders)

    # show these k-shot values on the x-axis (if present in the data)
    DESIRED_K_TICKS = [1, 5, 10, 20, 40, 80, 100]

    for probe in probes:
        df_probe = df[df["probe"] == probe]

        plt.figure(figsize=(15, 8))

        for enc in encoders:
            df_enc = df_probe[df_probe["encoder"] == enc]
            if df_enc.empty:
                continue

            plt.plot(
                df_enc["k_shot"],
                df_enc["roc_auc"],
                marker=marker_map[enc],
                linewidth=2.8,
                markersize=9,
                alpha=0.9,
                color=color_map[enc],
                label=prettify(enc),
            )

        df_best = (
            df_probe
            .loc[df_probe.groupby("k_shot")["roc_auc"].idxmax()]
            .sort_values("k_shot")
        )

        plt.plot(
            df_best["k_shot"],
            df_best["roc_auc"],
            color="black",
            linewidth=4,
            marker="o",
            markersize=12,
            alpha=0.3,
            label="Best ROC-AUC",
            zorder=5,
        )

        # annotate the best encoder above each k-shot point (paper-friendly, no title clutter)
        for _, row in df_best.iterrows():
            plt.annotate(
                prettify(row["encoder"]),
                xy=(row["k_shot"], row["roc_auc"]),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                color="black",
                alpha=0.9,
                zorder=6,
            )

        plt.xscale("log")

        # enforce desired ticks (only those that exist in the data)
        present_ticks = [k for k in DESIRED_K_TICKS if k in set(df_probe["k_shot"].unique())]
        if present_ticks:
            plt.xticks(present_ticks, [str(k) for k in present_ticks])
        else:
            # fallback: show whatever is present
            plt.xticks(sorted(df_probe["k_shot"].unique()))

        plt.xlabel("k-shot")
        plt.ylabel("ROC-AUC")
        plt.title(f"Probe: {probe}")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

        outfile = out_dir / f"{agg}_learning_curve_{probe}.png"
        plt.tight_layout()
        plt.savefig(outfile, dpi=300)
        plt.close()




def generate_best_tables(df, out_dir: Path, agg: str):
    """
    Generates two CSV files containing the best performing probes per k-shot and over all k-shots.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the results of the benchmark.
    out_dir (Path): Directory where the CSV files will be saved.
    agg (str): Aggregation method used in the benchmark (e.g. "mean", "mil").

    Returns:
    None
    """
    best_per_k = df.loc[df.groupby(["probe", "k_shot"])["roc_auc"].idxmax()]
    best_overall = df.loc[df.groupby("probe")["roc_auc"].idxmax()]

    best_per_k.to_csv(out_dir / f"{agg}_best_per_probe_per_k.csv", index=False)
    best_overall.to_csv(out_dir / f"{agg}_best_per_probe_overall.csv", index=False)


def generate_heatmaps(df, out_dir: Path, agg: str):
    """
    Generate heatmaps for each k-shot and the mean ROC-AUC over all k-shots.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the results of the benchmark.
    out_dir : Path
        Directory where the heatmaps will be saved.
    agg : str
        Aggregation method used to generate the results.

    Returns
    -------
    None
    """
    for k in sorted(df["k_shot"].unique()):
        df_k = df[df["k_shot"] == k]
        matrix = df_k.pivot(index="encoder", columns="probe", values="roc_auc")
        matrix = matrix.loc[[e for e in ENCODER_ORDER if e in matrix.index]]
        matrix.rename(index=prettify, inplace=True)

        plt.figure(figsize=(16, 10))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap=HEATMAP_CMAP,
            vmin=HEATMAP_VMIN,
            vmax=HEATMAP_VMAX,
            linewidths=0.5,
            cbar_kws={"label": "ROC-AUC"},
        )
        # plt.title(f"{agg.upper()} aggregation — k={k}")
        plt.title(f"k={k}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{agg}_heatmap_k{k}.png", dpi=300)
        plt.close()

    df_avg = df.groupby(["encoder", "probe"])["roc_auc"].mean().reset_index()
    matrix = df_avg.pivot(index="encoder", columns="probe", values="roc_auc")
    matrix = matrix.loc[[e for e in ENCODER_ORDER if e in matrix.index]]
    matrix.rename(index=prettify, inplace=True)

    plt.figure(figsize=(16, 10))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap=HEATMAP_CMAP,
        vmin=HEATMAP_VMIN,
        vmax=HEATMAP_VMAX,
        linewidths=0.5,
        cbar_kws={"label": "Mean ROC-AUC"},
    )
    plt.title(f"Mean ROC-AUC across all k-shots")
    plt.tight_layout()
    plt.savefig(out_dir / f"{agg}_heatmap_mean.png", dpi=300)
    plt.close()


def _stage_dir(stage: str) -> str:
    return "validation" if stage == "eval" else "testing"


def run_all_plots(agg: str, stage: str, dataset: str):
    """
    Plots all relevant figures for the given aggregation.

    Parameters
    ----------
    agg : str
        Aggregation type (e.g. "min", "mean", "max")
    stage : str
        Stage of the benchmark (e.g. "eval", "test")
    dataset : str
        Dataset name (e.g. "ucb")

    Returns
    -------
    None
    """
    benchmark_file = Path("outputs") / _stage_dir(stage) / dataset / f"{agg}_benchmark_results.csv"
    if not benchmark_file.exists():
        print(f"[ERROR] Missing benchmark file: {benchmark_file}")
        return

    df = pd.read_csv(benchmark_file)
    out_dir = Path("outputs") / _stage_dir(stage) / dataset / agg
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_learning_curve(df, out_dir, agg)
    generate_best_tables(df, out_dir, agg)
    generate_heatmaps(df, out_dir, agg)


def combine_mil_and_mean(stage: str, dataset: str):
    """
    Combines the mean and MIL benchmark results into a single CSV file.

    The file is saved to "outputs/<stage>/<dataset>/combined_benchmark_results.csv".

    Parameters
    ----------
    stage : str
        Stage of the benchmark (e.g. "eval", "test")
    dataset : str
        Dataset name (e.g. "ucb")

    Returns
    -------
    None
    """
    base_dir = Path("outputs") / _stage_dir(stage) / dataset
    mean_file = base_dir / "mean_benchmark_results.csv"
    mil_file = base_dir / "MIL_benchmark_results.csv"

    dfs = []
    if mean_file.exists():
        df = pd.read_csv(mean_file)
        df["agg"] = "mean"
        dfs.append(df)
    if mil_file.exists():
        df = pd.read_csv(mil_file)
        df["agg"] = "MIL"
        dfs.append(df)

    if dfs:
        pd.concat(dfs).to_csv(base_dir / "combined_benchmark_results.csv", index=False)


def run_all_plots_combined(stage: str, dataset: str):
    """
    Runs all plots (learning curve, best tables, and heatmaps) for the combined
    mean and MIL benchmark results.

    Parameters
    ----------
    stage : str
        Stage of the benchmark (e.g. "eval", "test")
    dataset : str
        Dataset name (e.g. "ucb")

    Returns
    -------
    None
    """
    combined_file = Path("outputs") / _stage_dir(stage) / dataset / "combined_benchmark_results.csv"
    if not combined_file.exists():
        return

    df = pd.read_csv(combined_file).drop(columns=["agg"], errors="ignore")
    out_dir = Path("outputs") / _stage_dir(stage) / dataset / "combined"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_learning_curve(df, out_dir, agg="combined")
    generate_best_tables(df, out_dir, agg="combined")
    generate_heatmaps(df, out_dir, agg="combined")
