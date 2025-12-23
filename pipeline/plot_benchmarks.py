import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


# ==========================================================
# Pretty names for encoders
# ==========================================================
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


# ==========================================================
# Global styling
# ==========================================================
sns.set_theme(style="whitegrid", font_scale=1.25)

mpl.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "grid.alpha": 0.2,
})


# ==========================================================
# Distinct color + marker mapping
# ==========================================================
def _get_encoder_style_maps(encoders):
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


# ==========================================================
# Learning curve plot
# ==========================================================
def plot_learning_curve(df, out_dir: Path, agg: str):
    df = df.sort_values("k_shot")
    df["pretty_encoder"] = df["encoder"].apply(prettify)

    encoders = sorted(df["encoder"].unique())
    probes = sorted(df["probe"].unique())

    color_map, marker_map = _get_encoder_style_maps(encoders)

    for probe in probes:
        df_probe = df[df["probe"] == probe]

        plt.figure(figsize=(15, 8))

        for enc in encoders:
            df_enc = df_probe[df_probe["encoder"] == enc]
            if df_enc.empty:
                continue

            df_enc = df_enc.sort_values("k_shot")

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

        # Best per k-shot
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

        for _, row in df_best.iterrows():
            plt.annotate(
                prettify(row["encoder"]),
                xy=(row["k_shot"], row["roc_auc"]),
                xytext=(0, 12),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75),
                arrowprops=dict(arrowstyle="->", lw=0.8, alpha=0.6),
            )

        plt.xscale("log")
        k_vals = sorted(df_probe["k_shot"].unique())
        plt.xticks(k_vals, k_vals, fontsize=11)

        plt.xlabel("k-shot", fontsize=14)
        plt.ylabel("ROC-AUC", fontsize=14)
        plt.title(f"{agg.upper()} aggregation — Probe: {probe}", fontsize=20)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

        outfile = out_dir / f"{agg}_learning_curve_{probe}.png"
        plt.tight_layout()
        plt.savefig(outfile, dpi=300)
        plt.close()
        print(f"[PNG] Saved → {outfile}")


# ==========================================================
# Tables of best results
# ==========================================================
def generate_best_tables(df, out_dir: Path, agg: str):
    best_per_k = df.loc[df.groupby(["probe", "k_shot"])["roc_auc"].idxmax()]
    best_overall = df.loc[df.groupby("probe")["roc_auc"].idxmax()]

    best_per_k.to_csv(out_dir / f"{agg}_best_per_probe_per_k.csv", index=False)
    best_overall.to_csv(out_dir / f"{agg}_best_per_probe_overall.csv", index=False)

    print("[CSV] Saved best tables")


# ==========================================================
# Heatmaps
# ==========================================================
def generate_heatmaps(df, out_dir: Path, agg: str):
    k_values = sorted(df["k_shot"].unique())

    for k in k_values:
        df_k = df[df["k_shot"] == k]
        matrix = df_k.pivot(index="encoder", columns="probe", values="roc_auc")
        matrix.rename(index=prettify, inplace=True)

        plt.figure(figsize=(16, 10))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap=sns.color_palette("Blues", as_cmap=True),
            linewidths=0.5,
            cbar_kws={"label": "ROC-AUC"},
            vmin=0.50,
            vmax=1.00,
        )

        plt.title(f"{agg.upper()} aggregation — k={k}", fontsize=18)
        plt.tight_layout()

        out_k = out_dir / f"{agg}_heatmap_k{k}.png"
        plt.savefig(out_k, dpi=300)
        plt.close()
        print(f"[Heatmap] Saved → {out_k}")

    # Mean heatmap
    df_avg = df.groupby(["encoder", "probe"])["roc_auc"].mean().reset_index()
    matrix = df_avg.pivot(index="encoder", columns="probe", values="roc_auc")
    matrix.rename(index=prettify, inplace=True)

    plt.figure(figsize=(16, 10))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.5,
        cbar_kws={"label": "Mean ROC-AUC"},
        vmin=0.50,
        vmax=1.00,
    )

    plt.title(f"{agg.upper()} aggregation — Mean ROC-AUC", fontsize=18)
    plt.tight_layout()

    out = out_dir / f"{agg}_heatmap_mean.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[Heatmap] Saved → {out}")


# ==========================================================
# Master function
# ==========================================================
def run_all_plots(agg: str, stage: str):
    """
    Generate all plots and tables for a given aggregation and stage.

    Args:
        agg   : aggregation type (mean, max, min)
        stage : 'eval' or 'test'
    """
    benchmark_file = Path("outputs") / stage / f"{agg}_benchmark_results.csv"
    if not benchmark_file.exists():
        print(f"[ERROR] Missing benchmark file: {benchmark_file}")
        return

    df = pd.read_csv(benchmark_file)

    out_dir = Path("outputs") / stage / agg
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loaded {len(df)} rows from → {benchmark_file}")

    plot_learning_curve(df, out_dir, agg)
    generate_best_tables(df, out_dir, agg)
    generate_heatmaps(df, out_dir, agg)

    print(f"[DONE] All plots saved under → {out_dir}")
