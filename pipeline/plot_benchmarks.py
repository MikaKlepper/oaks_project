import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

BENCHMARK_FILE = Path("outputs/benchmark_results.csv") # UPDATE THIS

# ==============================================================
# Prettify encoder names
# ==============================================================

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


# ==============================================================
# Global Seaborn + Matplotlib Styling
# ==============================================================

sns.set_theme(style="whitegrid", font_scale=1.25)

mpl.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "grid.alpha": 0.2,
})


# ==============================================================
# SEABORN LEARNING CURVE PLOTS (per probe)
# ==============================================================

def plot_learning_curve():
    if not BENCHMARK_FILE.exists():
        print("[ERROR] Missing benchmark file.")
        return

    df = pd.read_csv(BENCHMARK_FILE)
    df = df.sort_values("k_shot")
    df["pretty_encoder"] = df["encoder"].apply(prettify)

    output_dir = Path("outputs/benchmark_results")  # UPDATE THIS
    output_dir.mkdir(parents=True, exist_ok=True)

    encoders = sorted(df["encoder"].unique())
    probes = sorted(df["probe"].unique())

    # -------------------------------------------
    # FIXED GLOBAL COLOR MAP FOR ALL ENCODERS
    # -------------------------------------------
    # MUCH nicer than tab20 and consistent
    palette = sns.color_palette("husl", len(encoders))
    color_map = {enc: palette[i] for i, enc in enumerate(encoders)}

    for probe in probes:
        df_probe = df[df["probe"] == probe].copy()

        plt.figure(figsize=(15, 8))

        # ----------------------------------------------------
        # MANUAL PLOTTING → FIXED CONSISTENT COLORS
        # ----------------------------------------------------
        for enc in encoders:
            df_enc = df_probe[df_probe["encoder"] == enc]
            if df_enc.empty:
                continue

            df_enc = df_enc.sort_values("k_shot")

            plt.plot(
                df_enc["k_shot"],
                df_enc["roc_auc"],
                marker="o",
                linewidth=2.1,
                markersize=7,
                alpha=0.85,
                color=color_map[enc],      # FIXED COLOR
                label=prettify(enc),
            )

        # ----------------------------------------------------
        # BEST CURVE (soft black highlight)
        # ----------------------------------------------------
        df_best = df_probe.loc[df_probe.groupby("k_shot")["roc_auc"].idxmax()]
        df_best = df_best.sort_values("k_shot")

        plt.plot(
            df_best["k_shot"],
            df_best["roc_auc"],
            color="black",
            linewidth=4,
            marker="o",
            markersize=10,
            alpha=0.35,
            label="Best ROC-AUC",
            zorder=5
        )

        # ----------------------------------------------------
        # BEST POINT LABELS
        # ----------------------------------------------------
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

        # ----------------------------------------------------
        # PERFECT LOG-SCALE k AXIS
        # ----------------------------------------------------
        plt.xscale("log")
        k_values = sorted(df_probe["k_shot"].unique())
        plt.xticks(k_values, k_values, fontsize=11)

        plt.xlabel("k-shot", fontsize=14)
        plt.ylabel("ROC-AUC", fontsize=14)
        plt.title(f"Learning Curve for Probe: {probe}", fontsize=20)

        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Encoder")
        plt.tight_layout()

        outfile = output_dir / f"learning_curve_{probe}.png"
        plt.savefig(outfile, dpi=300)
        plt.close()

        print(f"[PNG] Saved → {outfile}")


# ==============================================================
# BEST TABLES (per probe, per k, overall)
# ==============================================================

def generate_best_table():
    df = pd.read_csv(BENCHMARK_FILE)

    best_per_k = df.loc[df.groupby(["probe", "k_shot"])["roc_auc"].idxmax()]
    best_overall = df.loc[df.groupby("probe")["roc_auc"].idxmax()]

    output_dir = Path("outputs/benchmark_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    best_per_k.to_csv(output_dir / "best_per_probe_per_k.csv", index=False)
    best_overall.to_csv(output_dir / "best_per_probe_overall.csv", index=False)

    print("[CSV] Saved best_per_probe_per_k.csv")
    print("[CSV] Saved best_per_probe_overall.csv")


# ==============================================================
# SEABORN HEATMAP: MODEL × PROBE
# ==============================================================

def heatmap_models_by_probes():
    df = pd.read_csv(BENCHMARK_FILE)
    output_dir = Path("outputs/benchmark_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    k_values = sorted(df["k_shot"].unique())
    # for all k-shot values create heatmaps
    for k in k_values:
        df_k = df[df["k_shot"] == k]
        # create pivot table, rows= encoder, columns=probe, values=roc_auc
        matrix_k = df_k.pivot(index="encoder", columns="probe", values="roc_auc")
        matrix_k.rename(index=prettify, inplace=True)

        plt.figure(figsize=(16, 10))
        #mask = matrix_k < 0.50  # highlight values less than 0.5
        sns.heatmap(
            matrix_k,
            annot=True,
            fmt=".2f",
            cmap=sns.color_palette("Blues", as_cmap=True),  # strong blue gradient
            linewidths=0.5,
            cbar_kws={"label": "ROC-AUC"},
            vmin=0.50,          # color range lower bound
            vmax=1.00,          # color range upper bound
            #mask=mask,          # hide values < .50 (white)
        )
       
        plt.title(f"Model Performance Across Probes (k={k})", fontsize=18)
        plt.xlabel("Probe")
        plt.ylabel("Encoder")
        plt.tight_layout()
        out_k = output_dir / f"model_vs_probe_heatmap_k{k}.png"
        plt.savefig(out_k, dpi=300)
        plt.close()
        print(f"[Heatmap] Saved --> {out_k}")
    
    # now average over k-shot values
    df_avg = (
        df.groupby(["encoder", "probe"])["roc_auc"]
        .mean()
        .reset_index()
    )

    matrix = df_avg.pivot(index="encoder", columns="probe", values="roc_auc")
    matrix.rename(index=prettify, inplace=True)

    plt.figure(figsize=(16, 10))
    sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap=sns.color_palette("Blues", as_cmap=True),  # strong blue gradient
            linewidths=0.5,
            cbar_kws={"label": "ROC-AUC"},
            vmin=0.50,          # color range lower bound
            vmax=1.00,          # color range upper bound
            #mask=mask,          # hide values < .50 (white)
        )
    plt.title("Model Performance Across Probes (Mean ROC-AUC over k-shot)", fontsize=18)
    plt.xlabel("Probe")
    plt.ylabel("Encoder")
    plt.tight_layout()
    out = output_dir / "model_vs_probe_heatmap_mean.png"
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"[Heatmap] Saved --> {out}")


# ==============================================================
# MAIN EXECUTION
# ==============================================================

if __name__ == "__main__":
    plot_learning_curve()
    generate_best_table()
    heatmap_models_by_probes()
