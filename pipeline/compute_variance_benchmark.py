import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

file = Path("/data/temporary/mika/repos/oaks_project/pipeline/outputs/eval/mean_benchmark_results_without_full_train.csv")

df = pd.read_csv(file)

probe_col = "probe"
model_col = "encoder"
metric_col = "roc_auc"

lambda_risk = 0.5

for col in [probe_col, model_col, metric_col]:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in CSV")

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

# stats per PROBE
probe_stats = (
    df.groupby(probe_col)[metric_col]
      .agg(mean="mean", variance="var", std="std")
      .reset_index()
)

probe_stats["utility"] = probe_stats["mean"] - lambda_risk * probe_stats["std"]
probe_stats = probe_stats.sort_values(by="mean", ascending=False)

probe_output = file.parent / "probe_roc_auc_stats.csv"
probe_stats.to_csv(probe_output, index=False)

# stats per MODEL
model_stats = (
    df.groupby(model_col)[metric_col]
      .agg(mean="mean", variance="var", std="std")
      .reset_index()
)

model_stats["utility"] = model_stats["mean"] - lambda_risk * model_stats["std"]
model_stats = model_stats.sort_values(by="mean", ascending=False)

model_output = file.parent / "model_roc_auc_stats.csv"
model_stats.to_csv(model_output, index=False)

# Print summaries
print("=== Probe statistics (sorted by mean) ===")
print(probe_stats)

print("\n=== Model statistics (sorted by mean) ===")
print(model_stats)

print("\nSaved to:")
print(probe_output)
print(model_output)

# Optional: create heatmap for kNN probes across models
df_knn = df[df[probe_col].str.lower().str.contains("knn")].copy()

if df_knn.empty:
    print("[WARN] No KNN probes found – skipping heatmap.")
else:
    knn_stats = (
        df_knn.groupby(model_col)[metric_col]
        .mean()
        .reset_index()
        .sort_values(metric_col, ascending=False)
    )

    knn_stats["pretty_encoder"] = knn_stats[model_col].apply(prettify)

    heatmap_matrix = pd.DataFrame(
        [knn_stats[metric_col].values],
        columns=knn_stats["pretty_encoder"].values,
        index=["kNN"]
    )

    plt.figure(figsize=(max(14, 0.75 * heatmap_matrix.shape[1]), 4))

    ax = sns.heatmap(
        heatmap_matrix,
        annot=True,
        fmt=".2f",
        cmap=sns.color_palette("Blues", as_cmap=True),
        vmin=0.5,
        vmax=0.70,
        linewidths=0.6,
        cbar_kws={
            "label": "Mean ROC-AUC",
            "ticks": [0.50, 0.60, 0.70],
            "shrink": 0.9,
            "aspect": 12,
            "pad": 0.02
        },
    )

    for t in ax.texts:
        t.set_fontsize(11)

    # Customize colorbar ticks and label
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Mean ROC-AUC", fontsize=13)

    plt.xlabel("Foundation model", fontsize=13)
    plt.ylabel("")
    plt.title("kNN probe — Mean ROC-AUC across k-shot", fontsize=17, pad=12)

    plt.tight_layout()

    knn_out = file.parent / "knn_mean_auc_heatmap.png"
    plt.savefig(knn_out, dpi=300)
    plt.close()

    print(f"[PNG] KNN heatmap saved -> {knn_out}")
