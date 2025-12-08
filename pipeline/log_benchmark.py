# pipeline/log_benchmark.py
import pandas as pd
from pathlib import Path

def log_benchmark(cfg, metrics):
    """
    Logs benchmark results into:
        outputs/<aggregation>_benchmark_results.csv

    Ensures:
    - No duplicate rows for same experiment
    - Existing entries are updated correctly
    """

    # CSV path depends on aggregation type
    BENCHMARK_FILE = Path(f"outputs/{cfg.aggregation.type}_benchmark_results.csv")

    # Row to insert or update
    new_row = {
        "encoder": cfg.features.encoder,
        "probe": cfg.probe.type,
        "k_shot": cfg.fewshot.k,
        "feature_type": cfg.features.feature_type,
        "aggregation": cfg.aggregation.type,
        "split": cfg.datasets.split,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
    }

    # ============================================================
    # CASE 1: File does NOT exist → create new file with row
    # ============================================================
    if not BENCHMARK_FILE.exists():
        df = pd.DataFrame([new_row])
        df.to_csv(BENCHMARK_FILE, index=False)
        print(f"[Benchmark] Created new benchmark file → {BENCHMARK_FILE}")
        return

    # ============================================================
    # CASE 2: Load existing CSV and update/append
    # ============================================================
    df = pd.read_csv(BENCHMARK_FILE)

    # Ensure aggregation column exists (migration support)
    if "aggregation" not in df.columns:
        df["aggregation"] = cfg.aggregation.type

    # Identify same experiment rows
    same_exp = (
        (df["encoder"] == new_row["encoder"]) &
        (df["probe"] == new_row["probe"]) &
        (df["k_shot"] == new_row["k_shot"]) &
        (df["feature_type"] == new_row["feature_type"]) &
        (df["split"] == new_row["split"]) &
        (df["aggregation"] == new_row["aggregation"])
    )

    matching_indices = df.index[same_exp]

    # ============================================================
    # CASE 2A: Row exists → update it
    # ============================================================
    if len(matching_indices) > 0:
        idx = matching_indices[0]  # update first match
        df.loc[idx] = new_row
        print(f"[Benchmark] Updated existing row for encoder={new_row['encoder']} "
              f"probe={new_row['probe']} k={new_row['k_shot']}")

    # ============================================================
    # CASE 2B: Row does not exist → append
    # ============================================================
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        print(f"[Benchmark] Added new row → encoder={new_row['encoder']} "
              f"probe={new_row['probe']} k={new_row['k_shot']}")

    # Save updated CSV
    df.to_csv(BENCHMARK_FILE, index=False)
    print(f"[Benchmark] Saved → {BENCHMARK_FILE}")
