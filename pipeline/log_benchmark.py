# pipeline/log_benchmark.py
import pandas as pd
from pathlib import Path

def log_benchmark(cfg, metrics):
    BENCHMARK_FILE = Path(f"outputs/{cfg.aggregation.type}_benchmark_results.csv")

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

    if BENCHMARK_FILE.exists():
        df = pd.read_csv(BENCHMARK_FILE)
        # if "aggregation" not in df.columns:
        #     df["aggregation"] = "mean"
        # check for same experiment
        same_exp = (
            (df["encoder"] == new_row["encoder"]) &
            (df["probe"] == new_row["probe"]) &
            (df["k_shot"] == new_row["k_shot"]) &
            (df["feature_type"] == new_row["feature_type"]) &
            (df["split"] == new_row["split"])
        )
        # if any existing experiment matches, update it
        if same_exp.any():
            df.loc[same_exp, :] = pd.DataFrame([new_row], index=df.index[same_exp])

        # else, append new row
        else:
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # no existing file
    else:
        df = pd.DataFrame([new_row])

    df.to_csv(BENCHMARK_FILE, index=False)
    print(f"[Benchmark] Updated → {BENCHMARK_FILE}")
