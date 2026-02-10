# pipeline/log_benchmark.py

import pandas as pd
from pathlib import Path


def log_benchmark(cfg, metrics):
    """
    Logs benchmark results into:

        outputs/<stage>/<aggregation>_benchmark_results.csv

    Stage:
      - eval -> validation benchmarks
      - test -> final test benchmarks
    """

    benchmark_file = (
        Path("outputs")
        / cfg.stage
        / f"{cfg.aggregation.type}_benchmark_results.csv"
    )
    benchmark_file.parent.mkdir(parents=True, exist_ok=True)

    new_row = {
        "encoder": cfg.features.encoder,
        "probe": cfg.probe.type,
        "k_shot": cfg.fewshot.k,
        "feature_type": cfg.features.feature_type,
        "split": cfg.datasets.split,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
    }

    # If benchmark file doesn't exist, create it with the new row
    if not benchmark_file.exists():
        pd.DataFrame([new_row]).to_csv(benchmark_file, index=False)
        print(f"[Benchmark] Created → {benchmark_file}")
        return

    df = pd.read_csv(benchmark_file)

    # check if there's already an entry for the same encoder, probe, k_shot, and feature_type
    same_exp = (
        (df["encoder"] == new_row["encoder"]) &
        (df["probe"] == new_row["probe"]) &
        (df["k_shot"] == new_row["k_shot"]) &
        (df["feature_type"] == new_row["feature_type"])
    )

    matches = df.index[same_exp]

    # if there's already an entry, update it; otherwise, append a new row
    if len(matches) > 0:
        df.loc[matches[0]] = new_row
        print(f"[Benchmark] Updated existing {cfg.stage} result")
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        print(f"[Benchmark] Added new {cfg.stage} result")

    df.to_csv(benchmark_file, index=False)
    print(f"[Benchmark] Saved → {benchmark_file}")
