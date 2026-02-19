# pipeline/log_benchmark.py

import pandas as pd
from pathlib import Path


def log_benchmark(cfg, metrics):
    """
    Logs benchmark results into:

        outputs/<stage>/<dataset>/<aggregation>_benchmark_results.csv

    Stage:
      - eval -> validation benchmarks
      - test -> final test benchmarks
    """

    dataset = cfg.datasets.name

    benchmark_file = (
        Path("outputs")
        / cfg.stage
        / dataset
        / f"{cfg.aggregation.type}_benchmark_results.csv"
    )
    benchmark_file.parent.mkdir(parents=True, exist_ok=True)

    new_row = {
        "dataset": dataset,                     
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

    if not benchmark_file.exists():
        pd.DataFrame([new_row]).to_csv(benchmark_file, index=False)
        print(f"[Benchmark] Created -> {benchmark_file}")
        return

    df = pd.read_csv(benchmark_file)


    same_exp = (
        (df["dataset"] == new_row["dataset"]) &
        (df["encoder"] == new_row["encoder"]) &
        (df["probe"] == new_row["probe"]) &
        (df["k_shot"] == new_row["k_shot"]) &
        (df["feature_type"] == new_row["feature_type"])
    )

    matches = df.index[same_exp]

    if len(matches) > 0:
        df.loc[matches[0]] = new_row
        print(f"[Benchmark] Updated existing {cfg.stage} result")
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        print(f"[Benchmark] Added new {cfg.stage} result")

    df.to_csv(benchmark_file, index=False)
    print(f"[Benchmark] Saved -> {benchmark_file}")
