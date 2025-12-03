# pipeline/log_benchmark.py
import pandas as pd
from pathlib import Path

BENCHMARK_FILE = Path("outputs/benchmark_results.csv")

def log_benchmark(cfg, metrics):
    row = {
        "encoder": cfg.features.encoder,
        "probe": cfg.probe.type,
        "k_shot": cfg.fewshot.k,
        "features_type": cfg.features.type,   # FIXED
        "split": cfg.datasets.split,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
    }

    if BENCHMARK_FILE.exists():
        df = pd.read_csv(BENCHMARK_FILE)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(BENCHMARK_FILE, index=False)
    print(f"[Benchmark] Updated → {BENCHMARK_FILE}")
