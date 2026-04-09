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
        "target_task": cfg.data.target_task,
        "experiment_tag": cfg.experiment.tag,
        "encoder": cfg.features.encoder,
        "probe": cfg.probe.type,
        "k_shot": cfg.fewshot.k,
        "feature_type": cfg.features.feature_type,
        "split": cfg.datasets.split,
        "calibration_enabled": cfg.calibration.enabled,
        "calibration_samples": cfg.calibration.get("num_samples", None),
        "calibration_seed": cfg.calibration.get("seed", None),
        "batch_size": cfg.runtime.batch_size,
        "epochs": cfg.runtime.epochs,
        "lr": cfg.runtime.lr,
        "weight_decay": cfg.runtime.weight_decay,
        "seed": cfg.runtime.get("seed", 42),
        "flow_input_dim": cfg.probe.get("flow_input_dim", None),
        "flow_layers": cfg.probe.get("flow_layers", None),
        "flow_hidden": cfg.probe.get("flow_hidden", None),
        "flow_train_max_tiles": cfg.probe.get("flow_train_max_tiles", None),
        "flow_topk_frac": cfg.probe.get("flow_topk_frac", None),
        "flow_tau_percentile": cfg.probe.get("flow_tau_percentile", None),
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


    if "target_task" not in df.columns:
        df["target_task"] = "liver_hypertrophy"
    if "experiment_tag" not in df.columns:
        df["experiment_tag"] = "legacy"
    if "calibration_enabled" not in df.columns:
        df["calibration_enabled"] = False
    if "calibration_samples" not in df.columns:
        df["calibration_samples"] = None
    if "calibration_seed" not in df.columns:
        df["calibration_seed"] = None

    same_exp = (
        (df["dataset"] == new_row["dataset"]) &
        (df["target_task"] == new_row["target_task"]) &
        (df["experiment_tag"] == new_row["experiment_tag"]) &
        (df["calibration_enabled"] == new_row["calibration_enabled"]) &
        (df["calibration_samples"].fillna(-1) == (new_row["calibration_samples"] if new_row["calibration_samples"] is not None else -1)) &
        (df["calibration_seed"].fillna(-1) == (new_row["calibration_seed"] if new_row["calibration_seed"] is not None else -1)) &
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
