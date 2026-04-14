import pandas as pd
from pathlib import Path


PLOT_COLUMNS = [
    "target_task",
    "experiment_tag",
    "encoder",
    "probe",
    "k_shot",
    "calibration_enabled",
    "calibration_samples",
    "calibration_seed",
    "roc_auc",
]

KEY_COLUMNS = [
    "target_task",
    "experiment_tag",
    "encoder",
    "probe",
    "k_shot",
    "calibration_enabled",
    "calibration_samples",
    "calibration_seed",
]

METRIC_COLUMNS = ["roc_auc"]


def _stage_dir(stage: str) -> str:
    return "validation" if stage == "eval" else "testing"


def _benchmark_file(cfg) -> Path:
    return (
        Path("outputs")
        / _stage_dir(str(cfg.stage))
        / cfg.datasets.name
        / f"{cfg.aggregation.type}_benchmark_results.csv"
    )


def _fallback_row(cfg, metrics) -> dict:
    return {
        "target_task": cfg.data.target_task,
        "experiment_tag": cfg.experiment.tag,
        "encoder": cfg.features.encoder,
        "probe": cfg.probe.type,
        "k_shot": cfg.fewshot.k,
        "calibration_enabled": cfg.calibration.enabled,
        "calibration_samples": cfg.calibration.get("num_samples", None),
        "calibration_seed": cfg.calibration.get("seed", None),
        "roc_auc": metrics["roc_auc"],
    }


def _find_registry_row(cfg, registry_path) -> dict | None:
    registry_path = Path(registry_path)
    if not registry_path.exists():
        return None

    df = pd.read_csv(registry_path)
    if df.empty:
        return None

    for col in KEY_COLUMNS + ["aggregation"]:
        if col not in df.columns:
            df[col] = None

    mask = (
        (df["stage"].astype(str) == str(cfg.stage)) &
        (df["dataset"].astype(str) == str(cfg.datasets.name)) &
        (df["target_task"].astype(str) == str(cfg.data.target_task)) &
        (df["experiment_tag"].astype(str) == str(cfg.experiment.tag)) &
        (df["encoder"].astype(str) == str(cfg.features.encoder)) &
        (df["probe"].astype(str) == str(cfg.probe.type)) &
        (df["aggregation"].astype(str) == str(cfg.aggregation.type))
    )
    matches = df.loc[mask]
    if matches.empty:
        return None
    return matches.iloc[-1].to_dict()


def _build_plot_row(cfg, metrics, registry_path) -> dict:
    registry_row = _find_registry_row(cfg, registry_path) if registry_path else None
    row = {col: registry_row.get(col) for col in PLOT_COLUMNS} if registry_row else _fallback_row(cfg, metrics)
    for col in METRIC_COLUMNS:
        row[col] = metrics[col]
    return row


def _same_experiment_mask(df: pd.DataFrame, row: dict) -> pd.Series:
    mask = pd.Series([True] * len(df), index=df.index)
    for col in KEY_COLUMNS:
        target = row.get(col)
        if target is None:
            mask &= df[col].isna()
        else:
            mask &= df[col].astype(str) == str(target)
    return mask


def log_benchmark(cfg, metrics, registry_path=None):
    """
    Logs benchmark results into:

        outputs/<stage>/<dataset>/<aggregation>_benchmark_results.csv

    Stage:
      - eval -> validation benchmarks
      - test -> final test benchmarks
    """

    benchmark_file = _benchmark_file(cfg)
    benchmark_file.parent.mkdir(parents=True, exist_ok=True)

    new_row = _build_plot_row(cfg, metrics, registry_path)

    if not benchmark_file.exists():
        pd.DataFrame([new_row]).to_csv(benchmark_file, index=False)
        print(f"[Benchmark] Created -> {benchmark_file}")
        return

    df = pd.read_csv(benchmark_file)

    for col in PLOT_COLUMNS:
        if col not in df.columns:
            df[col] = None

    matches = df.index[_same_experiment_mask(df, new_row)]

    if len(matches) > 0:
        df.loc[matches[0], PLOT_COLUMNS] = [new_row.get(col) for col in PLOT_COLUMNS]
        print(f"[Benchmark] Updated existing {cfg.stage} result")
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        print(f"[Benchmark] Added new {cfg.stage} result")

    df.to_csv(benchmark_file, index=False)
    print(f"[Benchmark] Saved -> {benchmark_file}")
