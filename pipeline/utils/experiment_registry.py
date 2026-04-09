from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


EXPERIMENT_KEY_COLUMNS = [
    "stage",
    "dataset",
    "target_task",
    "experiment_tag",
    "encoder",
    "probe",
    "k_shot",
    "feature_type",
    "split",
    "calibration_enabled",
    "calibration_samples",
    "calibration_seed",
]


def _registry_path(cfg) -> Path:
    custom = cfg.get("experiment_registry", {}).get("path")
    if custom:
        return Path(str(custom))
    return Path("outputs") / "registry" / "experiment_runs.csv"


def _artifact_counts(prepared) -> tuple[int | None, int | None]:
    data = prepared["data"]
    raw_artifacts = data.get("raw_feature_artifacts")
    derived_artifacts = data.get("feature_artifacts")
    raw_count = len(raw_artifacts) if raw_artifacts is not None else None
    derived_count = len(derived_artifacts) if derived_artifacts is not None else None
    return raw_count, derived_count


def build_experiment_row(
    cfg,
    prepared,
    *,
    stage: str,
    status: str,
    exp_root: Path,
    metrics: dict | None = None,
    checkpoint_path: Path | None = None,
    metrics_path: Path | None = None,
) -> dict:
    data = prepared["data"]
    runtime = prepared["runtime"]
    probe = prepared["probe"]
    raw_count, derived_count = _artifact_counts(prepared)

    row = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "status": status,
        "dataset": cfg.datasets.name,
        "train_dataset": cfg.datasets.get("train_name", cfg.datasets.name),
        "target_task": cfg.data.target_task,
        "target_mode": cfg.data.get("target_mode"),
        "target_finding": cfg.data.get("target_finding"),
        "target_column": cfg.data.get("target_column"),
        "experiment_tag": cfg.experiment.tag,
        "experiment_root": str(exp_root),
        "encoder": cfg.features.encoder,
        "probe": cfg.probe.type,
        "feature_type": cfg.features.feature_type,
        "feature_backend": data.get("feature_backend", "legacy"),
        "aggregation": cfg.aggregation.type,
        "split": cfg.datasets.split,
        "subset_csv": str(data.get("subset_csv")) if data.get("subset_csv") else None,
        "train_csv": str(data.get("train_csv")) if data.get("train_csv") else None,
        "val_csv": str(data.get("val_csv")) if data.get("val_csv") else None,
        "test_csv": str(data.get("test_csv")) if data.get("test_csv") else None,
        "registry_path": data.get("feature_registry_path"),
        "feature_bank_root": data.get("feature_bank_root"),
        "feature_bank_local_root": data.get("feature_bank_local_root"),
        "raw_slide_dir": str(data.get("raw_slide_dir")) if data.get("raw_slide_dir") else None,
        "slide_dir": str(data.get("slide_dir")) if data.get("slide_dir") else None,
        "animal_dir": str(data.get("animal_dir")) if data.get("animal_dir") else None,
        "resolved_raw_artifact_count": raw_count,
        "resolved_feature_artifact_count": derived_count,
        "num_samples": len(data.get("ids", [])),
        "num_classes": data.get("num_classes"),
        "embed_dim": data.get("embed_dim"),
        "k_shot": cfg.fewshot.get("k"),
        "batch_size": runtime.get("batch_size"),
        "epochs": runtime.get("epochs"),
        "lr": runtime.get("lr"),
        "optimizer": runtime.get("optimizer"),
        "weight_decay": runtime.get("weight_decay"),
        "momentum": runtime.get("momentum"),
        "loss": runtime.get("loss"),
        "seed": runtime.get("seed", 42),
        "hidden_dim": probe.get("hidden_dim"),
        "num_layers": probe.get("num_layers"),
        "knn_neighbors": probe.get("knn_neighbors"),
        "flow_input_dim": probe.get("flow_input_dim"),
        "flow_layers": probe.get("flow_layers"),
        "flow_hidden": probe.get("flow_hidden"),
        "flow_train_max_tiles": probe.get("flow_train_max_tiles"),
        "flow_topk_frac": probe.get("flow_topk_frac"),
        "flow_tau_percentile": probe.get("flow_tau_percentile"),
        "flow_pca_fit_max_tiles": probe.get("flow_pca_fit_max_tiles"),
        "calibration_enabled": bool(cfg.calibration.get("enabled", False)),
        "calibration_samples": cfg.calibration.get("num_samples"),
        "calibration_seed": cfg.calibration.get("seed"),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "metrics_path": str(metrics_path) if metrics_path else None,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "roc_auc": None,
    }
    if metrics is not None:
        row.update(
            {
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "roc_auc": metrics.get("roc_auc"),
            }
        )
    return row


def append_experiment_row(
    cfg,
    prepared,
    *,
    stage: str,
    status: str,
    exp_root: Path,
    metrics: dict | None = None,
    checkpoint_path: Path | None = None,
    metrics_path: Path | None = None,
) -> Path:
    row = build_experiment_row(
        cfg,
        prepared,
        stage=stage,
        status=status,
        exp_root=exp_root,
        metrics=metrics,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
    )
    out_path = _registry_path(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        df = pd.read_csv(out_path)
    else:
        df = pd.DataFrame(columns=row.keys())

    for col in row.keys():
        if col not in df.columns:
            df[col] = None

    key_mask = pd.Series([True] * len(df))
    for col in EXPERIMENT_KEY_COLUMNS:
        target = row[col]
        if target is None:
            key_mask &= df[col].isna()
        else:
            key_mask &= df[col].astype(str) == str(target)

    matches = df.index[key_mask]
    if len(matches) > 0:
        df.loc[matches[0], list(row.keys())] = list(row.values())
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(out_path, index=False)
    return out_path


def experiment_run_exists(
    *,
    stage: str,
    dataset: str,
    target_task: str,
    experiment_tag: str,
    encoder: str,
    probe: str,
    k_shot: int | None,
    aggregation: str,
    calibration_enabled: bool,
    calibration_samples: int | None,
    calibration_seed: int | None,
    feature_type: str = "animal",
    registry_path: str | Path = "outputs/registry/experiment_runs.csv",
    accepted_statuses: tuple[str, ...] = ("completed", "checkpoint_exists"),
) -> bool:
    registry_path = Path(registry_path)
    if not registry_path.exists():
        return False

    df = pd.read_csv(registry_path)
    if df.empty:
        return False

    required_defaults = {
        "target_task": "liver_hypertrophy",
        "experiment_tag": "legacy",
        "feature_type": "animal",
        "aggregation": "mean",
        "calibration_enabled": False,
        "calibration_samples": None,
        "calibration_seed": None,
        "status": None,
    }
    for col, default in required_defaults.items():
        if col not in df.columns:
            df[col] = default

    mask = (
        (df["stage"].astype(str) == str(stage)) &
        (df["dataset"].astype(str) == str(dataset)) &
        (df["target_task"].astype(str) == str(target_task)) &
        (df["experiment_tag"].astype(str) == str(experiment_tag)) &
        (df["encoder"].astype(str) == str(encoder)) &
        (df["probe"].astype(str) == str(probe)) &
        (df["feature_type"].astype(str) == str(feature_type)) &
        (df["aggregation"].astype(str) == str(aggregation)) &
        (df["calibration_enabled"].fillna(False).astype(bool) == bool(calibration_enabled))
    )

    if k_shot is None:
        mask &= df["k_shot"].isna()
    else:
        mask &= df["k_shot"].fillna(-1).astype(int) == int(k_shot)

    if calibration_samples is None:
        mask &= df["calibration_samples"].isna()
    else:
        mask &= df["calibration_samples"].fillna(-1).astype(int) == int(calibration_samples)

    if calibration_seed is None:
        mask &= df["calibration_seed"].isna()
    else:
        mask &= df["calibration_seed"].fillna(-1).astype(int) == int(calibration_seed)

    if accepted_statuses:
        mask &= df["status"].astype(str).isin(list(accepted_statuses))

    return bool(mask.any())
