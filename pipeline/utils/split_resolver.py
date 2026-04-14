from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.create_subset import create_seeded_holdout_subsets, _id_series
from utils.dataset_registry import TEST_ONLY_DATASETS
from utils.feature_bank_resolver import build_registry_from_cfg, MIL_PROBES


def _all_split_files_exist(path: Path) -> bool:
    return all((path / f"{name}.csv").exists() for name in ("train", "val", "test"))


def _apply_calibration_subset_if_needed(
    cfg,
    dataset_key: str,
    data_root: Path,
    generated_root: Path,
    target_task: str,
):
    if not cfg.calibration.enabled:
        return

    num_samples = cfg.calibration.get("num_samples")
    if num_samples is None:
        return

    source_csv = data_root / "UCB" / "ucb_test.csv"
    if not source_csv:
        raise ValueError(
            "Calibration enabled but no calibration source CSV provided. "
            "Set calibration.source_csv or pass --calibration_source_csv."
        )
    seed = int(cfg.calibration.get("seed", 42))
    calibration_dir = (
        generated_root
        / dataset_key
        / "calibration"
        / target_task
        / f"n{num_samples}_seed{seed}"
    )
    label_column = None
    positive_value = None
    target_mode = str(getattr(cfg.data, "target_mode", "")).lower()
    if target_mode == "column":
        label_column = getattr(cfg.data, "target_column", None)
        positive_value = getattr(cfg.data, "target_positive_value", None)
    elif target_mode == "any_abnormality":
        label_column = getattr(cfg.data, "target_column", "No microscopic finding")
        positive_value = getattr(cfg.data, "target_positive_value", False)

    df_source = pd.read_csv(source_csv)
    probe_type = str(getattr(cfg.probe, "type", "")).lower()
    if probe_type in MIL_PROBES:
        if "slide_id" in df_source.columns:
            sample_ids = df_source["slide_id"].astype(str).tolist()
        elif "slide_filename" in df_source.columns:
            sample_ids = df_source["slide_filename"].astype(str).apply(lambda x: Path(x).stem).tolist()
        else:
            sample_ids = _id_series(df_source).tolist()
        sample_type = "slide"
        storage_kind = "raw"
        aggregation = "none"
    else:
        sample_ids = _id_series(df_source).tolist()
        sample_type = str(getattr(cfg.features, "feature_type", "animal")).lower()
        storage_kind = "derived"
        aggregation = str(getattr(cfg.aggregation, "type", "mean")).lower()

    registry = build_registry_from_cfg(cfg)
    entries = registry.resolve_feature_entries(
        dataset="ucb",
        encoder=str(getattr(cfg.features, "encoder", "")).upper(),
        sample_type=sample_type,
        sample_ids=sample_ids,
        storage_kind=storage_kind,
        aggregation=aggregation,
    )
    available_ids = set(entries.keys())

    calibration_train_csv, calibration_test_csv = create_seeded_holdout_subsets(
        source_csv,
        sample_size=num_samples,
        seed=seed,
        train_csv=calibration_dir / "train.csv",
        test_csv=calibration_dir / "test.csv",
        label_column=label_column,
        positive_value=positive_value,
        available_ids=available_ids,
    )
    if dataset_key == "ucb":
        cfg.datasets.train = calibration_train_csv
        cfg.datasets.val = calibration_test_csv
        cfg.datasets.test = calibration_test_csv

        if cfg.datasets.split == "train":
            cfg.datasets.use_subset = True
            cfg.datasets.subset_csv = calibration_train_csv
            return

        if cfg.datasets.split in {"val", "test", "eval"}:
            cfg.datasets.use_subset = True
            cfg.datasets.subset_csv = calibration_test_csv
            return

    if dataset_key == "tggates":
        if cfg.datasets.split == "train":
            cfg.datasets.use_subset = True
            cfg.datasets.subset_csv = calibration_train_csv
            cfg.datasets.train = calibration_train_csv
            cfg.data.metadata_csv = data_root / "UCB" / "metadata.csv"
        return


def resolve_dataset_splits(cfg, dataset_key: str, dataset_folder: str) -> Path:
    data_root = Path(cfg.data.data_root)
    target_task = str(cfg.data.target_task).lower()
    generated_root = Path(cfg.splitting.get("generated_root", "outputs/generated_splits"))
    legacy_split_root = data_root / dataset_folder / "Splits"
    latest_split_root = Path(cfg.data.get("splits_dir") or legacy_split_root)
    generated_target_dir = generated_root / dataset_key / target_task

    # Simple rule 1: liver hypertrophy always uses legacy split root.
    if target_task == "liver_hypertrophy":
        if not _all_split_files_exist(legacy_split_root):
            raise FileNotFoundError(
                f"Missing legacy hypertrophy splits in {legacy_split_root}"
            )
        split_dir = legacy_split_root

    # Simple rule 2: any other task uses latest splits if available, else generated.
    else:
        latest_target_dir = latest_split_root / target_task
        if _all_split_files_exist(latest_target_dir):
            split_dir = latest_target_dir
        elif _all_split_files_exist(latest_split_root):
            split_dir = latest_split_root
        elif _all_split_files_exist(generated_target_dir):
            split_dir = generated_target_dir
        elif dataset_key == "tggates" and target_task == "any_abnormality":
            from split import generate_abnormality_splits

            generate_abnormality_splits(
                cfg.data.metadata_csv,
                cfg.data.organ,
                generated_target_dir,
                num_repeats=cfg.splitting.get("num_repeats", 1000),
            )
            split_dir = generated_target_dir
        else:
            split_dir = None

    if split_dir is None:
        raise FileNotFoundError(
            f"Could not find split CSVs for dataset='{dataset_key}' and target_task='{target_task}'."
        )

    cfg.datasets.train = split_dir / "train.csv"
    cfg.datasets.val = split_dir / "val.csv"
    cfg.datasets.test = split_dir / "test.csv"

    _apply_calibration_subset_if_needed(
        cfg,
        dataset_key,
        data_root,
        generated_root,
        target_task,
    )

    if (
        not cfg.calibration.enabled
        and cfg.datasets.split == "train"
        and cfg.fewshot.get("k") is not None
        and target_task != "liver_hypertrophy"
    ):
        k = int(cfg.fewshot.get("k"))
        seed = int(cfg.runtime.get("seed", 42))
        label_column = None
        positive_value = None
        target_mode = str(getattr(cfg.data, "target_mode", "")).lower()
        if target_mode == "column":
            label_column = getattr(cfg.data, "target_column", None)
            positive_value = getattr(cfg.data, "target_positive_value", None)
        elif target_mode == "any_abnormality":
            label_column = "abnormal"
            positive_value = True

        fewshot_dir = (
            generated_root
            / dataset_key
            / "fewshot"
            / target_task
            / f"k{k}_seed{seed}"
        )
        fewshot_train_csv, _ = create_seeded_holdout_subsets(
            cfg.datasets.train,
            sample_size=k,
            seed=seed,
            train_csv=fewshot_dir / "train.csv",
            test_csv=fewshot_dir / "holdout.csv",
            label_column=label_column,
            positive_value=positive_value,
        )
        cfg.datasets.use_subset = True
        cfg.datasets.subset_csv = fewshot_train_csv
        cfg.datasets.train = fewshot_train_csv

    return split_dir
