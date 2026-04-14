import ast
from pathlib import Path

import pandas as pd

from utils.feature_bank_resolver import resolve_prepared_feature_bank

MIL_PROBES = {"abmil", "clam", "dsmil", "flow"}
SEVERITY_MAP = {"minimal": 1, "slight": 2, "moderate": 3, "severe": 4}
TRUE_TOKENS = {"1", "true", "yes", "y", "positive", "present", "abnormal"}
FALSE_TOKENS = {"0", "false", "no", "n", "negative", "absent", "normal", ""}
ID_COLS = {"animal": "subject_organ_UID", "slide": "slide_id"}


def _id_column_for_feature_type(feature_type: str) -> str:
    if feature_type not in ID_COLS:
        raise ValueError(f"Unknown feature type: {feature_type}")
    return ID_COLS[feature_type]


def _active_entries(feature_bank_info: dict) -> dict:
    return feature_bank_info.get("active_feature_entries") or {}


def _token(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    if s.endswith(".0") and s[:-2].isdigit():
        return s[:-2]
    return s


_PARSE_WARNED = set()


def _parse_listlike(value, field_name: str) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            # Some metadata rows contain malformed list strings; treat as empty list.
            if field_name not in _PARSE_WARNED:
                print(f"[WARN] Could not parse list-like field '{field_name}'. Treating as empty list.")
                _PARSE_WARNED.add(field_name)
            return []
        if isinstance(parsed, list):
            return parsed
    raise ValueError(f"Expected list-like '{field_name}', got {type(value).__name__}")


def _normalize_severity(value):
    """Normalize severity value to integer grade."""
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)

    s = _token(value)
    if s.isdigit():
        return int(s)
    if s in SEVERITY_MAP:
        return SEVERITY_MAP[s]
    if "grade" in s:
        tail = s.split("grade", 1)[-1].strip()
        return int(tail) if tail.isdigit() else 0
    return 0


def _extract_target_location_severity(row, target_finding):
    """Extract presence, location and severity for one configured finding."""
    target = _token(target_finding)

    if "findings" in row and pd.notna(row["findings"]):
        for entry in _parse_listlike(row["findings"], "findings"):
            if isinstance(entry, list) and len(entry) >= 3 and _token(entry[0]) == target:
                return 1, entry[1], entry[2]
        return 0, None, None

    if "liver_findings_microscopy" in row and pd.notna(row["liver_findings_microscopy"]):
        for item in _parse_listlike(row["liver_findings_microscopy"], "liver_findings_microscopy"):
            if not isinstance(item, str) or target not in _token(item):
                continue
            location = item.split(";")[1].split(",")[0].strip() if ";" in item else None
            severity = _token(item).split("grade", 1)[-1].strip() if "grade" in _token(item) else None
            return 1, location, severity
        return 0, None, None

    return 0, None, None


def _extract_any_abnormality_label(row):
    """Extract binary any-abnormality label from available metadata."""
    if "No microscopic finding" in row and pd.notna(row["No microscopic finding"]):
        return (0 if _token(row["No microscopic finding"]) in TRUE_TOKENS else 1), None, None

    if "findings" in row and pd.notna(row["findings"]):
        return (1 if len(_parse_listlike(row["findings"], "findings")) > 0 else 0), None, None

    if "liver_findings_microscopy" in row and pd.notna(row["liver_findings_microscopy"]):
        cleaned = [x for x in _parse_listlike(row["liver_findings_microscopy"], "liver_findings_microscopy") if _token(x) not in {"", "nan"}]
        return (1 if len(cleaned) > 0 else 0), None, None

    return 0, None, None


def _coerce_label_series(series, positive_value=None):
    """Coerce one metadata column to binary 0/1 labels."""
    normalized = series.map(_token)
    if positive_value is not None:
        return (normalized == _token(positive_value)).astype(int)

    unknown = set(normalized.unique()) - TRUE_TOKENS - FALSE_TOKENS
    if unknown:
        raise ValueError(
            "Could not infer binary labels from target column values "
            f"{sorted(unknown)}. Set cfg.data.target_positive_value explicitly."
        )

    return normalized.isin(TRUE_TOKENS).astype(int)


def _apply_target_definition(df, cfg):
    """Build generic target columns used by train/eval."""
    target_mode = str(getattr(cfg.data, "target_mode", "finding")).lower()

    if target_mode == "finding":
        target_finding = getattr(cfg.data, "target_finding", "hypertrophy")
        parsed = df.apply(
            lambda row: _extract_target_location_severity(row, target_finding),
            axis=1,
        )
        df["TargetLabel"] = parsed.apply(lambda x: x[0]).astype(int)
        df["TargetLocation"] = parsed.apply(lambda x: x[1])
        df["TargetSeverity_raw"] = parsed.apply(lambda x: x[2])
        df["TargetSeverity"] = df["TargetSeverity_raw"].apply(_normalize_severity)

        if str(target_finding).lower() == "hypertrophy":
            df["HasHypertrophy"] = df["TargetLabel"]
            df["Location"] = df["TargetLocation"]
            df["Severity_raw"] = df["TargetSeverity_raw"]
            df["Severity"] = df["TargetSeverity"]

        return df

    if target_mode == "column":
        target_column = getattr(cfg.data, "target_column", None)
        if not target_column:
            raise ValueError(
                "cfg.data.target_column must be set when cfg.data.target_mode='column'"
            )
        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in metadata CSV"
            )

        positive_value = getattr(cfg.data, "target_positive_value", None)
        df["TargetLabel"] = _coerce_label_series(df[target_column], positive_value)
        df["TargetLocation"] = None
        df["TargetSeverity_raw"] = None
        df["TargetSeverity"] = 0
        return df

    if target_mode == "any_abnormality":
        parsed = df.apply(_extract_any_abnormality_label, axis=1)
        df["TargetLabel"] = parsed.apply(lambda x: x[0]).astype(int)
        df["TargetLocation"] = None
        df["TargetSeverity_raw"] = None
        df["TargetSeverity"] = 0
        return df

    raise ValueError(
        f"Unsupported cfg.data.target_mode='{target_mode}'. "
        "Use 'finding', 'column', or 'any_abnormality'."
    )


def _load_metadata(cfg):
    """Load metadata CSV, normalize IDs, and attach target columns."""
    df = pd.read_csv(cfg.data.metadata_csv)

    if "ORGAN" in df.columns:
        df = df[df["ORGAN"].str.lower() == cfg.data.organ.lower()].copy()

    if "subject_organ_UID" not in df.columns and "animal_number" in df.columns:
        df["subject_organ_UID"] = df["animal_number"].astype(str)

    if "slide_filename" in df.columns:
        before = len(df)
        df = df[df["slide_filename"].notna()].copy()
        dropped = before - len(df)
        if dropped > 0:
            print(f"[INFO] Dropped {dropped} samples with missing slide_filename")
    # Ensure canonical slide_id exists (works for TG-GATES and UCB)
    if "slide_id" not in df.columns:
        if "slide_filename" in df.columns:
            df["slide_id"] = (
                df["slide_filename"]
                .astype(str)
                .apply(lambda x: Path(x).stem)
            )
            print("[INFO] Using 'slide_filename' as slide_id (stemmed)")
        else:
            raise ValueError(
                "Metadata must contain either 'slide_id' or 'slide_filename'"
            )
    df = _apply_target_definition(df, cfg)

    print(f"[INFO] Loaded {len(df)} samples from {cfg.data.metadata_csv}")
    return df

def _select_split_csv(cfg):
    """Return active split/subset CSV path for current run."""
    if cfg.datasets.use_subset:
        if cfg.datasets.subset_csv:
            return cfg.datasets.subset_csv
        raise ValueError("use_subset=True but subset_csv missing")

    return cfg.datasets[cfg.datasets.split]


def _filter_by_split(df, split_csv, cfg):
    """Filter metadata rows to IDs defined in the split/subset CSV."""
    if split_csv is None:
        return df

    split_df = pd.read_csv(split_csv)
    ftype = str(cfg.features.feature_type).lower()
    id_col = _id_column_for_feature_type(ftype)

    if ftype == "animal":
        if id_col not in split_df.columns and "animal_number" in split_df.columns:
            print("[WARN] split CSV uses 'animal_number'; mapping it to 'subject_organ_UID'")
            split_df[id_col] = split_df["animal_number"].astype(str)
    else:
        if "slide_filename" in split_df.columns:
            before = len(split_df)
            split_df = split_df[split_df["slide_filename"].notna()].copy()
            dropped = before - len(split_df)
            if dropped > 0:
                print(f"[INFO] Dropped {dropped} split rows with missing slide_filename")
        if id_col not in split_df.columns and "slide_filename" in split_df.columns:
            print("[WARN] split CSV uses 'slide_filename'; mapping it to 'slide_id' (stemmed)")
            split_df["slide_id"] = split_df["slide_filename"].astype(str).apply(lambda x: Path(x).stem)

    if id_col not in split_df.columns:
        if ftype == "animal":
            raise ValueError("split csv must contain 'subject_organ_UID' or alias 'animal_number'")
        raise ValueError("split csv must contain 'slide_id' or alias 'slide_filename'")

    ids = split_df[id_col].astype(str)
    return df[df[id_col].astype(str).isin(ids)].reset_index(drop=True)

def _apply_subset_fraction(df, cfg):
    """Optionally subsample selected rows by fraction."""
    frac = cfg.datasets.subset_fraction if cfg.datasets.use_subset else None
    if frac is not None:
        seed = cfg.runtime.get("seed", 42)
        return df.sample(frac=frac, random_state=seed).reset_index(drop=True)
    return df.reset_index(drop=True)


def _drop_missing_feature_rows(df, cfg, feature_bank_info):
    probe_type = str(cfg.probe.type).lower()
    feature_type = str(cfg.features.feature_type).lower()
    mode = "raw" if probe_type in MIL_PROBES else "derived"
    id_column = "slide_id" if mode == "raw" else _id_column_for_feature_type(feature_type)
    original_rows = len(df)
    available_ids = set(_active_entries(feature_bank_info).keys())
    missing_ids = feature_bank_info.get("missing_required_ids") or []
    if missing_ids:
        print(
            f"[WARN] Missing {len(missing_ids)} required {mode} feature entries; "
            f"continuing with available samples only. First missing: {missing_ids[:10]}"
        )

    filtered = df[df[id_column].astype(str).isin(available_ids)].copy()
    dropped = original_rows - len(filtered)
    if feature_type == "animal" and mode == "raw":
        before_animals = df["subject_organ_UID"].astype(str).nunique()
        after_animals = filtered["subject_organ_UID"].astype(str).nunique()
        if before_animals - after_animals > 0:
            print(f"[WARN] Dropped {before_animals - after_animals} animals with no remaining raw slide features")
    if dropped > 0:
        print(f"[WARN] Dropped {dropped} metadata rows without available {mode} features")
    return filtered.reset_index(drop=True)


def _runtime_config(cfg) -> dict:
    return {
        "batch_size": cfg.runtime.batch_size,
        "epochs": cfg.runtime.epochs,
        "lr": cfg.runtime.lr,
        "optimizer": cfg.runtime.optimizer,
        "weight_decay": cfg.runtime.weight_decay,
        "momentum": cfg.runtime.momentum,
        "loss": cfg.runtime.loss,
        "device": cfg.runtime.device,
        "num_workers": cfg.runtime.num_workers,
    }


def _probe_config(cfg) -> dict:
    return {
        "type": cfg.probe.type,
        "hidden_dim": cfg.probe.hidden_dim,
        "num_layers": cfg.probe.num_layers,
        "knn_neighbors": cfg.probe.knn_neighbors,
        "flow_input_dim": getattr(cfg.probe, "flow_input_dim", None),
        "flow_layers": getattr(cfg.probe, "flow_layers", None),
        "flow_hidden": getattr(cfg.probe, "flow_hidden", None),
        "flow_train_max_tiles": getattr(cfg.probe, "flow_train_max_tiles", None),
        "flow_topk_frac": getattr(cfg.probe, "flow_topk_frac", None),
        "flow_tau_percentile": getattr(cfg.probe, "flow_tau_percentile", None),
        "flow_pca_fit_max_tiles": getattr(cfg.probe, "flow_pca_fit_max_tiles", None),
    }


def _data_bundle(cfg, df, ids, labels, feature_bank_info) -> dict:
    target_mode = str(getattr(cfg.data, "target_mode", "finding")).lower()
    num_classes = 2 if target_mode in {"finding", "column", "any_abnormality"} else len(set(labels))
    return {
        "df": df,
        "ids": ids,
        "labels": labels,
        "num_classes": num_classes,
        "severity": df["TargetSeverity"].tolist(),
        "location": df["TargetLocation"].tolist(),
        "dataset": cfg.datasets.name,
        "target_mode": target_mode,
        "target_finding": getattr(cfg.data, "target_finding", None),
        "target_column": getattr(cfg.data, "target_column", None),
        **feature_bank_info,
        "split": cfg.datasets.split,
        "subset_csv": cfg.datasets.subset_csv if cfg.datasets.use_subset else None,
        "train_csv": cfg.datasets.train,
        "val_csv": cfg.datasets.val,
        "test_csv": cfg.datasets.test,
        "aggregate": cfg.aggregation.type,
        "embed_dim": cfg.features.embed_dim,
        "features_type": cfg.features.feature_type,
        "use_cache": cfg.features.use_cache,
        "d_type": cfg.features.d_type,
    }



def prepare_dataset_inputs(cfg):
    """Prepare final data/runtime/probe config consumed by train/eval."""
    df = _load_metadata(cfg)

    split_csv = _select_split_csv(cfg)
    df = _filter_by_split(df, split_csv, cfg)
    df = _apply_subset_fraction(df, cfg)

    ftype = cfg.features.feature_type
    feature_bank_info = resolve_prepared_feature_bank(cfg, df)
    df = _drop_missing_feature_rows(df, cfg, feature_bank_info)

    if df.empty:
        raise RuntimeError(
            "No samples remain after applying split filters and feature-bank availability checks."
        )

    feature_bank_info = resolve_prepared_feature_bank(cfg, df)

    if ftype == "animal":
        ids = df["subject_organ_UID"].astype(str).tolist()
    else:
        ids = df["slide_id"].astype(str).tolist()

    labels = df["TargetLabel"].tolist()

    return {
        "data": _data_bundle(cfg, df, ids, labels, feature_bank_info),
        "runtime": _runtime_config(cfg),
        "probe": _probe_config(cfg),
    }
