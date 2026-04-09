# data/prepare_dataset.py

import ast
import pandas as pd
from pathlib import Path

from utils.feature_bank_resolver import (
    feature_bank_enabled,
    resolve_prepared_feature_bank,
)


def _normalize_severity(severity):
    """
    Normalize severity from a string or number to an integer.

    If the severity is None, return 0.
    If the severity is a string, strip and convert to lower case.
    If the severity string is in the tg_map, return the corresponding corresponding value.
    If the severity string contains "grade", try to parse an integer from the string.
    If the severity is an integer or float, return the integer value.
    If none of the above conditions are met, return 0.

    Parameters
    ----------
    severity : str, int, float, or None
        The severity to normalize.

    Returns
    -------
    int
        The normalized severity.
    """
    if severity is None:
        return 0

    if isinstance(severity, str):
        s = severity.lower().strip()

        if s.isdigit():
            return int(s)

        tg_map = {
            "minimal": 1,
            "slight": 2,
            "moderate": 3,
            "severe": 4,
        }

        if s in tg_map:
            return tg_map[s]

        if "grade" in s:
            try:
                return int(s.split("grade")[-1].strip())
            except Exception:
                return 0

    if isinstance(severity, (int, float)):
        return int(severity)

    return 0


def _extract_target_location_severity(row, target_finding):
    """
    Extract presence, location, and severity for a configured finding.
    """
    target = str(target_finding).lower().strip()

    if "findings" in row and isinstance(row["findings"], str):
        try:
            findings = ast.literal_eval(row["findings"])
        except Exception:
            return 0, None, None

        if isinstance(findings, list):
            for entry in findings:
                if isinstance(entry, list) and len(entry) >= 3:
                    if isinstance(entry[0], str) and entry[0].lower() == target:
                        return 1, entry[1], entry[2]

        return 0, None, None

    if "liver_findings_microscopy" in row:
        findings = row["liver_findings_microscopy"]

        if isinstance(findings, str):
            try:
                findings = ast.literal_eval(findings)
            except Exception:
                return 0, None, None

        if not isinstance(findings, list):
            return 0, None, None

        for item in findings:
            if isinstance(item, str) and target in item.lower():
                location = None
                severity = None

                if ";" in item:
                    try:
                        location = item.split(";")[1].split(",")[0].strip()
                    except Exception:
                        pass

                if "grade" in item.lower():
                    severity = item.lower().split("grade")[-1].strip()

                return 1, location, severity

        return 0, None, None

    return 0, None, None


def _extract_any_abnormality_label(row):
    """
    Extract a binary "any abnormality present" label from available metadata.
    """
    if "No microscopic finding" in row and pd.notna(row["No microscopic finding"]):
        return (0 if bool(row["No microscopic finding"]) else 1), None, None

    if "findings" in row and isinstance(row["findings"], str):
        try:
            findings = ast.literal_eval(row["findings"])
        except Exception:
            findings = None

        if isinstance(findings, list):
            return (1 if len(findings) > 0 else 0), None, None

    if "liver_findings_microscopy" in row:
        findings = row["liver_findings_microscopy"]

        if isinstance(findings, str):
            try:
                findings = ast.literal_eval(findings)
            except Exception:
                findings = None

        if isinstance(findings, list):
            cleaned = [
                item for item in findings
                if isinstance(item, str) and item.strip() and item.strip().lower() != "nan"
            ]
            return (1 if len(cleaned) > 0 else 0), None, None

    return 0, None, None


def _coerce_label_series(series, positive_value=None):
    """
    Coerce a metadata label column into a binary 0/1 series.
    """
    if positive_value is not None:
        if pd.api.types.is_bool_dtype(series):
            positive = str(positive_value).strip().lower()
            if positive not in {"true", "false", "1", "0"}:
                raise ValueError(
                    "For boolean target columns, cfg.data.target_positive_value "
                    "must be one of: true, false, 1, 0."
                )
            positive_bool = positive in {"true", "1"}
            return (series.fillna(False) == positive_bool).astype(int)

        if pd.api.types.is_numeric_dtype(series):
            try:
                positive_num = float(positive_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "For numeric target columns, cfg.data.target_positive_value "
                    "must be numeric."
                ) from exc
            return (series.fillna(0) == positive_num).astype(int)

        normalized = series.fillna("").astype(str).str.strip().str.lower()
        positive = str(positive_value).strip().lower()
        return (normalized == positive).astype(int)

    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int)

    normalized = series.fillna("").astype(str).str.strip().str.lower()
    true_values = {"1", "true", "yes", "y", "positive", "present", "abnormal"}
    false_values = {"0", "false", "no", "n", "negative", "absent", "normal", ""}

    unknown = set(normalized.unique()) - true_values - false_values
    if unknown:
        raise ValueError(
            "Could not infer binary labels from target column values "
            f"{sorted(unknown)}. Set cfg.data.target_positive_value explicitly."
        )

    return normalized.isin(true_values).astype(int)


def _apply_target_definition(df, cfg):
    """
    Build the generic target columns used by the training pipeline.
    """
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
    """
    Loads a metadata dataframe from a CSV file and performs the following operations:
        - Filters by the specified organ
        - Builds target columns from either a parsed finding or a metadata column
        - Creates generic columns:
          'TargetLabel', 'TargetLocation', 'TargetSeverity_raw', 'TargetSeverity'

    :param cfg: the experiment configuration
    :return: a parsed metadata dataframe
    """
    df = pd.read_csv(cfg.data.metadata_csv)
    # df = pd.read_excel(cfg.data.metadata_csv)

    if "ORGAN" in df.columns:
        df = df[df["ORGAN"].str.lower() == cfg.data.organ.lower()].copy()

    if "subject_organ_UID" not in df.columns and "animal_number" in df.columns:
        df["subject_organ_UID"] = df["animal_number"].astype(str)

    # if "slide_id" not in df.columns and "slide_filename" in df.columns:
    #     df["slide_id"] = df["slide_filename"].astype(str)
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
    """
    Select the split CSV based on the configuration.

    If cfg.datasets.use_subset=True, returns the subset CSV specified in cfg.datasets.subset_csv.
    Otherwise, returns the split CSV specified in cfg.datasets[split].

    Raises
    -------
    ValueError
        If cfg.datasets.use_subset=True but cfg.datasets.subset_csv is missing.
    """
    if cfg.datasets.use_subset:
        if cfg.datasets.subset_csv:
            return cfg.datasets.subset_csv
        raise ValueError("use_subset=True but subset_csv missing")

    return cfg.datasets[cfg.datasets.split]


# def _filter_by_split(df, split_csv, cfg):
#     """
#     Filters a dataframe by selecting samples based on a split CSV.

#     :param df: the dataframe to filter
#     :param split_csv: the split CSV to use for filtering
#     :param cfg: the experiment configuration
#     :return: a filtered dataframe
#     """
#     if split_csv is None:
#         return df

#     split_df = pd.read_csv(split_csv)
#     ftype = cfg.features.feature_type

#     if ftype == "animal":
#         if "subject_organ_UID" not in split_df.columns:
#             raise ValueError("split csv must contain 'subject_organ_UID'")

#         ids = split_df["subject_organ_UID"].astype(str)
#         return df[df["subject_organ_UID"].astype(str).isin(ids)].reset_index(drop=True)

#     elif ftype == "slide":
#         if "slide_id" not in split_df.columns:
#             raise ValueError("split csv must contain 'slide_id'")

#         ids = split_df["slide_id"].astype(str)
#         return df[df["slide_id"].astype(str).isin(ids)].reset_index(drop=True)

#     else:
#         raise ValueError(f"Unknown feature type: {ftype}")

def _filter_by_split(df, split_csv, cfg):
    """
    Filters a dataframe by selecting samples based on a split CSV.
    """
    if split_csv is None:
        return df

    split_df = pd.read_csv(split_csv)
    ftype = cfg.features.feature_type

    if ftype == "animal":

        if "subject_organ_UID" not in split_df.columns:
            if "animal_number" in split_df.columns:
                print(
                    "[WARN] split CSV uses 'animal_number'; "
                    "mapping it to 'subject_organ_UID'"
                )
                split_df["subject_organ_UID"] = split_df["animal_number"].astype(str)
            else:
                raise ValueError(
                    "split csv must contain 'subject_organ_UID' "
                    "or alias 'animal_number'"
                )

        ids = split_df["subject_organ_UID"].astype(str)
        return (
            df[df["subject_organ_UID"].astype(str).isin(ids)]
            .reset_index(drop=True)
        )

    # elif ftype == "slide":

    #     if "slide_id" not in split_df.columns:
    #         raise ValueError("split csv must contain 'slide_id'")

    #     ids = split_df["slide_id"].astype(str)
    #     return df[df["slide_id"].astype(str).isin(ids)].reset_index(drop=True)

    elif ftype == "slide":

        if "slide_filename" in split_df.columns:
            before = len(split_df)
            split_df = split_df[split_df["slide_filename"].notna()].copy()
            dropped = before - len(split_df)
            if dropped > 0:
                print(
                    f"[INFO] Dropped {dropped} split rows with missing slide_filename"
                )

        if "slide_id" not in split_df.columns:
            if "slide_filename" in split_df.columns:
                print(
                    "[WARN] split CSV uses 'slide_filename'; "
                    "mapping it to 'slide_id' (stemmed)"
                )
                split_df["slide_id"] = (
                    split_df["slide_filename"]
                    .astype(str)
                    .apply(lambda x: Path(x).stem)
                )
            else:
                raise ValueError(
                    "split csv must contain 'slide_id' "
                    "or alias 'slide_filename'"
                )

        ids = split_df["slide_id"].astype(str)
        return df[df["slide_id"].astype(str).isin(ids)].reset_index(drop=True)


    else:
        raise ValueError(f"Unknown feature type: {ftype}")

def _apply_subset_fraction(df, cfg):
    """
    Apply a subset fraction to a dataframe.

    If cfg.datasets.use_subset=True and cfg.datasets.subset_fraction is not None,
    sample a fraction of the dataframe and return it.
    Otherwise, return the original dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to sample from.
    cfg : omegaconf.DictConfig
        The configuration dictionary.

    Returns
    -------
    pandas.DataFrame
        The sampled dataframe.
    """
    if cfg.datasets.use_subset and cfg.datasets.subset_fraction is not None:
        return df.sample(frac=cfg.datasets.subset_fraction, random_state=42).reset_index(drop=True)
    return df.reset_index(drop=True)


def _select_feature_dirs(cfg):
    """
    Select feature directories based on config.

    Always returns a single raw_slide_dir.
    Cached dirs may or may not be used downstream.
    """
    return {
        "raw_slide_dir": Path(cfg.data.raw_slide_dir),
        "slide_dir": Path(cfg.data.slide_dir),
        "animal_dir": Path(cfg.data.animal_dir),
    }



def prepare_dataset_inputs(cfg):
    """
    Prepare input data for training or evaluation.

    Returns a dictionary containing:
      - 'data': a dictionary containing:
          - 'df': the filtered metadata dataframe
          - 'ids': a list of slide or animal IDs
          - 'labels': a list of binary target labels
          - 'num_classes': the number of classes
          - 'severity': a list of severity labels
          - 'location': a list of location labels
          - 'raw_slide_dir', 'slide_dir', 'animal_dir': raw, processed slide and animal feature directories
          - 'features_dir': the feature directory to use (slide or animal)
          - 'split', 'subset_csv', 'train_csv', 'val_csv', 'test_csv': dataset split information
      - 'runtime': a dictionary containing:
          - 'batch_size', 'epochs', 'lr', 'optimizer', 'weight_decay', 'momentum', 'loss', 'device', 'num_workers': runtime hyperparameters
      - 'probe': a dictionary containing:
          - 'type', 'hidden_dim', 'num_layers', 'knn_neighbors': probe hyperparameters

    :param cfg: the experiment configuration
    :return: a dictionary containing input data and hyperparameters
    """
    df = _load_metadata(cfg)

    split_csv = _select_split_csv(cfg)
    df = _filter_by_split(df, split_csv, cfg)
    df = _apply_subset_fraction(df, cfg)

    dirs = _select_feature_dirs(cfg)
    ftype = cfg.features.feature_type
    feature_bank_payload = {}

    if ftype == "animal":
        ids = df["subject_organ_UID"].astype(str).tolist()
        features_dir = dirs["animal_dir"]
    else:
        ids = df["slide_id"].astype(str).tolist()
        features_dir = dirs["slide_dir"]

    if feature_bank_enabled(cfg):
        feature_bank_payload = resolve_prepared_feature_bank(cfg, df)

    labels = df["TargetLabel"].tolist()

    return {
        "data": {
            "df": df,
            "ids": ids,
            "labels": labels,
            "num_classes": len(set(labels)),
            "severity": df["TargetSeverity"].tolist(),
            "location": df["TargetLocation"].tolist(),
            "dataset": cfg.datasets.name,
            "target_mode": getattr(cfg.data, "target_mode", "finding"),
            "target_finding": getattr(cfg.data, "target_finding", None),
            "target_column": getattr(cfg.data, "target_column", None),

            # directories
            "raw_slide_dir": dirs["raw_slide_dir"],
            "slide_dir": dirs["slide_dir"],
            "animal_dir": dirs["animal_dir"],
            "features_dir": features_dir,
            **feature_bank_payload,

            # meta
            "split": cfg.datasets.split,
            "subset_csv": cfg.datasets.subset_csv if cfg.datasets.use_subset else None,
            "train_csv": cfg.datasets.train,
            "val_csv": cfg.datasets.val,
            "test_csv": cfg.datasets.test,

            # feature options
            "aggregate": cfg.aggregation.type,
            "embed_dim": cfg.features.embed_dim,
            "features_type": ftype,
            "use_cache": cfg.features.use_cache,
            "d_type": cfg.features.d_type,
        },

        "runtime": {
            "batch_size": cfg.runtime.batch_size,
            "epochs": cfg.runtime.epochs,
            "lr": cfg.runtime.lr,
            "optimizer": cfg.runtime.optimizer,
            "weight_decay": cfg.runtime.weight_decay,
            "momentum": cfg.runtime.momentum,
            "loss": cfg.runtime.loss,
            "device": cfg.runtime.device,
            "num_workers": cfg.runtime.num_workers,
        },

        "probe": {
            "type": cfg.probe.type,
            "hidden_dim": cfg.probe.hidden_dim,
            "num_layers": cfg.probe.num_layers,
            "knn_neighbors": cfg.probe.knn_neighbors,

            # flow specific
            "flow_input_dim": getattr(cfg.probe, "flow_input_dim", None),
            "flow_layers": getattr(cfg.probe, "flow_layers", None),
            "flow_hidden": getattr(cfg.probe, "flow_hidden", None),

            "flow_train_max_tiles": getattr(cfg.probe, "flow_train_max_tiles", None),
            "flow_topk_frac": getattr(cfg.probe, "flow_topk_frac", None),
            "flow_tau_percentile": getattr(cfg.probe, "flow_tau_percentile", None),
            "flow_pca_fit_max_tiles": getattr(cfg.probe, "flow_pca_fit_max_tiles", None),
        },
    }
