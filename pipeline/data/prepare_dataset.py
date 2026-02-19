# data/prepare_dataset.py

import ast
import pandas as pd
from pathlib import Path

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

def _extract_hypertrophy_location_severity(row):
    """
    Extract location and severity of hypertrophy from a given row.

    Parameters
    ----------
    row : pandas.Series
        A single row from a pandas DataFrame containing the data.

    Returns
    -------
    int, str, str
        A tuple containing the presence of hypertrophy, location and severity.
        Presence is 1 if hypertrophy is found, 0 otherwise.
        Location and severity are None if not found.
    """
    if "findings" in row and isinstance(row["findings"], str):
        try:
            findings = ast.literal_eval(row["findings"])
        except Exception:
            return 0, None, None

        if isinstance(findings, list):
            for entry in findings:
                if isinstance(entry, list) and len(entry) >= 3:
                    if isinstance(entry[0], str) and entry[0].lower() == "hypertrophy":
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
            if isinstance(item, str) and "hypertrophy" in item.lower():
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


def _load_metadata(cfg):
    """
    Loads a metadata dataframe from a CSV file and performs the following operations:
        - Filters by the specified organ
        - Extracts the location and severity of hypertrophy from the 'liver_findings_microscopy' column
        - Creates a column 'HasHypertrophy' indicating the presence of hypertrophy
        - Creates a column 'Location' containing the location of hypertrophy
        - Creates a column 'Severity_raw' containing the raw severity of hypertrophy
        - Creates a column 'Severity' containing the normalized severity of hypertrophy

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


    parsed = df.apply(_extract_hypertrophy_location_severity, axis=1)

    df["HasHypertrophy"] = parsed.apply(lambda x: x[0])
    df["Location"] = parsed.apply(lambda x: x[1])
    df["Severity_raw"] = parsed.apply(lambda x: x[2])
    df["Severity"] = df["Severity_raw"].apply(_normalize_severity)

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
          - 'labels': a list of labels (HasHypertrophy)
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

    if ftype == "animal":
        ids = df["subject_organ_UID"].astype(str).tolist()
        features_dir = dirs["animal_dir"]
    else:
        ids = df["slide_id"].astype(str).tolist()
        features_dir = dirs["slide_dir"]

    labels = df["HasHypertrophy"].tolist()

    return {
        "data": {
            "df": df,
            "ids": ids,
            "labels": labels,
            "num_classes": len(set(labels)),
            "severity": df["Severity"].tolist(),
            "location": df["Location"].tolist(),
            "dataset": cfg.datasets.name,

            # directories
            "raw_slide_dir": dirs["raw_slide_dir"],
            "slide_dir": dirs["slide_dir"],
            "animal_dir": dirs["animal_dir"],
            "features_dir": features_dir,

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
        },
    }
