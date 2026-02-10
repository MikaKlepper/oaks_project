# data/prepare_dataset.py

import ast
import pandas as pd
from pathlib import Path


# helper to robustly extract Location & Severity for Hypertrophy from the 'findings' string
def _extract_hypertrophy_location_severity(finding_str):
    """
    Parse the 'findings' string and return (location, severity) for Hypertrophy.

    Examples of findings strings:

      "[['Hypertrophy', 'Centrilobular', 'slight', False]]"
      "[['Ground glass appearance', 'Centrilobular', 'minimal', False], "
      " ['Hypertrophy', 'Centrilobular', 'minimal', False]]"

    If no Hypertrophy entry is found or parsing fails → (None, None).
    """
    if not isinstance(finding_str, str):
        return None, None

    try:
        findings = ast.literal_eval(finding_str)
    except Exception:
        return None, None

    if not isinstance(findings, list):
        return None, None

    for entry in findings:
        # Expect list like: ['Hypertrophy', 'Centrilobular', 'slight', False]
        if isinstance(entry, list) and len(entry) >= 3:
            lesion = entry[0]
            if isinstance(lesion, str) and lesion.lower() == "hypertrophy":
                location = entry[1]
                severity = entry[2]
                return location, severity

    return None, None

#  helpers to load metadata, filter by organ, create labels, then filter by split CSV and optional subset fraction
def _load_metadata(cfg):
    """
    Load metadata from CSV and filter by organ.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Configuration object containing data paths and parameters.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the filtered metadata.

    """
    df = pd.read_csv(cfg.data.metadata_csv)
    df = df[df["ORGAN"].str.lower() == cfg.data.organ.lower()].copy()

    # hypertrophy label (binary)
    df["HasHypertrophy"] = df["findings"].str.contains("Hypertrophy", na=False).astype(int)

    # extraction of Location & Severity for hypertrophy
    loc_sev = df["findings"].apply(_extract_hypertrophy_location_severity)
    df["Location"] = loc_sev.apply(lambda x: x[0])
    df["Severity"] = loc_sev.apply(lambda x: x[1])

    # Keep Location/Severity only where Hypertrophy is present
    df["Location"] = df["Location"].where(df["HasHypertrophy"] == 1)
    df["Severity"] = df["Severity"].where(df["HasHypertrophy"] == 1)

    df["subject_organ_UID"] = df["subject_organ_UID"].astype(str)

    print(f"[INFO] Loaded {len(df)} samples from {cfg.data.metadata_csv} for organ {cfg.data.organ}")
    return df


# helper to select the appropriate split CSV based on config
def _select_split_csv(cfg):
    """
    Select the appropriate split CSV based on the pipeline configuration.

    If cfg.datasets.use_subset is True, returns the path to the subset CSV file.
    If cfg.datasets.use_subset is False, returns the path to the full split CSV file specified by cfg.datasets.split.

    Raises
    -------
    ValueError
        If cfg.datasets.use_subset is True but cfg.datasets.subset_csv is missing.
    """
    if cfg.datasets.use_subset:
        if cfg.datasets.subset_csv:
            return cfg.datasets.subset_csv
        raise ValueError("use_subset=True but subset_csv missing")

    # normal full split
    return cfg.datasets[cfg.datasets.split]


# helper to filter by split
def _filter_by_split(df, split_csv, cfg):
    """
    Filter the dataframe by the given split CSV and feature type.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to filter.
    split_csv : str
        The path to the split CSV file or None.
    cfg : omegaconf.DictConfig
        The pipeline configuration.

    Returns
    -------
    pd.DataFrame
        The filtered DataFrame.
    """
    if split_csv is None:
        return df

    split_df = pd.read_csv(split_csv)
    ftype = cfg.features.feature_type  

    if ftype == "animal":
        ids = split_df["subject_organ_UID"].astype(str).tolist()
        return df[df["subject_organ_UID"].astype(str).isin(ids)].reset_index(drop=True)

    elif ftype == "slide":
        ids = split_df["slide_id"].astype(str).tolist()
        return df[df["slide_id"].astype(str).isin(ids)].reset_index(drop=True)

    else:
        raise ValueError(f"Unknown features.features_type: {ftype}")


# helper to apply optional subset fraction to the filtered split
def _apply_subset_fraction(df, cfg):
    """
    Apply an optional subset fraction to the filtered split.

    If cfg.datasets.use_subset=True and cfg.datasets.subset_fraction is not None,
    this function will randomly sample a fraction of the filtered split.

    The random_state parameter is set to 42 for reproducibility.

    Parameters:
    - df (pd.DataFrame): The DataFrame to sample from.
    - cfg (OmegaConf): The pipeline configuration.

    Returns:
    - pd.DataFrame: The sampled DataFrame if cfg.datasets.use_subset=True and
      cfg.datasets.subset_fraction is not None, otherwise the original DataFrame.
    """
    if cfg.datasets.use_subset and cfg.datasets.subset_fraction is not None:
        return df.sample(frac=cfg.datasets.subset_fraction, random_state=42).reset_index(drop=True)
    return df.reset_index(drop=True)


# helper to select feature directories based on config
def _select_feature_dirs(cfg):
    """
    Select feature directories based on config.

    Returns a dictionary with the following keys:
        "raw_slide_dir": Path to the raw slide feature directory
        "slide_dir": Path to the processed slide feature directory
        "animal_dir": Path to the processed animal feature directory

    Parameters:
    - cfg (OmegaConf): The pipeline configuration

    Returns:
    - dict: A dictionary with the selected feature directories
    """
    return {
        "raw_slide_dir": Path(cfg.data.raw_slide_dir),
        "slide_dir":     Path(cfg.data.slide_dir),
        "animal_dir":    Path(cfg.data.animal_dir),
    }

# main function to prepare dataset inputs, called from train.py and eval.py
def prepare_dataset_inputs(cfg):
    """
    Prepare dataset inputs for training, evaluation, and downstream processing.

    Parameters:
    - cfg (OmegaConf): The pipeline configuration.

    Returns:
    - dict: A dictionary containing the prepared dataset, runtime options, and probe options.
    """
    df = _load_metadata(cfg)

    split_csv = _select_split_csv(cfg)
    df = _filter_by_split(df, split_csv, cfg)
    df = _apply_subset_fraction(df, cfg)

    dirs = _select_feature_dirs(cfg)

    ftype = cfg.features.feature_type

    # create list of IDs and labels based on feature type (slide vs animal)
    if ftype == "animal":
        ids = df["subject_organ_UID"].astype(str).tolist()
        features_dir = dirs["animal_dir"]
    else:
        ids = df["slide_id"].astype(str).tolist()
        features_dir = dirs["slide_dir"]

    labels = df["HasHypertrophy"].tolist()

    # final prepared data dictionary with all necessary info for downstream processing, training, and evaluation
    return {
        "data": {
            "df": df,
            "ids": ids,
            "labels": labels,
            "num_classes": len(set(labels)),
            "severity": df["Severity"].tolist(),
            "location": df["Location"].tolist(),

            # DIRS
            "raw_slide_dir": dirs["raw_slide_dir"],
            "slide_dir":     dirs["slide_dir"],
            "animal_dir":    dirs["animal_dir"],

            "features_dir": features_dir,

            # meta
            "split": cfg.datasets.split,
            "subset_csv": cfg.datasets.subset_csv if cfg.datasets.use_subset else None,
            "train_csv": cfg.datasets.train,
            "val_csv":   cfg.datasets.val,
            "test_csv":  cfg.datasets.test,

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
        }
    }
