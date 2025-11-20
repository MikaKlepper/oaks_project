import pandas as pd
from pathlib import Path

def _load_metadata(cfg):
    
    """
    Load the metadata CSV and filter it by organ.

    Args:
        cfg (dict): Configuration dictionary containing the data section.

    Returns:
        pd.DataFrame: Filtered metadata dataframe.
    """
    df = pd.read_csv(cfg.data.metadata_csv)
    df = df[df["ORGAN"].str.lower() == cfg.data.organ.lower()].copy()

    df["HasHypertrophy"] = df["findings"].str.contains("Hypertrophy", na=False).astype(int)

    df[["Location", "Severity"]] = df["findings"].str.extract(
        r"\['Hypertrophy'\s*,\s*'([^']+)'\s*,\s*'([^']+)'"
    )
    df["Location"] = df["Location"].where(df["HasHypertrophy"] == 1)
    df["Severity"] = df["Severity"].where(df["HasHypertrophy"] == 1)

    df["subject_organ_UID"] = df["subject_organ_UID"].astype(str)
    print(f"[INFO] Loaded {len(df)} samples from {cfg.data.metadata_csv} for organ {cfg.data.organ}")
    return df


def _select_split_csv(cfg):
    
    """
    Select the correct split CSV based on the configuration.

    If use_subset is True, it will check if a subset CSV is provided. If not, it will
    use the subset fraction to select the correct split CSV from the datasets config.
    If use_subset is False, it will directly return the split CSV from the datasets config.

    Raises:
        ValueError: If use_subset is True and no subset_csv or subset_fraction is provided.
    """
    if cfg.datasets.use_subset:
        if cfg.datasets.subset_csv:
            return cfg.datasets.subset_csv
        elif cfg.datasets.subset_fraction is not None:
            return cfg.datasets[cfg.datasets.split]
        else:
            raise ValueError("use_subset=True but no subset_csv or subset_fraction provided")
    else:
        return cfg.datasets[cfg.datasets.split]


def _filter_by_split(df, split_csv):
    """
    Filter the dataframe to only include the animal-level IDs specified in the split CSV.

    If the split CSV is None, the dataframe is returned as is. Otherwise, the dataframe
    is filtered to only include the animal-level IDs specified in the split CSV.

    Args:
        df (pd.DataFrame): The dataframe to filter
        split_csv (str | None): Path to the split CSV file (optional)

    Returns:
        pd.DataFrame: The filtered dataframe
    """
    if split_csv is None:
        return df
    ids = pd.read_csv(split_csv)["subject_organ_UID"].astype(str).tolist()
    return df[df["subject_organ_UID"].astype(str).isin(ids)].reset_index(drop=True)



def _apply_subset_fraction(df, cfg):
    """
    Apply a fractional subset to the dataframe if specified.

    If `cfg.datasets.use_subset` is True and `cfg.datasets.subset_fraction` is not None,
    the dataframe is sampled at the specified fraction, and the number of samples
    before and after the subset is printed.

    Args:
        df (pd.DataFrame): The dataframe to subset
        cfg (DictConfig): The OAKS configuration

    Returns:
        pd.DataFrame: The subset dataframe
    """
    if cfg.datasets.use_subset and cfg.datasets.subset_fraction is not None:
        return df.sample(frac=cfg.datasets.subset_fraction, random_state=42).reset_index(drop=True)
    return df.reset_index(drop=True)


def _extract_features_dir(df, cfg):
   
    """
    Extract the feature directory for the given split.

    Args:
        df (pd.DataFrame): The dataframe containing animal-level IDs.
        cfg (DictConfig): The OAKS configuration.

    Returns:
        Path: The feature directory for the given split.

    Raises:
        ValueError: If the split is invalid or feature files are missing.
    """
    split = cfg.datasets.split

    feature_dirs = {
        "train": cfg.features.train_animal_dir,
        "val":   cfg.features.val_animal_dir,
        "test":  cfg.features.test_animal_dir,
    }

    if split not in feature_dirs:
        raise ValueError(f"Invalid split: {split}")
    fdir = Path(feature_dirs[split])
    # available = {p.stem for p in fdir.glob("*.pt")}
    # expected = set(df["subject_organ_UID"].astype(str))

    # missing = expected - available
    # if missing:
    #     raise ValueError(f"Missing {len(missing)} feature files in {fdir}")
       

    return fdir

def _select_slide_dir(cfg):
    split = cfg.datasets.split
    slide_dirs = {
        "train": cfg.features.train_slide_dir,
        "val":   cfg.features.val_slide_dir,
        "test":  cfg.features.test_slide_dir,
        
    }
    if split not in slide_dirs:
        raise ValueError(f"Invalid split '{split}'. Expected train/val/test.")
    return Path(slide_dirs[split])

def prepare_dataset_inputs(cfg):
  
    """
    Prepare dataset inputs for the model.

    This function loads the metadata, filters by split, applies the subset fraction,
    validates the existence of feature files, and selects the slide directory.

    Args:
        cfg (DictConfig): The OAKS configuration.

    Returns:
        dict: A dictionary containing the dataframe, animal IDs, labels, slide directory,
            feature directory, and aggregation type.
    """
    df = _load_metadata(cfg)
    split_csv = _select_split_csv(cfg)
    df = _filter_by_split(df, split_csv)
    df = _apply_subset_fraction(df, cfg)
    features_dir = _extract_features_dir(df, cfg)

    slide_dir = _select_slide_dir(cfg)

    animal_ids = df["subject_organ_UID"].tolist()
    labels = df["HasHypertrophy"].tolist()

    return {
        "df": df,
        "animal_ids": animal_ids,
        "labels": labels,   
        "slide_dir": slide_dir,
        "features_dir": features_dir,
        "split": cfg.datasets.split,
        "aggregate": cfg.aggregation.type,
        "embed_dim":cfg.features.embed_dim
    }
