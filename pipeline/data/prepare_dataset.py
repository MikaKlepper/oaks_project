import pandas as pd
from pathlib import Path


# ============================================================
# METADATA LOADING
# ============================================================

def _load_metadata(cfg):
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


# ============================================================
# SPLIT CSV SELECTION
# ============================================================

def _select_split_csv(cfg):
    if cfg.datasets.use_subset:
        if cfg.datasets.subset_csv:
            return cfg.datasets.subset_csv
        else:
            raise ValueError("use_subset=True but no subset_csv provided")
    else:
        return cfg.datasets[cfg.datasets.split]


# ============================================================
# SPLIT FILTERING (ANIMAL or SLIDE)
# ============================================================

def _filter_by_split(df, split_csv, cfg):
    if split_csv is None:
        return df

    split_df = pd.read_csv(split_csv)
    ftype = cfg.features.type  # "animal" or "slide"

    if ftype == "animal":
        ids = split_df["subject_organ_UID"].astype(str).tolist()
        return df[df["subject_organ_UID"].astype(str).isin(ids)].reset_index(drop=True)

    elif ftype == "slide":
        ids = split_df["slide_id"].astype(str).tolist()
        return df[df["slide_id"].astype(str).isin(ids)].reset_index(drop=True)

    else:
        raise ValueError(f"Unknown features.type: {ftype}")


# ============================================================
# FRACTIONAL SUBSET
# ============================================================

def _apply_subset_fraction(df, cfg):
    if cfg.datasets.use_subset and cfg.datasets.subset_fraction is not None:
        return df.sample(frac=cfg.datasets.subset_fraction, random_state=42).reset_index(drop=True)
    return df.reset_index(drop=True)


# ============================================================
# NEW DIRECTORY SELECTION
# ============================================================

def _select_feature_dirs(cfg):
    """
    Uses the new architecture: config_loader already sets:

        cfg.features.slide_dir
        cfg.features.animal_dir

    to the correct split-specific experiment folders.
    """
    return {
        "slide_dir": Path(cfg.features.slide_dir),
        "animal_dir": Path(cfg.features.animal_dir),
    }


# ============================================================
# MAIN: PREPARE DATASET INPUTS
# ============================================================

def prepare_dataset_inputs(cfg):
    df = _load_metadata(cfg)

    split_csv = _select_split_csv(cfg)
    df = _filter_by_split(df, split_csv, cfg)
    df = _apply_subset_fraction(df, cfg)

    dirs = _select_feature_dirs(cfg)

    # ID selection
    if cfg.features.type == "animal":
        ids = df["subject_organ_UID"].tolist()
        features_dir = dirs["animal_dir"]
    else:
        ids = df["slide_id"].tolist()
        features_dir = dirs["slide_dir"]

    labels = df["HasHypertrophy"].tolist()

    # FINAL RETURN STRUCTURE
    return {
        "data": {
            "df": df,
            "ids": ids,
            "labels": labels,
            "num_classes": len(set(labels)),

            "slide_dir": dirs["slide_dir"],
            "features_dir": features_dir,

            "split": cfg.datasets.split,
            "subset_csv": cfg.datasets.subset_csv if cfg.datasets.use_subset else None,

            "train_csv": cfg.datasets.train,
            "val_csv": cfg.datasets.val,
            "test_csv": cfg.datasets.test,

            "aggregate": cfg.aggregation.type,
            "embed_dim": cfg.features.embed_dim,
            "features_type": cfg.features.type,
            "use_cache": cfg.features.use_cache,
            "d_type": cfg.features.d_type
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
