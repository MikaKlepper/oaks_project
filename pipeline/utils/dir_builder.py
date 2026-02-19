# utils/dir_builder.py

from pathlib import Path


def build_feature_dirs(
    features_root: str,
    encoder: str,
    cache_root: str,
    split: str,
    aggregation: str,
    dataset_key: str,
    dataset_folder: str,
):
    """
    Build dataset-aware raw feature directories and cache directories.

    Assumes the following raw feature layout:

        features_root/
        └── <DATASET_FOLDER>/          # e.g. TG-GATES, UCB
            ├── Trainings_FM/
            ├── Validations_FM/
            └── Tests_FM/

    Cache directories are always dataset-aware and use the dataset key.
    """

    encoder = encoder.upper()
    agg = aggregation.lower()
    dataset_key = dataset_key.lower()

    features_root = Path(features_root)
    dataset_root = features_root / dataset_folder

    # raw feature directories (split-indexed)
    raw_slide_dirs = {
        "train": dataset_root / "Trainings_FM"   / encoder / "features",
        "val":   dataset_root / "Validations_FM" / encoder / "features",
        "test":  dataset_root / "Tests_FM"       / encoder / "features",
    }

    # cache directories
    slide_dir = (
        Path(cache_root)
        / dataset_key
        / agg
        / encoder
        / split
        / "slides"
    )
    animal_dir = (
        Path(cache_root)
        / dataset_key
        / agg
        / encoder
        / split
        / "animals"
    )

    slide_dir.mkdir(parents=True, exist_ok=True)
    animal_dir.mkdir(parents=True, exist_ok=True)

    return {
        "raw_slide_dir": raw_slide_dirs[split],
        "slide_dir": slide_dir,
        "animal_dir": animal_dir,
    }
