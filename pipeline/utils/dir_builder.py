# utils/dir_builder.py

from pathlib import Path


def build_feature_dirs(features_root: str, encoder: str, cache_root: str, split: str, aggregation: str):
    
    """
    Builds directories for feature caching:
    - Raw input directory from TG-GATES
    - Global feature cache for slides and animals
    
    Parameters
    ----------
    features_root : str
        Path to the root directory containing all features
    encoder : str
        Encoder name (e.g., "UNI", "H_OPTITMUS_0")
    cache_root : str
        Path to the root directory containing all feature caches
    split : str
        Split name (e.g., "train", "val", "test")
    aggregation : str
        Aggregation method (e.g., "mean", "max", "min")
    
    Returns
    -------
    dict
        Dictionary with the following keys:
            "raw_slide_dir": Path to the raw input directory
            "slide_dir": Path to the global feature cache for slides
            "animal_dir": Path to the global feature cache for animals
    """
    encoder = encoder.upper()
    agg = aggregation.lower()

   # raw feature directories from TG-GATES
    raw_slide_dirs = {
        "train": Path(features_root) / "Trainings_FM"   / encoder / "features",
        "val":   Path(features_root) / "Validations_FM" / encoder / "features",
        "test":  Path(features_root) / "Tests_FM"       / encoder / "features",
    }

    # global feature cache directories
    slide_dir  = Path(cache_root) / agg /  encoder / split / "slides"
    animal_dir = Path(cache_root) / agg /  encoder / split / "animals"

    # make sure directories exist
    slide_dir.mkdir(parents=True, exist_ok=True)
    animal_dir.mkdir(parents=True, exist_ok=True)

    return {
        "raw_slide_dir": raw_slide_dirs[split],
        "slide_dir": slide_dir,
        "animal_dir": animal_dir,
    }
