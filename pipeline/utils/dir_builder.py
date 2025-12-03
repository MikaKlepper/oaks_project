# utils/dir_builder.py

from pathlib import Path


def build_feature_dirs(features_root: str, encoder: str, cache_root: str, split: str):
    """
    NEW DESIGN: GLOBAL CACHING ONLY

    RAW SLIDE FEATURES (input):
        <features_root>/Trainings_FM/<ENC>/features
        <features_root>/Validations_FM/<ENC>/features
        <features_root>/Tests_FM/<ENC>/features

    PROCESSED SLIDE FEATURES (cached):
        <cache_root>/<ENC>/<split>/slides/

    PROCESSED ANIMAL FEATURES (cached):
        <cache_root>/<ENC>/<split>/animals/
    """

    encoder = encoder.upper()

    # ---------------------------------------------------------
    # RAW input directory from TG-GATES
    # ---------------------------------------------------------
    raw_slide_dirs = {
        "train": Path(features_root) / "Trainings_FM"   / encoder / "features",
        "val":   Path(features_root) / "Validations_FM" / encoder / "features",
        "test":  Path(features_root) / "Tests_FM"       / encoder / "features",
    }

    # ---------------------------------------------------------
    # GLOBAL FEATURE CACHE
    # ---------------------------------------------------------
    slide_dir  = Path(cache_root) / encoder / split / "slides"
    animal_dir = Path(cache_root) / encoder / split / "animals"

    # Make sure directories exist
    slide_dir.mkdir(parents=True, exist_ok=True)
    animal_dir.mkdir(parents=True, exist_ok=True)

    return {
        "raw_slide_dir": raw_slide_dirs[split],
        "slide_dir": slide_dir,
        "animal_dir": animal_dir,
    }
