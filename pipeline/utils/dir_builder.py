# utils/dir_builder.py

from pathlib import Path


def build_feature_dirs(features_root: str, encoder: str, exp_root: str, split: str):
    """
    Directory layout:

    RAW SLIDE FEATURES (input):
        /data/.../Trainings_FM/<ENC>/features
        /data/.../Validations_FM/<ENC>/features
        /data/.../Tests_FM/<ENC>/features

    PROCESSED SLIDE FEATURES (output):
        <exp_root>/<split>/slide_features/

    PROCESSED ANIMAL FEATURES (output):
        <exp_root>/<split>/animal_features/
    """

    encoder = encoder.upper()
    root = Path(features_root)
    exp = Path(exp_root)

    # ----------------------------------------
    # RAW slide features (from TG-GATES)
    # ----------------------------------------
    raw_slide_dirs = {
        "train": root / "Trainings_FM"   / encoder / "features",
        "val":   root / "Validations_FM" / encoder / "features",
        "test":  root / "Tests_FM"       / encoder / "features",
    }

    # ----------------------------------------
    # PROCESSED directories inside experiment
    # ----------------------------------------
    slide_dirs = {
        "train": exp / "train" / "slide_features",
        "val":   exp / "eval"  / "slide_features",
        "test":  exp / "test"  / "slide_features",
    }

    animal_dirs = {
        "train": exp / "train" / "animal_features",
        "val":   exp / "eval"  / "animal_features",
        "test":  exp / "test"  / "animal_features",
    }

    # Ensure directories exist
    slide_dirs[split].mkdir(parents=True, exist_ok=True)
    animal_dirs[split].mkdir(parents=True, exist_ok=True)

    return {
        "raw_slide_dir": raw_slide_dirs[split],
        "slide_dir": slide_dirs[split],
        "animal_dir": animal_dirs[split],
    }
