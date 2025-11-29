# utils/dir_builder.py

from pathlib import Path

def build_feature_dirs(features_root: str, encoder: str, exp_root: str, split: str):
    """
    Build slide + animal dirs depending on SPLIT.

    slide_dirs => ALWAYS from the original TG-GATES structure
    animal_dirs => ALWAYS inside experiment_root/{train/val/}/animal_features
    """

    encoder = encoder.upper()

    root = Path(features_root)

    # slide features are fixed
    slide_dirs = {
        "train": root / "Trainings_FM"   / encoder / "features",
        "val":   root / "Validations_FM" / encoder / "features",
        "test":  root / "Tests_FM"       / encoder / "features",
    }

    # animal features are inside experiment root
    exp = Path(exp_root)
    animal_dirs = {
        "train": exp / "train" / "animal_features",
        "val":   exp / "eval"  / "animal_features",
        "test":  exp / "test"  / "animal_features",
    }

    animal_dirs[split].mkdir(parents=True, exist_ok=True)

    return {
        "slide_dir":  slide_dirs[split],
        "animal_dir": animal_dirs[split],
    }
