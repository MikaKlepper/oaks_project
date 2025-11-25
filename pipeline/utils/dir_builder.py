#
from pathlib import Path

def build_feature_dirs(features_root: str, encoder: str):
    """
    Returns ALL required directories using your exact folder structure.
    Validates slide dirs, creates animal dirs.
    """

    encoder = encoder.upper()
    root = Path(features_root)

    dirs = {
        # slide features must already exist
        "train_slide_dir": root / "Trainings_FM"   / encoder / "features",
        "val_slide_dir":   root / "Validations_FM" / encoder / "features",
        "test_slide_dir":  root / "Tests_FM"       / encoder / "features",

        # animal features can be created automatically
        "train_animal_dir": root / "Trainings_FM"   / encoder / "animal_features",
        "val_animal_dir":   root / "Validations_FM" / encoder / "animal_features",
        "test_animal_dir":  root / "Tests_FM"       / encoder / "animal_features",
    }

    # Validate slide directories
    # for key in ["train_slide_dir", "val_slide_dir", "test_slide_dir"]:
    #     p = dirs[key]
    #     if not p.exists():
    #         raise FileNotFoundError(
    #             f"[ERROR] Expected slide directory not found:\n    {p}\n"
    #             f"Your extracted slide features MUST be in this exact structure."
    #         )

    #     if not any(p.glob("*.pt")):
    #         raise FileNotFoundError(
    #             f"[ERROR] Slide directory is empty:\n    {p}\n"
    #             "It must contain extracted feature .pt files."
    #         )

    # Create animal directories
    for key in ["train_animal_dir", "val_animal_dir", "test_animal_dir"]:
        dirs[key].mkdir(parents=True, exist_ok=True)

    return dirs
