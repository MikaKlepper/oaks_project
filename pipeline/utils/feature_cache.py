# utils/feature_cache.py

from pathlib import Path
from data.process_slide_features import process_slide_features
from data.features_per_animal import group_features_by_animal


def ensure_cached_features(prepared):
    """
    Ensures global slide/animal feature cache exists.
    Cache should contain ALL features for the dataset, not only k-shot subset.

    Rules:
      - Only recompute if required IDs are missing or cache is empty.
      - Extra cache files are allowed and expected.
    """

    data = prepared["data"]
    if data.get("feature_backend") == "feature_bank":
        print("[CACHE] Feature-bank backend active -> skipping legacy cache generation.")
        return

    df   = data["df"]

    slide_dir  = Path(data["slide_dir"])
    animal_dir = Path(data["animal_dir"])
    ftype      = data["features_type"].lower()

    #cConvert expected IDs to strings
    expected_slide_ids  = set(df["slide_id"].astype(str).tolist())
    expected_animal_ids = (
        set(df["subject_organ_UID"].astype(str).tolist())
        if "subject_organ_UID" in df.columns else None
    )

    cached_slide_ids  = set(f.stem for f in slide_dir.glob("*.pt"))
    cached_animal_ids = set(f.stem for f in animal_dir.glob("*.pt"))

    # slide features are required for both slide and animal modes, so check them first
    missing_slides = expected_slide_ids - cached_slide_ids

    if missing_slides:
        print(f"[CACHE] Missing {len(missing_slides)} slide features -> Recomputing…")
        process_slide_features(prepared)
    elif len(cached_slide_ids) == 0:
        print("[CACHE] Slide cache empty -> Recomputing all slide features…")
        process_slide_features(prepared)
    else:
        print(f"[CACHE] Slide cache OK ({len(cached_slide_ids)} files) -> SKIP.")

    # STOP HERE IF SLIDE MODE
    if ftype == "slide":
        return

   # animal-level features depend on slide features, so check them after ensuring slide cache is ready
    if expected_animal_ids is None:
        raise ValueError("Animal mode requires 'subject_organ_UID' in df.")

    missing_animals = expected_animal_ids - cached_animal_ids

    if missing_animals:
        print(f"[CACHE] Missing {len(missing_animals)} animal features -> Aggregating…")
        group_features_by_animal(prepared)
    elif len(cached_animal_ids) == 0:
        print("[CACHE] Animal cache empty -> Aggregating all animals…")
        group_features_by_animal(prepared)
    else:
        print(f"[CACHE] Animal cache OK ({len(cached_animal_ids)} files) -> SKIP.")
