# utils/feature_cache.py
from pathlib import Path
from data.process_slide_features import process_slide_features
from data.features_per_animal import group_features_by_animal


def ensure_cached_features(prepared):
    """
    Validates slide/animal cached features by matching expected IDs
    with .pt filenames in cache.
    """

    data = prepared["data"]
    df = data["df"]

    slide_dir  = Path(data["slide_dir"])
    animal_dir = Path(data["animal_dir"])
    ftype      = data["features_type"].lower()

    # Convert IDs to strings for consistent filename matching
    expected_slide_ids = sorted(df["slide_id"].astype(str).tolist())
    expected_animal_ids = (
        sorted(df["subject_organ_UID"].astype(str).tolist())
        if "subject_organ_UID" in df.columns else None
    )
    cached_slide_ids = sorted([f.stem for f in slide_dir.glob("*.pt")])
    cached_animal_ids = sorted([f.stem for f in animal_dir.glob("*.pt")])

    if cached_slide_ids == expected_slide_ids:
        print(f"[CACHE] Slide cache OK ({len(cached_slide_ids)}/{len(expected_slide_ids)}) → skip.")
    else:
        print("[CACHE] Slide cache mismatch:")
        missing = set(expected_slide_ids) - set(cached_slide_ids)
        extra   = set(cached_slide_ids) - set(expected_slide_ids)

        if missing:
            print(f"  Missing slide IDs: {sorted(list(missing))[:10]}{' ...' if len(missing)>10 else ''}")
        if extra:
            print(f"  Extra slide files: {sorted(list(extra))[:10]}{' ...' if len(extra)>10 else ''}")

        print("[CACHE] Recomputing slide features...")
        process_slide_features(prepared)

    # If slide mode is done
    if ftype == "slide":
        return

    # Animal mode
    if expected_animal_ids is None:
        raise ValueError("Animal mode requires 'subject_organ_UID' in df.")

    if cached_animal_ids == expected_animal_ids:
        print(f"[CACHE] Animal cache OK ({len(cached_animal_ids)}/{len(expected_animal_ids)}) → skip.")
    else:
        print("[CACHE] Animal cache mismatch:")
        missing = set(expected_animal_ids) - set(cached_animal_ids)
        extra   = set(cached_animal_ids) - set(expected_animal_ids)

        if missing:
            print(f"  Missing animal IDs: {sorted(list(missing))[:10]}{' ...' if len(missing)>10 else ''}")
        if extra:
            print(f"  Extra animal files: {sorted(list(extra))[:10]}{' ...' if len(extra)>10 else ''}")

        print("[CACHE] Aggregating animal features...")
        group_features_by_animal(prepared)
