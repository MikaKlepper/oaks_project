# utils/feature_cache.py
from pathlib import Path
from data.process_slide_features import process_slide_features
from data.features_per_animal import group_features_by_animal


def ensure_cached_features(prepared):
    """
    Ensure features exist in the GLOBAL cache and compute them only if needed.

    Rules:
      - Always check slide cache first
      - If slide cache is empty → compute slides
      - For animal mode: same for animal cache
      - If features already exist → skip expensive processing
    """

    data = prepared["data"]

    slide_dir  = Path(data["slide_dir"])
    animal_dir = Path(data["animal_dir"])
    ftype      = data["features_type"].lower()

    # -----------------------------
    # ALWAYS CHECK SLIDE CACHE FIRST
    # -----------------------------
    slide_exists = any(slide_dir.glob("*.pt"))
    if slide_exists:
        print(f"[CACHE] Slide features already exist → SKIP slide processing.")
    else:
        print(f"[CACHE] Slide cache empty → RUN slide processing.")
        process_slide_features(prepared)

    # -----------------------------
    # IF SLIDE MODE → DONE
    # -----------------------------
    if ftype == "slide":
        return

    # -----------------------------
    # ANIMAL MODE → CHECK ANIMAL CACHE
    # -----------------------------
    animal_exists = any(animal_dir.glob("*.pt"))
    if animal_exists:
        print(f"[CACHE] Animal features already exist → SKIP animal aggregation.")
    else:
        print(f"[CACHE] Animal cache empty → RUN animal aggregation.")
        group_features_by_animal(prepared)
