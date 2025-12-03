from pathlib import Path
import torch
from tqdm import tqdm


# --------------------------------------------------------
# Load raw slide feature file and normalize shapes
# --------------------------------------------------------
def load_raw_feature(path):
    """
    Loads slide features from any encoder.
    Normalizes shapes:

        (D,)         → (1, D)
        (N, D)       → (N, D)
        (N, D, 1, 1) → (N, D)  # CNN backbones (ResNet50, Hibou, etc.)
    """
    x = torch.load(path, map_location="cpu")

    # CNN outputs (N, D, 1, 1)
    if x.ndim == 4 and x.shape[-2:] == (1, 1):
        x = x.squeeze(-1).squeeze(-1)

    # Single vector
    if x.ndim == 1:
        x = x.unsqueeze(0)

    # Valid tile-bag
    elif x.ndim == 2:
        pass

    else:
        raise ValueError(
            f"[ERROR] Invalid tensor shape {tuple(x.shape)} in {path}. "
            "Expected (D,), (N,D), or (N,D,1,1)."
        )

    return x


# --------------------------------------------------------
# Slide feature processing (raw → final (D,))
# --------------------------------------------------------
def process_slide_features(prepared):
    """
    Safe, non-parallel slide preprocessing:
    - Handles huge (>100MB) tilebags
    - Works with all encoders including CNN feature maps
    - Creates exactly one (D,) vector per slide
    """
    data = prepared["data"]
    df = data["df"]

    raw_dir = Path(data["raw_slide_dir"])
    out_dir = Path(data["slide_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregate = data.get("aggregate", "mean")
    embed_dim = int(data["embed_dim"])

    slide_ids = df["slide_id"].astype(str).tolist()

    print(f"[SLIDE] Processing {len(slide_ids)} slides (no multiprocessing)")
    print(f"[SLIDE] Raw dir:  {raw_dir}")
    print(f"[SLIDE] Out dir:  {out_dir}")
    print(f"[SLIDE] Aggregate mode: {aggregate}")
    print(f"[SLIDE] Expected embedding dim: {embed_dim}\n")

    # ----------------------------------------------------
    # Aggregation method
    # ----------------------------------------------------
    def reduce_fn(x):
        if aggregate == "mean":
            return x.mean(0)
        if aggregate == "max":
            return x.max(0)[0]
        if aggregate == "min":
            return x.min(0)[0]
        raise ValueError(f"[ERROR] Unknown aggregation mode '{aggregate}'")

    # ----------------------------------------------------
    # Process each slide (cached)
    # ----------------------------------------------------
    for sid in tqdm(slide_ids, ncols=120):
        raw_path = raw_dir / f"{sid}.pt"
        out_path = out_dir / f"{sid}.pt"

        # Skip if cached
        if out_path.exists():
            continue

        if not raw_path.exists():
            print(f"[WARN] Missing raw slide file: {raw_path}")
            continue

        # Load slide
        try:
            x = load_raw_feature(raw_path)
        except Exception as e:
            print(f"[ERR] Could not load {raw_path}: {e}")
            continue

        # Aggregate tiles → one vector
        reduced = reduce_fn(x)

        # Remove unnecessary dims
        if reduced.ndim > 1:
            reduced = reduced.reshape(-1)  # flatten safely

        if reduced.numel() != embed_dim:
            print(
                f"[ERR] Slide {sid}: got dim {reduced.numel()}, expected {embed_dim}. Skipping."
            )
            del x
            continue

        # Save
        try:
            torch.save(reduced, out_path)
        except Exception as e:
            print(f"[ERR] Failed to save {out_path}: {e}")

        # Cleanup
        del x
        del reduced
        torch.cuda.empty_cache()

    print("\n[SLIDE] Slide-level feature creation complete.\n")
