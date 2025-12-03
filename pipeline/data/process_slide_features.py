from pathlib import Path
import torch
from tqdm import tqdm


def load_raw_feature(path):
    """
    Loads a raw tile bag or PRISM slide vector.
    Normalizes shape:
        (D,)  → (1, D)
        (N,D) stays (N,D)
    """
    x = torch.load(path, map_location="cpu")
     # ----------------------------
    # FIX FOR RESNET / CNN BACKBONES
    # If feature map is (N, D, 1, 1) -> squeeze to (N, D)
    # ----------------------------
    if x.ndim == 4 and x.shape[2:] == (1, 1):
        x = x.squeeze(-1).squeeze(-1)
        
    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim == 2:
        pass
    else:
        raise ValueError(f"Invalid tensor shape {x.shape} in {path}")

    return x


def process_slide_features(prepared):
    """
    Memory-safe single-process slide preprocessing.
    Suitable for very large (100MB+) tilebag files.
    """

    data = prepared["data"]
    df = data["df"]

    raw_dir = Path(data["raw_slide_dir"])
    out_dir = Path(data["slide_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregate = data.get("aggregate", "mean")
    embed_dim = data["embed_dim"]

    slide_ids = df["slide_id"].astype(str).tolist()

    print(f"[SLIDE] Processing {len(slide_ids)} slides (no multiprocessing)")
    print(f"[SLIDE] Raw dir: {raw_dir}")
    print(f"[SLIDE] Out dir: {out_dir}\n")

    def reduce_fn(x):
        if aggregate == "mean":
            return x.mean(0)
        if aggregate == "max":
            return x.max(0)[0]
        if aggregate == "min":
            return x.min(0)[0]
        raise ValueError(f"Invalid aggregation {aggregate}")

    for sid in tqdm(slide_ids, ncols=120):
        raw_path = raw_dir / f"{sid}.pt"
        out_path = out_dir / f"{sid}.pt"

        if out_path.exists():
            continue
        if not raw_path.exists():
            print(f"[WARN] Missing raw slide: {raw_path}")
            continue

        # --- Load one slide at a time ---
        try:
            x = load_raw_feature(raw_path)
        except Exception as e:
            print(f"[ERR] Failed loading {raw_path}: {e}")
            continue

        reduced = reduce_fn(x)

        if reduced.ndim > 1:
            reduced = reduced.squeeze(0)

        if reduced.numel() != embed_dim:
            print(f"[ERR] Wrong dim for slide {sid}: {reduced.numel()} expected {embed_dim}")
            continue

        torch.save(reduced, out_path)

        # free memory aggressively
        del x
        del reduced
        torch.cuda.empty_cache()  # safe even if on CPU

    print("\n[SLIDE] Slide-level feature creation complete.\n")
