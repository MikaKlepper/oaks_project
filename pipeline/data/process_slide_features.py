from pathlib import Path
import torch
from tqdm import tqdm

# load slide features for each encoder type, and normalize shapes to (N, D) or (1, D)
def load_raw_features(path):
    """
    Loads slide features from any encoder.
    Normalizes shapes:

        (D,)         -> (1, D)  # single vector (e.g. PRISM (Slide-level embeddings))
        (N, D)       -> (N, D)  # standard tile-level FMs
        (N, D, 1, 1) -> (N, D)  # CNN backbones (ResNet50, Hibou, etc.)
    """
    x = torch.load(path, map_location="cpu", mmap=True)

    # CNN outputs (N, D, 1, 1)
    if x.ndim == 4 and x.shape[-2:] == (1, 1):
        x = x.squeeze(-1).squeeze(-1)

    # Single vector (D,) -> (1, D)
    if x.ndim == 1:
        x = x.unsqueeze(0)

    # If already (N, D), do nothing (tile-level FMs)
    elif x.ndim == 2:
        pass

    else:
        raise ValueError(
            f"[ERROR] Invalid tensor shape {tuple(x.shape)} in {path}. "
            "Expected (D,), (N,D), or (N,D,1,1)."
        )

    return x

def process_slide_features(prepared):
    """
    Process slide features from raw tile-level embeddings.

    This function takes prepared data which contains the following:
    - raw_slide_dir: directory containing raw tile-level embeddings for each slide
    - slide_dir: directory to store aggregated slide-level embeddings
    - aggregate: aggregation method to apply to tile-level embeddings in each slide
    - embed_dim: expected dimension of the aggregated slide-level embeddings

    The function will load each slide's tile-level embeddings, aggregate them using the specified method,
    and save the resulting slide-level embeddings to the specified slide_dir.

    If a slide's tile-level embeddings are missing or if the aggregated vector does not match the expected embed_dim,
    the function will print a warning and skip that slide.

    :param prepared: prepared data containing the necessary information
    :return: None
    """
    data = prepared["data"]
    df = data["df"]

    raw_dir = Path(data["raw_slide_dir"])
    out_dir = Path(data["slide_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregate = data.get("aggregate", "mean")
    embed_dim = int(data["embed_dim"])

    slide_ids = df["slide_id"].astype(str)

    print(f"[SLIDE] Processing {len(slide_ids)} slides")

    # aggregation
    if aggregate == "mean":
        reduce_fn = lambda x: x.mean(0)
    elif aggregate == "max":
        reduce_fn = lambda x: x.max(0).values
    elif aggregate == "min":
        reduce_fn = lambda x: x.min(0).values
    else:
        raise ValueError(f"Unknown aggregation mode '{aggregate}'")

    for sid in tqdm(slide_ids, ncols=120):
        raw_path = raw_dir / f"{sid}.pt"
        out_path = out_dir / f"{sid}.pt"

        if out_path.exists():
            continue
        if not raw_path.exists():
            print(f"[WARN] Missing {raw_path}")
            continue

        try:
            x = load_raw_features(raw_path)
            vec = reduce_fn(x)
        except Exception as e:
            print(f"[ERR] {sid}: {e}")
            continue

        if vec.numel() != embed_dim:
            print(f"[ERR] {sid}: dim {vec.numel()} ≠ {embed_dim}")
            continue

        torch.save(vec, out_path)

    print("\n[SLIDE] Slide-level feature creation complete.\n")
