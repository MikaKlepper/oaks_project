from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import torch
from tqdm import tqdm
import os


def load_pt(path):
    """Load a processed slide embedding (D,) or raw slide FM (1,D)."""
    x = torch.load(path, map_location="cpu")

    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim == 2:
        pass
    else:
        raise ValueError(f"Invalid feature shape {x.shape} in {path}")

    return x


def group_features_by_animal(
    prepared,
    max_processes=min(os.cpu_count(), 32),
    min_completion_ratio=0.9,
):
    """
    Correct design:

        INPUT  = prepared["data"]["slide_dir"]      <-- processed slide embeddings
        OUTPUT = prepared["data"]["animal_dir"]     <-- animal-level embeddings
    """

    data = prepared["data"]
    df = data["df"]

    # *** CORRECTED DIRECTORY ***
    slide_feat_dir = Path(data["slide_dir"])   # always read slides from here
    out_dir        = Path(data["animal_dir"])  # always save animals here
    out_dir.mkdir(parents=True, exist_ok=True)

    embed_dim = data["embed_dim"]
    aggregate = data.get("aggregate", "mean")

    groups = df.groupby("subject_organ_UID")["slide_id"].apply(list)
    animals = list(groups.index)

    print(f"[ANIMAL] Using {max_processes} worker processes…")
    print(f"[ANIMAL] Input slide features: {slide_feat_dir}")
    print(f"[ANIMAL] Output animal features: {out_dir}\n")

    pool = ProcessPoolExecutor(max_workers=max_processes)

    try:
        for animal in tqdm(animals, ncols=120):
            outpath = out_dir / f"{animal}.pt"

            # Skip if already computed
            if outpath.exists():
                continue

            slide_ids = groups[animal]
            slide_paths = [slide_feat_dir / f"{sid}.pt" for sid in slide_ids]

            # Load all slide embeddings for this animal
            futures = {pool.submit(load_pt, p): p for p in slide_paths if p.exists()}

            feats = []
            for fut in as_completed(futures):
                try:
                    feats.append(fut.result())
                except Exception:
                    pass

            total_exp = len(slide_paths)
            total_got = len(feats)

            if total_exp == 0:
                print(f"[ERR] Animal {animal} has zero slide IDs in split.")
                continue

            ratio = total_got / total_exp

            if ratio < min_completion_ratio:
                print(f"[SKIP] {animal}: only {total_got}/{total_exp} slides available")
                continue

            # Combine slide embeddings
            big = torch.cat(feats, dim=0)

            # Final pooling
            if aggregate == "mean":
                reduced = big.mean(0)
            elif aggregate == "max":
                reduced = big.max(0)[0]
            elif aggregate == "min":
                reduced = big.min(0)[0]
            else:
                raise ValueError(f"Unknown aggregation: {aggregate}")

            # Verify shape
            if reduced.numel() != embed_dim:
                print(f"[SKIP] Wrong dim for {animal}: {reduced.numel()} vs expected {embed_dim}")
                continue

            torch.save(reduced, outpath)

    finally:
        print("[CLEANUP] Shutting down workers…")
        pool.shutdown(wait=True, cancel_futures=True)

    print("[ANIMAL] Animal feature creation complete.\n")
