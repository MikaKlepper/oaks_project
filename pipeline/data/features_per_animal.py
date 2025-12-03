from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import torch
from tqdm import tqdm
import os


# -----------------------------
# Safe loader for (D,) slide embeddings
# -----------------------------
def load_pt(path: Path):
    """
    Loads a processed slide embedding.
    Should always be (D,) since slide stage already aggregated.
    """
    x = torch.load(path, map_location="cpu")

    if x.ndim != 1:
        raise ValueError(f"[ERROR] Expected (D,) slide vector, got {tuple(x.shape)} in {path}")

    return x



# -----------------------------
# MAIN: Aggregate processed slides → animal embeddings
# -----------------------------
def group_features_by_animal(
    prepared,
    max_processes=min(os.cpu_count(), 16),
    min_completion_ratio=0.9,
):
    data = prepared["data"]
    df = data["df"]

    # Read processed slide features
    slide_dir = Path(data["slide_dir"])
    animal_dir = Path(data["animal_dir"])
    animal_dir.mkdir(parents=True, exist_ok=True)

    embed_dim = int(data["embed_dim"])
    aggregate = data.get("aggregate", "mean")

    # Group all slide IDs by animal_id
    groups = df.groupby("subject_organ_UID")["slide_id"].apply(list)
    animals = list(groups.index)

    print(f"[ANIMAL] Starting animal-level aggregation")
    print(f"[ANIMAL] Input slides : {slide_dir}")
    print(f"[ANIMAL] Output animals: {animal_dir}")
    print(f"[ANIMAL] Workers: {max_processes}, min completion: {min_completion_ratio}\n")

    # -----------------------------
    # Create worker pool
    # -----------------------------
    pool = ProcessPoolExecutor(max_workers=max_processes)

    try:
        for animal in tqdm(animals, ncols=120):
            out_path = animal_dir / f"{animal}.pt"

            # Skip if already cached
            if out_path.exists():
                continue

            slide_ids = groups[animal]
            slide_paths = [slide_dir / f"{sid}.pt" for sid in slide_ids]

            # Filter missing slides
            existing = [p for p in slide_paths if p.exists()]
            missing_count = len(slide_paths) - len(existing)

            if missing_count > 0:
                print(f"[WARN] Animal {animal}: {missing_count} missing slide files")

            if len(existing) == 0:
                print(f"[SKIP] Animal {animal} has no valid slide feature files")
                continue

            # -----------------------------
            # Submit tasks
            # -----------------------------
            futures = {pool.submit(load_pt, p): p for p in existing}

            feats = []
            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    feats.append(fut.result())
                except Exception as e:
                    print(f"[ERR] Failed loading slide {p}: {e}")

            total_expected = len(slide_paths)
            total_loaded = len(feats)
            ratio = total_loaded / total_expected

            if ratio < min_completion_ratio:
                print(f"[SKIP] {animal}: {total_loaded}/{total_expected} loaded ({ratio:.1%}) < threshold")
                continue

            # -----------------------------
            # Stack (N, D) into final animal vector
            # -----------------------------
            big = torch.stack(feats, dim=0)  # (Nslides, D)

            if aggregate == "mean":
                animal_vec = big.mean(0)
            elif aggregate == "max":
                animal_vec = big.max(0).values
            elif aggregate == "min":
                animal_vec = big.min(0).values
            else:
                raise ValueError(f"[ERROR] Unknown aggregation '{aggregate}'")

            # Validate dimension
            if animal_vec.numel() != embed_dim:
                print(
                    f"[SKIP] Wrong dimension for {animal}: {animal_vec.numel()} "
                    f"(expected {embed_dim})"
                )
                continue

            torch.save(animal_vec, out_path)

    finally:
        print("[CLEANUP] Shutting down animal aggregation workers…")
        pool.shutdown(wait=True, cancel_futures=True)

    print("\n[ANIMAL] Animal-level feature creation complete.\n")
