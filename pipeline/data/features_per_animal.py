from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import torch
from tqdm import tqdm
import os


def load_pt(path):
    return torch.load(path, map_location="cpu")


def group_features_by_animal(
    prepared,
    max_processes=min(os.cpu_count(), 32),
    min_completion_ratio=0.9,
):
    data = prepared["data"]
    df = data["df"]
    slide_dir = Path(data["slide_dir"])
    out_dir = Path(data["features_dir"])
    embed_dim = data["embed_dim"]
    aggregate = data.get("aggregate", "mean")

    out_dir.mkdir(parents=True, exist_ok=True)

    groups = df.groupby("subject_organ_UID")["slide_id"].apply(list)
    animals = list(groups.index)

    print(f"[FAST] Processes used: {max_processes}")

    # -------------------------------
    # Create pool OUTSIDE try/finally
    # -------------------------------
    pool = ProcessPoolExecutor(max_workers=max_processes)

    try:
        for animal in tqdm(animals, ncols=120):
            outpath = out_dir / f"{animal}.pt"
            if outpath.exists():
                continue

            slide_ids = groups[animal]
            slide_paths = [slide_dir / f"{sid}.pt" for sid in slide_ids]

            futures = {pool.submit(load_pt, p): p for p in slide_paths}

            feats = []
            failed = []

            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    feats.append(fut.result())
                except Exception:
                    failed.append(p)

            total_exp = len(slide_paths)
            total_got = len(feats)
            ratio = total_got / total_exp

            # ----------------------------------------------
            # Require minimum completeness (90% default)
            # ----------------------------------------------
            if ratio < min_completion_ratio:
                print(
                    f"[SKIP] Animal '{animal}': {total_got}/{total_exp} "
                    f"({ratio:.1%}) < required {min_completion_ratio:.0%}. Skipping."
                )
                continue

            big = torch.cat(feats, dim=0)

            if aggregate == "mean":
                reduced = big.mean(0)
            elif aggregate == "max":
                reduced = big.max(0)[0]
            elif aggregate == "min":
                reduced = big.min(0)[0]
            else:
                raise ValueError(f"Unknown aggregation '{aggregate}'")

            if reduced.numel() != embed_dim:
                print(
                    f"[SKIP] Dim mismatch for animal '{animal}' "
                    f"(expected {embed_dim}, got {reduced.numel()})."
                )
                continue

            torch.save(reduced, outpath)

    finally:
        # ----------------------------------------------
        # CRITICAL: Shut down process pools
        # ----------------------------------------------
        print("[CLEANUP] Shutting down process pool...")
        pool.shutdown(wait=True, cancel_futures=True)

    print("[FAST] Animal-level feature creation complete.")
