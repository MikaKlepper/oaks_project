# data/create_datasets.py
import torch
from pathlib import Path
from torch.utils.data import Dataset
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


def _load_pt_file(args):
    """Worker function for multiprocessing preload."""
    fpath, dtype, features_type, embed_dim = args

    feats = torch.load(fpath).to(dtype)

    # Validation happens inside worker to fail early
    if features_type == "animal":
        if feats.ndim != 1 or feats.shape[0] != embed_dim:
            raise ValueError(
                f"Animal feature wrong shape {tuple(feats.shape)} for file {fpath}"
            )

    elif features_type == "slide":
        if feats.ndim != 2 or feats.shape[1] != embed_dim:
            raise ValueError(
                f"Slide feature wrong shape {tuple(feats.shape)} for file {fpath}"
            )

    return fpath.stem, feats
    


class ToxicologyDataset(Dataset):
    """
    Dataset with:
    - Disk loading OR
    - Full multiprocessing preload when use_cache=True
    """

    def __init__(self, prepared):
        data = prepared["data"]

        # -------------------------------------------------------------
        # IDs and labels
        # -------------------------------------------------------------
        self.ids = list(data["ids"])
        self.labels = list(data["labels"])

        if len(self.ids) != len(self.labels):
            raise ValueError("ids and labels must have the same length.")

        # -------------------------------------------------------------
        # Configuration
        # -------------------------------------------------------------
        self.features_dir = Path(data["features_dir"])
        self.features_type = data.get("features_type", "animal").lower()
        self.embed_dim = int(data["embed_dim"])

        dtype_str = data.get("dtype", "float32")
        self.dtype = getattr(torch, dtype_str)

        # The only flag that activates preload
        self.use_cache = bool(data.get("use_cache", False))

        if self.features_type not in ("animal", "slide"):
            raise ValueError("features_type must be 'animal' or 'slide'.")

        # Storage for preloaded features
        self.features = None

        # -------------------------------------------------------------
        # FAST MULTIPROCESSING PRELOAD
        # -------------------------------------------------------------
        if self.use_cache:
            print(f"[DATA] Multiprocessing preload of {len(self.ids)} items...")
            self.features = self._preload_all_parallel()
            print("[DATA] Preloading complete.\n")

    def __len__(self):
        return len(self.ids)

    # -------------------------------------------------------------
    # Preload everything using multiprocessing
    # -------------------------------------------------------------
    def _preload_all_parallel(self):
        tasks = []
        for _id in self.ids:
            fpath = self.features_dir / f"{_id}.pt"
            if not fpath.exists():
                raise FileNotFoundError(f"Missing feature file: {fpath}")

            tasks.append((fpath, self.dtype, self.features_type, self.embed_dim))

        results = {}
        max_workers = min(os.cpu_count(), 32)  # limit to avoid oversaturation

        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_load_pt_file, t): t[0] for t in tasks}

            for fut in as_completed(futures):
                fpath = futures[fut]
                try:
                    name, feats = fut.result()
                    results[name] = feats
                except Exception as e:
                    raise RuntimeError(f"Failed loading {fpath}: {e}")

        # Maintain original ID order, but results accessed by ID
        return results

    # -------------------------------------------------------------
    # Disk loading fallback (use_cache=False)
    # -------------------------------------------------------------
    def _load_from_disk(self, _id: str):
        fpath = self.features_dir / f"{_id}.pt"

        if not fpath.exists():
            raise FileNotFoundError(f"Feature file not found: {fpath}")

        feats = torch.load(fpath).to(self.dtype)

        if self.features_type == "animal":
            if feats.ndim != 1 or feats.shape[0] != self.embed_dim:
                raise ValueError(
                    f"Animal-level features for {_id} wrong shape {tuple(feats.shape)}"
                )

        elif self.features_type == "slide":
            if feats.ndim != 2 or feats.shape[1] != self.embed_dim:
                raise ValueError(
                    f"Slide-level features for {_id} wrong shape {tuple(feats.shape)}"
                )

        return feats

    # -------------------------------------------------------------
    # Final item
    # -------------------------------------------------------------
    def __getitem__(self, idx):
        _id = self.ids[idx]

        if self.use_cache:
            feats = self.features[_id]  # already in RAM
        else:
            feats = self._load_from_disk(_id)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feats, label
