import torch
from torch.utils.data import Dataset
from pathlib import Path


class ToxicologyDataset(Dataset):
    """
    FINAL LEAN VERSION

    EXPECTED INPUTS (from prepare_dataset_inputs):

        features_type='slide'  → features_dir contains FINAL processed slide embeddings (D,)
        features_type='animal' → features_dir contains FINAL animal embeddings (D,)

    This Dataset:
        - NEVER loads raw tile bags
        - NEVER performs aggregation
        - ONLY loads final (D,) tensors

    Output per sample:
        feats: (D,)
        label: long
    """

    def __init__(self, prepared):
        data = prepared["data"]

        # Basic identifiers
        self.ids = list(data["ids"])
        self.labels = list(data["labels"])
        if len(self.ids) != len(self.labels):
            raise ValueError("IDs and labels mismatch")

        # IMPORTANT:
        # Directory now ALWAYS contains final (D,) embeddings:
        #   slide → created by process_slide_features()
        #   animal → created by group_features_by_animal()
        self.features_dir = Path(data["features_dir"])
        self.features_type = data.get("features_type", "slide").lower()
        self.dtype = getattr(torch, data.get("d_type", "float32"))

        if self.features_type not in ("slide", "animal"):
            raise ValueError("features_type must be 'slide' or 'animal'")

        # Optional in-RAM caching of (D,) vectors
        self.use_cache = bool(data.get("use_cache", False))
        self.cache = {} if self.use_cache else None

        if self.use_cache:
            print(f"[DATA] Preloading {len(self.ids)} precomputed embeddings…")
            self._preload_all()
            print("[DATA] Preloading complete.\n")

    def __len__(self):
        return len(self.ids)

    # ------------------------------------------------------------------
    # PRELOAD small (D,) vectors into RAM
    # ------------------------------------------------------------------
    def _preload_all(self):
        for _id in self.ids:
            f = self.features_dir / f"{_id}.pt"
            if not f.exists():
                raise FileNotFoundError(f"Missing precomputed feature: {f}")

            x = torch.load(f, map_location="cpu").to(self.dtype)

            # Guarantee 1D (D,)
            if x.ndim > 1:
                x = x.squeeze(0)

            self.cache[_id] = x

    # ------------------------------------------------------------------
    # LOAD a single precomputed embedding from disk
    # ------------------------------------------------------------------
    def _load_single(self, _id):
        f = self.features_dir / f"{_id}.pt"
        if not f.exists():
            raise FileNotFoundError(f"Missing precomputed feature: {f}")

        x = torch.load(f, map_location="cpu").to(self.dtype)

        # Guarantee 1D (D,)
        if x.ndim > 1:
            x = x.squeeze(0)

        return x

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        _id = self.ids[idx]

        # Use RAM cache or disk load
        feats = self.cache[_id] if self.use_cache else self._load_single(_id)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return feats, label
