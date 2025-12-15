import torch
from torch.utils.data import Dataset
from pathlib import Path


class ToxicologyDataset(Dataset):
    """
    FINAL VERSION — aligned with global slide/animal cache system.

    Expected in prepared["data"]:
        features_type : "slide" or "animal"
        slide_dir     : directory with final slide embeddings (D,)
        animal_dir    : directory with final animal embeddings (D,)
        embed_dim     : expected D
        ids           : sample IDs
        labels        : sample labels
        use_cache     : optional RAM caching
    """

    def __init__(self, prepared):
        data = prepared["data"]

        # -----------------------------------------------
        # Basic metadata
        # -----------------------------------------------
        self.ids    = list(data["ids"])
        self.labels = list(data["labels"])
        self.severity = list(data.get("severity", [None] * len(self.ids)))
        self.location = list(data.get("location", [None] * len(self.ids)))


        if len(self.ids) != len(self.labels):
            raise ValueError("[DATA] IDs and labels mismatch")

        # -----------------------------------------------
        # Feature type: slide | animal
        # -----------------------------------------------
        self.features_type = data.get("features_type", "slide").lower()

        if self.features_type == "slide":
            self.features_dir = Path(data["slide_dir"])
        elif self.features_type == "animal":
            self.features_dir = Path(data["animal_dir"])
        else:
            raise ValueError("[DATA] features_type must be 'slide' or 'animal'")

        if not self.features_dir.exists():
            raise FileNotFoundError(f"[DATA] Features directory missing: {self.features_dir}")

        self.embed_dim = int(data["embed_dim"])
        self.dtype     = getattr(torch, data.get("d_type", "float32"))

        # -----------------------------------------------
        # Optional RAM cache
        # -----------------------------------------------
        self.use_cache = bool(data.get("use_cache", False))
        self.cache = {} if self.use_cache else None

        if self.use_cache:
            print(f"[DATA] Preloading {len(self.ids)} embeddings from {self.features_dir}…")
            self._preload_all()
            print("[DATA] Preloading complete.\n")

    def __len__(self):
        return len(self.ids)

    # ------------------------------------------------------------
    # RAM PRELOAD
    # ------------------------------------------------------------
    def _preload_all(self):
        for _id in self.ids:
            x = self._load_feature_from_disk(_id)
            self.cache[_id] = x

    # ------------------------------------------------------------
    # Disk loader — safe and strict
    # ------------------------------------------------------------
    def _load_feature_from_disk(self, _id):
        f = self.features_dir / f"{_id}.pt"

        if not f.exists():
            raise FileNotFoundError(
                f"[DATA] Missing feature file for ID {_id}: {f}\n"
                "→ Run process_slide_features() or group_features_by_animal() first."
            )

        x = torch.load(f, map_location="cpu").to(self.dtype)

        # Guarantee final shape (D,)
        if x.ndim > 1:
            x = x.flatten()

        if x.numel() != self.embed_dim:
            raise ValueError(
                f"[DATA] Feature {_id} has wrong dimension {x.numel()} "
                f"(expected {self.embed_dim}) in {f}"
            )

        return x

    # ------------------------------------------------------------
    # PyTorch interface
    # ------------------------------------------------------------
    def __getitem__(self, idx):
        _id = self.ids[idx]
        feats = self.cache[_id] if self.use_cache else self._load_feature_from_disk(_id)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feats, label
