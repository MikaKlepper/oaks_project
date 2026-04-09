import torch
from torch.utils.data import Dataset
from pathlib import Path
import h5py
import numpy as np


class ToxicologyDataset(Dataset):
    """
    Pooled dataset.
    One sample = one aggregated feature vector (D,) + one label.

    Works for BOTH slide-level and animal-level pooled features,
    as decided upstream in prepare_dataset_inputs().
    """

    def __init__(self, prepared):
        data = prepared["data"]

        self.ids = list(data["ids"])
        self.labels = list(data["labels"])
        self.severity = list(data.get("severity", [None] * len(self.ids)))
        self.location = list(data.get("location", [None] * len(self.ids)))

        if len(self.ids) != len(self.labels):
            raise ValueError("IDs and labels length mismatch")

        # features_dir should already be set to either slide_dir or animal_dir in prepare_dataset_inputs()
        self.features_dir = Path(data["features_dir"])
        self.feature_backend = data.get("feature_backend", "legacy")
        self.feature_artifacts = data.get("feature_artifacts") or {}
        self._h5_files = {}

        self.embed_dim = int(data["embed_dim"])
        self.dtype = getattr(torch, data.get("d_type", "float32"))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        _id = self.ids[idx]
        if self.feature_backend == "feature_bank":
            artifact = self.feature_artifacts.get(str(_id))
            if artifact is None:
                raise FileNotFoundError(f"Missing feature artifact for {_id}")
            h5_path = artifact["resolved_hdf5_path"]
            h5_file = self._h5_files.get(h5_path)
            if h5_file is None:
                h5_file = h5py.File(h5_path, "r")
                self._h5_files[h5_path] = h5_file
            x = torch.from_numpy(
                np.asarray(h5_file[artifact["hdf5_key"]])
            ).to(self.dtype)
        else:
            path = self.features_dir / f"{_id}.pt"

            if not path.exists():
                raise FileNotFoundError(f"Missing feature file: {path}")

            x = torch.load(path, map_location="cpu").to(self.dtype)

        if x.ndim > 1:
            x = x.flatten()

        if x.numel() != self.embed_dim:
            raise ValueError(
                f"Feature {_id} has shape {tuple(x.shape)}, expected ({self.embed_dim},)"
            )

        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

    def __del__(self):
        for h5_file in self._h5_files.values():
            try:
                h5_file.close()
            except Exception:
                pass
