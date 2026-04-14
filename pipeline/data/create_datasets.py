import torch
from torch.utils.data import Dataset
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

        self.feature_entries = data.get("feature_entries") or {}
        self._h5_files = {}

        self.embed_dim = int(data["embed_dim"])
        self.dtype = getattr(torch, data.get("d_type", "float32"))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        entry = self.feature_entries.get(str(sample_id))
        if entry is None:
            raise FileNotFoundError(f"Missing feature registry entry for {sample_id}")

        h5_path = entry["resolved_hdf5_path"]
        h5_file = self._h5_files.get(h5_path)
        if h5_file is None:
            h5_file = h5py.File(h5_path, "r")
            self._h5_files[h5_path] = h5_file

        x = torch.from_numpy(np.asarray(h5_file[entry["hdf5_key"]])).to(self.dtype)

        if x.ndim > 1:
            x = x.flatten()

        if x.numel() != self.embed_dim:
            raise ValueError(
                f"Feature {sample_id} has shape {tuple(x.shape)}, expected ({self.embed_dim},)"
            )

        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

    def __del__(self):
        for h5_file in self._h5_files.values():
            try:
                h5_file.close()
            except Exception:
                pass
