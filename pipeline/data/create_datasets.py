import torch
from torch.utils.data import Dataset
from pathlib import Path


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

        self.embed_dim = int(data["embed_dim"])
        self.dtype = getattr(torch, data.get("d_type", "float32"))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        _id = self.ids[idx]
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
