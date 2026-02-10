import torch
from torch.utils.data import Dataset
from pathlib import Path


class ToxicologyDataset(Dataset):
    def __init__(self, prepared):
        """
        Initialize ToxicologyDataset.

        Parameters:
        prepared (dict): Prepared data containing the following:
            - data (dict): Data containing the following:
                - ids (list): List of IDs for the dataset
                - labels (list): List of labels for the dataset
                - features_type (str): Feature type ('slide' or 'animal')
                - slide_dir (str): Directory containing slide features (if features_type='slide')
                - animal_dir (str): Directory containing animal features (if features_type='animal')
                - embed_dim (int): Embedding dimension of the features
                - d_type (str): Data type of the features (default: 'float32')
                - use_cache (bool): Whether to use RAM cache for the features (default: False)

        Raises:
        ValueError: If IDs and labels length mismatch
        ValueError: If features_type is not 'slide' or 'animal'
        """
        data = prepared["data"]

        self.ids = list(data["ids"])
        self.labels = list(data["labels"])

        if len(self.ids) != len(self.labels):
            raise ValueError("IDs and labels length mismatch")

        # Feature type
        ftype = data.get("features_type", "slide")
        if ftype == "slide":
            self.features_dir = Path(data["slide_dir"])
        elif ftype == "animal":
            self.features_dir = Path(data["animal_dir"])
        else:
            raise ValueError("features_type must be 'slide' or 'animal'")

        self.embed_dim = int(data["embed_dim"])
        self.dtype = getattr(torch, data.get("d_type", "float32"))

        # Optional RAM cache
        self.use_cache = bool(data.get("use_cache", False))
        self.cache = {} if self.use_cache else None

        if self.use_cache:
            for _id in self.ids:
                self.cache[_id] = self._load(_id)

    def __len__(self):
        return len(self.ids)

    def _load(self, _id):
        path = self.features_dir / f"{_id}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing feature file: {path}")

        x = torch.load(path, map_location="cpu").to(self.dtype)

        if x.ndim != 1 or x.numel() != self.embed_dim:
            raise ValueError(
                f"Feature {_id} has shape {tuple(x.shape)}, expected ({self.embed_dim},)"
            )

        return x

    def __getitem__(self, idx):
        _id = self.ids[idx]
        x = self.cache[_id] if self.use_cache else self._load(_id)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
