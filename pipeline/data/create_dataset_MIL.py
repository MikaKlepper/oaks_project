import torch
from torch.utils.data import Dataset
import h5py
import numpy as np


class ToxicologyMILDataset(Dataset):
    """
    A PyTorch Dataset for Multiple Instance Learning (MIL) on toxicology data.
    Each sample corresponds to a "bag" of tiles (features) from one or more slides,
    with a single label per bag.
    The dataset supports both slide-level and animal-level features, depending on the configuration.
    Caching Strategy:
      - For small datasets (<=600 samples), bags are cached in memory after first load.
      - For larger datasets, no caching is used to avoid memory issues.
    """

    def __init__(self, prepared):
        data = prepared["data"]
        df = data["df"]

        self.ids = list(data["ids"])
        self.labels = list(data["labels"])

        self.raw_slide_entries = data.get("raw_feature_entries") or {}
        self._h5_files = {}

        self.embed_dim = int(data["embed_dim"])

        self.severity = list(data.get("severity", [None] * len(self.ids)))
        self.location = list(data.get("location", [None] * len(self.ids)))

        ftype = data["features_type"]

        if ftype == "slide":
            self.groups = {sid: [sid] for sid in self.ids}

        elif ftype == "animal":
            self.groups = (
                df.groupby("subject_organ_UID")["slide_id"]
                  .apply(list)
                  .to_dict()
            )
        else:
            raise ValueError("features_type must be 'slide' or 'animal'")

        self.use_bag_cache = len(self.ids) <= 600
        self.bag_cache = {} if self.use_bag_cache else None

        print(
            f"[MIL Dataset] Bag caching "
            f"{'ENABLED' if self.use_bag_cache else 'DISABLED'} "
            f"({len(self.ids)} samples)"
        )

    def __len__(self):
        return len(self.ids)

    def _load_slide(self, slide_id):
        """
        Load one raw slide bag from the feature bank.
        """
        entry = self.raw_slide_entries.get(str(slide_id))
        if entry is None:
            return None

        h5_path = entry["resolved_hdf5_path"]
        h5_file = self._h5_files.get(h5_path)
        if h5_file is None:
            h5_file = h5py.File(h5_path, "r")
            self._h5_files[h5_path] = h5_file

        x = torch.from_numpy(np.asarray(h5_file[entry["hdf5_key"]]))
        if x.ndim == 4 and x.shape[-2:] == (1, 1):
            x = x.squeeze(-1).squeeze(-1)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        elif x.ndim != 2:
            raise ValueError(
                f"[ERROR] Invalid tensor shape {tuple(x.shape)} for slide {slide_id}. "
                "Expected (D,), (N,D), or (N,D,1,1)."
            )
        return x

    def __del__(self):
        for h5_file in self._h5_files.values():
            try:
                h5_file.close()
            except Exception:
                pass

    def __getitem__(self, idx):
        sample_id = self.ids[idx]

        if self.use_bag_cache and sample_id in self.bag_cache:
            bag = self.bag_cache[sample_id]
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return bag, label

        slide_ids = self.groups[sample_id]
        tiles = []

        for sid in slide_ids:
            feats = self._load_slide(sid)
            if feats is not None:
                tiles.append(feats)

        if len(tiles) == 0:
            raise RuntimeError(
                f"[MIL] No tiles found for sample {sample_id} "
                "(missing feature-bank raw slide entries)"
            )

        bag = torch.cat(tiles, dim=0)

        if self.use_bag_cache:
            self.bag_cache[sample_id] = bag

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return bag, label
