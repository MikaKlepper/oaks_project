import torch
from torch.utils.data import Dataset
from pathlib import Path

from data.process_slide_features import load_raw_features


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

        self.raw_slide_dir = [data["raw_slide_dir"]]

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
        Try all raw feature directories for a slide.
        """
        for d in self.raw_slide_dir:
            path = d / f"{slide_id}.pt"
            if path.exists():
                return load_raw_features(path)
        return None

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
                f"(searched {len(self.raw_slide_dir)} dirs)"
            )

        bag = torch.cat(tiles, dim=0)

        if self.use_bag_cache:
            self.bag_cache[sample_id] = bag

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return bag, label
