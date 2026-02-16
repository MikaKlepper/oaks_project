import torch
from torch.utils.data import Dataset
from pathlib import Path

from data.process_slide_features import load_raw_features


class ToxicologyMILDataset(Dataset):
    """
    Multiple Instance Learning (MIL) dataset.

    One sample = one bag of tile embeddings (N, D) + one label.

    - Slide-level MIL:
        one slide → one bag of tiles
    - Animal-level MIL:
        one animal → multiple slides → concatenated tile bags

    Notes
    -----
    - Uses raw (tile-level) slide features
    - Padding & masks handled by collate_mil
    - Bag-level caching is enabled automatically for small datasets (e.g. k ≤ 100)
    """

    def __init__(self, prepared):
        data = prepared["data"]
        df = data["df"]

        self.ids = list(data["ids"])
        self.labels = list(data["labels"])
        self.raw_slide_dir = Path(data["raw_slide_dir"])
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

        # Enable bag caching for small datasets to speed up training (at the cost of memory)
        self.use_bag_cache = len(self.ids) <= 600
        self.bag_cache = {} if self.use_bag_cache else None

        if self.use_bag_cache:
            print(f"[MIL Dataset] Bag caching ENABLED ({len(self.ids)} samples)")
        else:
            print(f"[MIL Dataset] Bag caching DISABLED ({len(self.ids)} samples)")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]

        # Use cached bag if it exists and caching is enabled
        if self.use_bag_cache and sample_id in self.bag_cache:
            bag = self.bag_cache[sample_id]
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return bag, label

        slide_ids = self.groups[sample_id]
        tiles = []

        for sid in slide_ids:
            path = self.raw_slide_dir / f"{sid}.pt"
            if path.exists():
                tiles.append(load_raw_features(path))

        if len(tiles) == 0:
            raise RuntimeError(f"No tiles found for sample {sample_id}")

        bag = torch.cat(tiles, dim=0)

        # Cache the bag for future use if caching is enabled
        if self.use_bag_cache:
            self.bag_cache[sample_id] = bag

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return bag, label
