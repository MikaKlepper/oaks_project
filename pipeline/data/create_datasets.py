# data/create_datasets.py
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset


class SlideDataset(Dataset):
    def __init__(self, metadata_csv, organ="Liver",
                 split_csv=None, subset_csv=None,
                 features_dir=None, subset_fraction=None, export_dir=None):
        """
        Dataset for animal-level (aggregated) data.
        Uses pre-enriched metadata and filters by split or subset.

        
        Args:
            metadata_csv (str): Path to metadata file (must contain subject_organ_UID, wsi_path, HasHypertrophy, Location, Severity).
            organ (str): Organ filter (e.g., "Liver" or "Kidney").
            split_csv (str, optional): CSV with subject_organ_UID/wsi_path for train/val/test.
            subset_csv (str, optional): CSV with subject_organ_UID/wsi_path for filtering.
            features_dir (str, optional): Directory with precomputed animal-level .pt features.
            subset_fraction (float, optional): Random fraction of data to use.
            export_dir (str, optional): Directory to export WSI paths.
        """
        self.features_dir = Path(features_dir) if features_dir else None
        self.subset_fraction = subset_fraction
        self.export_dir = Path(export_dir) if export_dir else None

        # Load metadata and filter by organ
        meta_df = pd.read_csv(metadata_csv)
        meta_df = meta_df[meta_df["ORGAN"].str.lower() == organ.lower()].copy()

        # Add hypertrophy binary label
        meta_df["HasHypertrophy"] = meta_df["findings"].str.contains("Hypertrophy", na=False).astype(int)

        # Extract location + severity (optional info)
        meta_df[["Location", "Severity"]] = meta_df["findings"].str.extract(
            r"\['Hypertrophy'\s*,\s*'([^']+)'\s*,\s*'([^']+)'"
        )
        meta_df["Location"] = meta_df["Location"].where(meta_df["HasHypertrophy"] == 1, None)
        meta_df["Severity"] = meta_df["Severity"].where(meta_df["HasHypertrophy"] == 1, None)

        # Standardize columns
        meta_df = meta_df.rename(columns={"FILE_LOCATION": "wsi_path"})
        meta_df["subject_organ_UID"] = meta_df["subject_organ_UID"].astype(str)

        # Decide which subset of metadata to keep
        if split_csv:  # train/val/test split
            ids = self._load_ids(split_csv)
        elif subset_csv:  # custom balanced subset
            ids = self._load_ids(subset_csv)
        else:
            ids = set(meta_df["subject_organ_UID"])  # use all animals

        before = len(meta_df)
        df = meta_df[meta_df["subject_organ_UID"].isin(ids)].copy()
        print(f"[INFO] Filtered metadata: {before} → {len(df)}")

        # Apply fractional subset if specified
        if self.subset_fraction is not None:
            before = len(df)
            df = df.sample(frac=self.subset_fraction, random_state=42)
            print(f"[INFO] Fractional subset applied: {before} → {len(df)}")

        # Filter by available animal features
        if self.features_dir:
            available = {p.stem for p in self.features_dir.glob("*.pt")}
            before = len(df)
            df = df[df["subject_organ_UID"].isin(available)]
            print(f"[INFO] Features filtered: {before} → {len(df)}")

        # Optionally export WSI paths
        if self.export_dir:
            self.export_dir.mkdir(parents=True, exist_ok=True)
            base = Path(split_csv).stem if split_csv else Path(subset_csv).stem
            suffix = f"_frac{self.subset_fraction}" if self.subset_fraction else "_full"
            export_path = self.export_dir / f"{base}{suffix}_animal_paths.csv"
            df[["wsi_path"]].to_csv(export_path, index=False)
            print(f"[INFO] Exported WSI paths → {export_path}")

        # Store final data
        self.df = df
        self.animal_ids = df["subject_organ_UID"].tolist()
        self.targets = df["HasHypertrophy"].tolist()
        self.paths = df["wsi_path"].tolist()
        self.locations = df["Location"].tolist()
        self.severities = df["Severity"].tolist()

    # Helper to load IDs from split/subset CSVs
    def _load_ids(self, csv_path):
        """Load animal UIDs from a split or subset CSV."""
        df = pd.read_csv(csv_path)
        if "subject_organ_UID" not in df.columns:
            raise ValueError(f"{csv_path} must contain a 'subject_organ_UID' column for animal-level training")
        return set(df["subject_organ_UID"].astype(str))

    def __len__(self):
        return len(self.animal_ids)

    def __getitem__(self, idx):
        animal_id = self.animal_ids[idx]
        label = torch.tensor(self.targets[idx], dtype=torch.float32)

        if self.features_dir:
            feature_path = self.features_dir / f"{animal_id}.pt"
            feats = torch.load(feature_path)
            # average over all tiles and slides for each animal
            feats = feats.mean(dim=0)
            return feats, label
        else:
            return self.paths[idx], label


# subset_set = SlideDataset(
#     subset_csv="/home/mikaklepper/temporary/repos/slide_2_vec/liver_slide_full_paths_s2v.csv",
#     metadata_csv="/data/pa_cpgarchive/archives/toxicology/open-tg-gates/metadata_SD/open_tggates_master_list.csv",
#     organ="Liver",
#     features_dir=""
#     "/data/temporary/mika/outputs_slide_2_vec/uni_output/uni_output/features"
# )

# print(f"[CHECK] Number of slides in subset_set: {len(subset_set)}")
# print("[CHECK] First 5 slide_ids:", subset_set.slide_ids[:5])
# print("[CHECK] First 5 labels:", subset_set.targets[:5])

# feats, label = subset_set[0]
# print("[CHECK] Features shape:", feats.shape)
# print("[CHECK] Label:", label)


# train_subset = SlideDataset(
#     metadata_csv="/data/pa_cpgarchive/archives/toxicology/open-tg-gates/metadata_SD/open_tggates_master_list.csv",
#     organ="Liver",
#     split_csv="/data/temporary/mika/repos/splitting_data/Splits/train.csv",
#     subset_fraction=0.1,   # <-- 10% of training set
#     export_dir="/data/temporary/mika/repos/splitting_data/Subsets"
# )

# print(f"[CHECK] Number of slides in 10% train subset: {len(train_subset)}")
# print("[CHECK] First 5 slide_ids:", train_subset.slide_ids[:5])
# print("[CHECK] First 5 WSIs:", train_subset.paths[:5])
# print("[CHECK] First 5 labels:", train_subset.targets[:5])

# path, label = train_subset[0]
# print("[CHECK] Example path:", path)
# print("[CHECK] Example label:", label)

