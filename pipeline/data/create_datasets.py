# data/create_datasets.py
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset

class ToxicologyDataset(Dataset):
    def __init__(self, prepared):
        """
        Args:
            prepared (dict):
                {
                    "data": {
                        "df": DataFrame,
                        "ids": list,
                        "labels": list,
                        "slide_dir": Path,
                        "features_dir": Path,
                        "split": str,
                        "aggregate": str,
                        "embed_dim": int,
                        "features_type": str,
                        ... not used
                    },
                    "runtime": {...}   # not used by dataset
                }
        """
        
        data = prepared["data"] 

        self.df = data["df"]
        self.ids = data["ids"]
        self.labels = data["labels"]
        self.features_dir = Path(data["features_dir"])
        self.slide_dir = Path(data["slide_dir"])
        self.aggregate = data["aggregate"]
        self.embed_dim = data["embed_dim"]
        self.features_type = data["features_type"]

        self.use_cache = data["use_cache"] # store featurs in memory if needed
        self.cache = {}  # id -> features tensor

        if self.use_cache:
            print(f"[Dataset] Preloading {len(self.ids)} features into RAM...")
            for id in self.ids:
                fpath = self.features_dir / f"{id}.pt"
                if not fpath.exists():
                    raise FileNotFoundError(f"Missing feature file: {fpath}")
                feats = torch.load(fpath)
                if self.aggregate == "mean":
                    feats = feats.mean(dim=0)
                self.cache[id] = feats
            print(f"[Dataset] Cached {len(self.cache)} tensors in RAM.")


        # quick check
        if len(self.ids) != len(self.labels):
            raise ValueError("Animal IDs and labels must have the same length.")
  
    def __len__(self):
        return len(self.ids)

    def _load_features(self, id):
        """
        Load the features for a given animal ID.

        Parameters
        ----------
        animal_id : str
            The animal ID.

        Returns
        -------
        torch.Tensor
            The loaded features.

        Raises
        ------
        FileNotFoundError
            If the feature file does not exist.
        ValueError
            If the aggregation type is unknown.
        """

        # check cache first
        if self.use_cache and id in self.cache:
            return self.cache[id]
        

        fpath = self.features_dir / f"{id}.pt"
        if not fpath.exists():
            raise FileNotFoundError(f"Missing feature file: {fpath}")

        feats = torch.load(fpath)

        if self.aggregate == "mean":
            return feats.mean(dim=0)
        elif self.aggregate in ["none", None]:
            return feats
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregate}")

    def __getitem__(self, idx):
        """
        Returns the features and label for the given animal index.

        Args:
            idx (int): Index of the animal to retrieve.

        Returns:
            tuple: (feats, label)
                feats (Tensor): The features for the given animal.
                label (Tensor): The label for the given animal.
        """
        id = self.ids[idx]
        feats = self._load_features(id)
        if feats.shape[0] != self.embed_dim:
            raise ValueError(f"Expected {self.embed_dim} features, but got {feats.shape[0]}")
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return feats, label





# class AnimalDataset(Dataset):
#     def __init__(self, cfg):
#         self.cfg = cfg
      
#         # config values
#         self.organ = cfg.data.organ
#         self.metadata_csv = cfg.data.metadata_csv
#         self.aggregate = cfg.aggregation.type

#         # subset controls
#         self.use_subset = cfg.datasets.use_subset
#         self.subset_csv = cfg.datasets.subset_csv
#         self.subset_fraction = cfg.datasets.subset_fraction

#         # split mode ("train", "val", "test", "all")
#         self.split = cfg.datasets.get("split", "train")
#         # choose feature directory depending on split
#         if self.split == "train":
#             self.features_dir = Path(cfg.features.train_animal_dir)
#         elif self.split == "val":
#             self.features_dir = Path(cfg.features.val_animal_dir)
#         elif self.split == "test":
#             self.features_dir = Path(cfg.features.test_animal_dir)
#         # elif self.split == "all":
#         #     self.features_dir = None # handled later
#         else:
#             raise ValueError(f"Unknown split '{self.split}' for feature directory selection.")

        
#         # determine CSV path based on split
#         if self.use_subset:
#                 # Highest priority: subset CSV
#                 if self.subset_csv:
#                     split_csv = self.subset_csv
#                 # Else: subset fraction of split
#                 elif self.subset_fraction is not None:
#                     split_csv = cfg.datasets.get(self.split)
#                 else:
#                     raise ValueError("datasets.use_subset=True but no subset_csv or subset_fraction provided.")
#         else:
#             # no subset
#             split_csv = cfg.datasets.get(self.split)
#         # build final dataset dataframe
#         self.df, self.animal_ids = self._build_dataset(
#             metadata_csv=self.metadata_csv,
#             split_csv=split_csv,
#         )

#         # labels aligned with dataframe rows
#         self.labels = self.df["HasHypertrophy"].tolist()


#     def _build_dataset(self, metadata_csv, split_csv=None):
#         """
#         Build the final dataset dataframe by loading and preprocessing metadata from a CSV file, 
#         collecting data based on the split CSV, applying a fractional subset if specified, 
#         and checking that all required features exist.

#         Args:
#             metadata_csv (str): Path to the metadata CSV file
#             split_csv (str | None): Path to the split CSV file (optional)

#         Returns:
#             df (pd.DataFrame): The final dataset dataframe
#             animal_ids (list): List of animal UIDs corresponding dataframe rows
#         """
#         df = self._load_and_preprocess_metadata(metadata_csv)
#         df= self._collect_data(df, split_csv)
#         if self.use_subset and self.subset_fraction is not None:
#             df = self._apply_fractional_subset(df)
            
#         self._check_features(df)
        
#         animal_ids = df["subject_organ_UID"].astype(str).tolist()
#         return df.reset_index(drop=True), animal_ids

#     def _load_and_preprocess_metadata(self,metadata_csv):
#         """
#         Load metadata from a CSV file and preprocess it by filtering by organ, 
#         extracting binary Hypertrophy labels, and optionally extracting location and severity 
#         information. The dataframe is then renamed to match the expected column names.

#         Args:
#             metadata_csv (str): Path to the metadata CSV file

#         Returns:
#             pd.DataFrame: Preprocessed metadata dataframe
#         """
#         df = pd.read_csv(metadata_csv) # load metadata
#         df = df[df["ORGAN"].str.lower() == self.organ.lower()].copy() # filter by organ
#         df["HasHypertrophy"] = df["findings"].str.contains("Hypertrophy", na=False).astype(int) # make a new hypertrophy column, binary value
#         df[["Location", "Severity"]] = df["findings"].str.extract(r"\['Hypertrophy'\s*,\s*'([^']+)'\s*,\s*'([^']+)'") # extract location and severity if needed
#         df["Location"] = df["Location"].where(df["HasHypertrophy"] == 1, None) 
#         df["Severity"] = df["Severity"].where(df["HasHypertrophy"] == 1, None)
#         df = df.rename(columns={"FILE_LOCATION": "wsi_path"}) # rename and fix base paths
#         df["subject_organ_UID"] = df["subject_organ_UID"].astype(str) # convert subject_organ_UID to string

#         print(f"[INFO] Loaded {len(df)} samples from {metadata_csv} for organ {self.organ}")
#         return df
    
#     def _collect_data(self, df, split_csv=None):
#         """
#         Collect the relevant data from the dataframe based on the split CSV.

#         If a split CSV is provided, the dataframe is filtered to only include
#         the animal-level IDs specified in the CSV. Otherwise, all animal-level
#         IDs from the dataframe are used.

#         Args:
#             df (pd.DataFrame): The dataframe to filter
#             split_csv (str | None): Path to the split CSV file (optional)

#         Returns:
#             pd.DataFrame: The filtered dataframe
#         """
#         if split_csv:  # train/val/test split
#             ids = self._load_ids(split_csv)
#         else:
#             ids = set(df["subject_organ_UID"])  # use all animals

#         df = df[df["subject_organ_UID"].astype(str).isin(ids)]
#         return df.reset_index(drop=True)

#     def _load_ids(self, csv_path):
#         """
#         Load animal-level IDs from a CSV file.

#         Args:
#             csv_path (str): Path to the CSV file containing animal-level IDs

#         Returns:
#             set: Set of animal-level IDs

#         Raises:
#             ValueError: If the CSV file does not contain a 'subject_organ_UID' column
#         """
#         df = pd.read_csv(csv_path)
#         if "subject_organ_UID" not in df.columns:
#             raise ValueError(f"{csv_path} must contain a 'subject_organ_UID' column for animal-level training")
#         return set(df["subject_organ_UID"].astype(str))

#     def _apply_fractional_subset(self, df):
#         """
#         Apply a fractional subset to the dataframe if specified.

#         If `self.subset_fraction` is None, the dataframe is returned unchanged.
#         Otherwise, the dataframe is sampled at the specified fraction, and the number
#         of samples before and after the subset is printed.

#         Args:
#             df (pd.DataFrame): The dataframe to subset

#         Returns:
#             pd.DataFrame: The subset dataframe
#         """
#         if self.subset_fraction is None:
#             return df
#         before = len(df)
#         df = df.sample(frac=self.subset_fraction, random_state=42)
#         after = len(df)
#         print(f"Applied fractional subset of {self.subset_fraction} to {before} samples to {after} samples")
#         return df.reset_index(drop=True)
    
#     def _check_features(self, df):
#         if self.features_dir is None:
#             raise ValueError("Features directory is not provided")
        
#         available = {p.stem for p in self.features_dir.glob("*.pt")}
#         ids = set(df["subject_organ_UID"].astype(str))
#         missing = ids - available

#         if missing:
#             raise ValueError(f"Features for {len(missing)} animals  are not available in {self.features_dir}")
        
#     def _get_features(self, animal_id):
#         feature_path = self.features_dir / f"{animal_id}.pt"
#         if not feature_path.exists():
#             raise FileNotFoundError(f"Missing feature file: {feature_path}")

#         feats = torch.load(feature_path)

#         # Aggregation logic
#         if self.aggregate == "mean":
#             return feats.mean(dim=0)
#         if self.aggregate in ["none", None]:
#             return feats

#         raise ValueError(f"Unknown aggregation method: {self.aggregate}")
    

#     def __len__(self):
#         return len(self.animal_ids)

#     def __getitem__(self, idx):
#         animal_id = self.animal_ids[idx]
#         label = torch.tensor(self.labels[idx], dtype=torch.long)
#         feats = self._get_features(animal_id)

#         return feats, label




# # subset_set = SlideDataset(
# #     subset_csv="/home/mikaklepper/temporary/repos/slide_2_vec/liver_slide_full_paths_s2v.csv",
# #     metadata_csv="/data/pa_cpgarchive/archives/toxicology/open-tg-gates/metadata_SD/open_tggates_master_list.csv",
# #     organ="Liver",
# #     features_dir=""
# #     "/data/temporary/mika/outputs_slide_2_vec/uni_output/uni_output/features"
# # )

# # print(f"[CHECK] Number of slides in subset_set: {len(subset_set)}")
# # print("[CHECK] First 5 slide_ids:", subset_set.slide_ids[:5])
# # print("[CHECK] First 5 labels:", subset_set.targets[:5])

# # feats, label = subset_set[0]
# # print("[CHECK] Features shape:", feats.shape)
# # print("[CHECK] Label:", label)


# # train_subset = SlideDataset(
# #     metadata_csv="/data/pa_cpgarchive/archives/toxicology/open-tg-gates/metadata_SD/open_tggates_master_list.csv",
# #     organ="Liver",
# #     split_csv="/data/temporary/mika/repos/splitting_data/Splits/train.csv",
# #     subset_fraction=0.1,   # <-- 10% of training set
# #     export_dir="/data/temporary/mika/repos/splitting_data/Subsets"
# # )

# # print(f"[CHECK] Number of slides in 10% train subset: {len(train_subset)}")
# # print("[CHECK] First 5 slide_ids:", train_subset.slide_ids[:5])
# # print("[CHECK] First 5 WSIs:", train_subset.paths[:5])
# # print("[CHECK] First 5 labels:", train_subset.targets[:5])

# # path, label = train_subset[0]
# # print("[CHECK] Example path:", path)
# # print("[CHECK] Example label:", label)

