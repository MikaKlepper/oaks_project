import pandas as pd
from pathlib import Path
import torch

def group_features_by_animal(prepared):
    """
    Aggregate slide-level features into animal-level features.

    Parameters
    ----------
    prepared : dict
        A dictionary containing the following keys:
            - "df": A pandas DataFrame containing the slide metadata.
            - "animal_dir": A Path object pointing to the directory where the animal-level features will be saved.
            - "slide_dir": A Path object pointing to the directory where the slide-level features are stored.
            - "split": A string indicating the split (e.g. "train", "val", "test").

    Returns
    -------
    dict
        A dictionary containing summary statistics about the aggregation process.
    """

    df = prepared["df"]
    animal_dir = prepared["features_dir"]
    slide_dir = prepared["slide_dir"]
    split = prepared["split"]

    print(f"[INFO] Creating animal features for split: {split}")
    print(f"[INFO] Slide feature directory:  {slide_dir}")
    print(f"[INFO] Animal feature directory: {animal_dir}")
    print(f"[INFO] Total number of rows in DF: {len(df)}")

    # create output directory if not exists
    animal_dir.mkdir(parents=True, exist_ok=True)

    # column check
  
    required_cols = {"slide_id", "subject_organ_UID"}

    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # group slide IDs by animal
    grouped = df.groupby("subject_organ_UID")["slide_id"].apply(list)


    summary = {
        "split": split,
        "num_animals": len(grouped),
        "animals": []
    }

    print(f"[INFO] Aggregating features for {len(grouped)} animals...")

    missing_files = []

    for subject_id, slide_ids in grouped.items():
        out_file = animal_dir / f"{subject_id}.pt"

        summary["animals"].append(
            {"subject": subject_id, "num_slides": len(slide_ids)}
        )

        # skip if it is already there
        if out_file.exists():
            continue

        tensors = []
        for slide_id in slide_ids:
            feat_path = slide_dir / f"{slide_id}.pt"

            if feat_path.exists():
                tensors.append(torch.load(feat_path, map_location="cpu"))
            else:
                missing_files.append(feat_path)

        # Save combined tensor or warn
        if tensors:
            combined = torch.cat(tensors, dim=0)
            torch.save(combined, out_file)
        else:
            print(f"[WARN] No valid slide features for animal {subject_id}")

    print("\n[INFO] Finished producing animal-level features.")

    if missing_files:
        print(f"[WARNING] Missing features for {len(missing_files)} slides.")
        for m in missing_files[:10]:
            print(" -", m)
        if len(missing_files) > 10:
            print(" ...")
    else:
        print("[INFO] No missing feature files detected.")

    return summary
