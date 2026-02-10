from pathlib import Path
import torch
from tqdm import tqdm

## This script aggregates processed slide features into animal-level features.

# load all pt files in slide_dir, each should be (D,) vector, then aggregate to (D,) animal vector
def load_pt(path: Path):
    """
    Loads a processed slide embedding.
    Should always be (D,) since slide stage already aggregated.
    """
    x = torch.load(path, map_location="cpu")

    if x.ndim != 1:
        raise ValueError(f"Expected (D,) slide vector, got {tuple(x.shape)} in {path}")

    return x


# aggregate slide features to animal level, then save as {animal}.pt in animal_dir
def group_features_by_animal(prepared, min_completion_ratio=0.9):
    """
    Aggregate processed slide features into animal-level features.

    :param prepared: Data dictionary from config file
    :param min_completion_ratio: Minimum ratio of valid slide features to total expected features for an animal
    :return: None
    """
    
    data = prepared["data"]
    df = data["df"]

    slide_dir = Path(data["slide_dir"])
    animal_dir = Path(data["animal_dir"])
    animal_dir.mkdir(parents=True, exist_ok=True)

    embed_dim = int(data["embed_dim"])
    aggregate = data.get("aggregate", "mean")

    # group slide IDs per animal
    groups = df.groupby("subject_organ_UID")["slide_id"].apply(list)
    animals = list(groups.index) # create list of animals for tqdm

    print(f"[ANIMAL] Animal-level aggregation")
    print(f"[ANIMAL] Slide dir : {slide_dir}")
    print(f"[ANIMAL] Output dir: {animal_dir}\n")

    for animal in tqdm(animals, ncols=120):
        out_path = animal_dir / f"{animal}.pt"

        # skip cached
        if out_path.exists():
            continue

        slide_ids = groups[animal]
        # load slide features for this animal
        slide_paths = [slide_dir / f"{sid}.pt" for sid in slide_ids]

        feats = []
        missing = 0

        for p in slide_paths:
            if not p.exists():
                missing += 1
                continue
            try:
                feats.append(load_pt(p))
            except Exception as e:
                print(f"[ERR] Failed loading {p}: {e}")

        total_expected = len(slide_paths)
        total_loaded = len(feats)

        if total_loaded == 0:
            print(f"[SKIP] {animal}: no valid slide features")
            continue

        ratio = total_loaded / total_expected
        if ratio < min_completion_ratio:
            print(f"[SKIP] {animal}: {total_loaded}/{total_expected} ({ratio:.1%})")
            continue

        # stack into (N, D) tensor for each animal, then aggregate across N
        stacked = torch.stack(feats, dim=0)

        if aggregate == "mean":
            animal_vec = stacked.mean(0)
        elif aggregate == "max":
            animal_vec = stacked.max(0).values
        elif aggregate == "min":
            animal_vec = stacked.min(0).values
        else:
            raise ValueError(f"Unknown aggregation '{aggregate}'")

        # Sanity check
        if animal_vec.numel() != embed_dim:
            print(f"[SKIP] {animal}: wrong dimension {animal_vec.numel()} ≠ {embed_dim}")
            continue

        torch.save(animal_vec, out_path)

    print("\n[ANIMAL] Animal-level feature creation complete.\n")
