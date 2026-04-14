import pandas as pd
from pathlib import Path


def norm(x):
    """
    Normalize a string by stripping whitespace and removing leading zeros.

    Parameters
    ----------
    x : str
        The string to normalize.

    Returns
    -------
    str
        The normalized string, or "0" if the string is empty after normalization.
    """
    s = str(x).strip().lstrip("0")
    return s or "0"


def load_ids(csv_path, *, preferred_col: str | None = None):
    """
    Load a set of slide IDs from a CSV file.

    Parameters
    ----------
    csv_path : str or Path
        The path to the CSV file containing the slide IDs.

    Returns
    -------
    set of str
        A set of normalized slide IDs read from the CSV file.
    """
    if not csv_path:
        return set()

    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[WARN] Missing CSV: {csv_path}")
        return set()

    df = pd.read_csv(csv_path)
    if preferred_col and preferred_col in df.columns:
        col = preferred_col
    elif "slide_id" in df.columns:
        col = "slide_id"
    elif "subject_organ_UID" in df.columns:
        col = "subject_organ_UID"
    elif "animal_number" in df.columns:
        col = "animal_number"
    elif "slide_filename" in df.columns:
        col = "slide_filename"
        df[col] = df[col].astype(str).map(lambda x: Path(x).stem)
    else:
        raise ValueError(
            f"Could not infer ID column from subset CSV: {csv_path}. "
            "Expected one of slide_id, subject_organ_UID, animal_number, slide_filename."
        )
    return {norm(x) for x in df[col].astype(str)}


def check_subset_consistency(prepared):
    """
    Check metadata consistency, feature-bank coverage, and split leakage.

    Parameters
    ----------
    prepared : dict
        A dictionary containing the following information:
            - data : dict
                Contains the following information:
                    - df : pandas.DataFrame
                        The metadata of the dataset
                    - features_type : str
                        The type of feature (slide or animal)
                    - split : str
                        The split to check (train, val, test)
                    - ids : set of str
                        The subset of IDs to check
                    - subset_csv : str or Path or None
                        The path to the subset CSV file (optional)
                    - train_csv : str or Path or None
                        The path to the train CSV file (optional)
                    - val_csv : str or Path or None
                        The path to the validation CSV file (optional)
                    - test_csv : str or Path or None
                        The path to the test CSV file (optional)

    Returns
    -------
    None
    """
    print("\n========== CONSISTENCY CHECK ==========")

    data = prepared["data"]
    df = data["df"].copy()

    ftype = data["features_type"]
    split = data["split"]
    meta_col = "slide_id" if ftype == "slide" else "subject_organ_UID"

    # Normalize metadata IDs
    df[meta_col] = df[meta_col].astype(str).apply(norm)

    dataset_ids = {norm(x) for x in data["ids"]}

    print(f"[INFO] Feature type: {ftype}")
    print(f"[INFO] Split: {split}")
    print(f"[INFO] Dataset size: {len(dataset_ids)}")

    # set of metadata IDs for this feature type (slide vs animal)
    meta_ids = set(df[meta_col])
    missing = dataset_ids - meta_ids

    if missing:
        print(f"[ERROR] {len(missing)} IDs missing from metadata")
        print(list(missing)[:10])
    else:
        print("[OK] Metadata consistency check passed")

    # feature-bank availability check
    feature_map = (
        data.get("feature_entries")
        or data.get("raw_feature_entries")
        or {}
    )
    feats = {norm(k) for k in feature_map.keys()}
    missing = dataset_ids - feats

    print(f"[INFO] Found {len(feats)} feature entries")

    if missing:
        print(f"[ERROR] {len(missing)} missing feature entries")
        print(list(missing)[:10])
    else:
        print("[OK] Feature files check passed")

    # check if subset IDs are present in any other split
    subset_csv = data.get("subset_csv")
    if not subset_csv:
        print("\n[INFO] No subset CSV → skipping leakage check")
        print("\n========== CHECK COMPLETE ==========\n")
        return

    subset_ids = load_ids(subset_csv, preferred_col=meta_col)
    train_ids  = load_ids(data.get("train_csv"), preferred_col=meta_col)
    val_ids    = load_ids(data.get("val_csv"), preferred_col=meta_col)
    test_ids   = load_ids(data.get("test_csv"), preferred_col=meta_col)

    expected = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }[split]

    wrong_split = subset_ids - expected
    if wrong_split:
        print(f"[ERROR] {len(wrong_split)} subset IDs not in '{split}' split")
        print(list(wrong_split)[:10])
    else:
        print("[OK] Subset belongs to correct split")

    leaks = {
        "train": subset_ids & train_ids,
        "val": subset_ids & val_ids,
        "test": subset_ids & test_ids,
    }

    leaks[split] = set()  # ignore correct split

    leaked = False
    for name, ids in leaks.items():
        if ids:
            leaked = True
            print(f"[ERROR] {len(ids)} IDs leak into {name.upper()}")
            print(list(ids)[:10])

    if not leaked:
        print("[OK] No split leakage detected")

    print("\n========== CHECK COMPLETE ==========\n")
