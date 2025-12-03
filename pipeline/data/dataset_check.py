import pandas as pd
from pathlib import Path


# =========================
# NORMALIZATION
# =========================
def norm(x):
    """Normalize ID for consistent matching."""
    s = str(x).strip()
    s = s.lstrip("0")
    return s if s != "" else "0"


# =========================
# LOADERS
# =========================
def load_ids(csv_path):
    """Load normalized IDs from CSV (supports slide or animal mode)."""
    if csv_path is None:
        return set()

    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[WARN] CSV not found: {csv_path}")
        return set()

    df = pd.read_csv(csv_path)
    if "slide_id" in df.columns:
        col = "slide_id"
    else:
        col = "subject_organ_UID"

    return {norm(x) for x in df[col].astype(str)}


def load_feature_ids(feature_dir: Path):
    """Return all normalized .pt feature IDs in the directory."""
    if not feature_dir.exists():
        print(f"[ERROR] Feature directory missing: {feature_dir}")
        return set()

    return {norm(p.stem) for p in feature_dir.glob("*.pt")}


# =========================
# CONSISTENCY CHECK
# =========================
def check_subset_consistency(prepared):

    print("\n========== CHECKING SUBSET CONSISTENCY ==========")

    data = prepared["data"]

    df        = data["df"].copy()
    ftype     = data["features_type"]
    split     = data["split"]

    # Normalize metadata
    if "slide_id" in df.columns:
        df["slide_id"] = df["slide_id"].apply(norm)
    if "subject_organ_UID" in df.columns:
        df["subject_organ_UID"] = df["subject_organ_UID"].apply(norm)

    ids        = {norm(x) for x in data["ids"]}

    # Directories for processed features
    slide_feat_dir  = Path(data["slide_dir"])     # processed slide features
    animal_feat_dir = Path(data["animal_dir"])    # processed animal features
    features_dir     = Path(data["features_dir"])  # the one actually used by dataset

    subset_csv = data.get("subset_csv", None)
    train_csv  = data.get("train_csv", None)
    val_csv    = data.get("val_csv", None)
    test_csv   = data.get("test_csv", None)

    print(f"[INFO] Split: {split}")
    print(f"[INFO] Feature type: {ftype}")
    print(f"[INFO] Num loaded IDs: {len(ids)}")

    # ------------------------------
    # 1) Metadata consistency
    # ------------------------------
    print("\n--- Checking metadata dataframe ---")

    meta_col = "slide_id" if ftype == "slide" else "subject_organ_UID"
    df_ids = set(df[meta_col])

    missing_meta = ids - df_ids
    if missing_meta:
        print(f"[ERROR] {len(missing_meta)} IDs missing from metadata df!")
        print(list(missing_meta)[:10])
    else:
        print("[OK] All dataset IDs found in metadata.")

    # ------------------------------
    # 2) Check processed feature files
    # ------------------------------
    print("\n--- Checking processed feature files ---")

    processed_ids = load_feature_ids(features_dir)
    print(f"[INFO] Found {len(processed_ids)} processed feature vectors in {features_dir}")

    missing_feat = ids - processed_ids
    if missing_feat:
        print(f"[ERROR] Missing {len(missing_feat)} processed features!")
        print(list(missing_feat)[:10])
    else:
        print("[OK] All processed features present.")

    # ------------------------------
    # 3) Subset leakage check
    # ------------------------------
    if subset_csv is not None:
        print("\n--- Checking subset + split leakage ---")

        subset_ids = load_ids(subset_csv)

        # Expected IDs by split
        train_ids = load_ids(train_csv)
        val_ids   = load_ids(val_csv)
        test_ids  = load_ids(test_csv)

        if split == "train":
            expected = train_ids
        elif split == "val":
            expected = val_ids
        else:
            expected = test_ids

        # Check subset in correct split
        missing_from_split = subset_ids - expected
        if missing_from_split:
            print(f"[ERROR] {len(missing_from_split)} subset IDs do NOT belong to split '{split}'!")
            print(list(missing_from_split)[:10])
        else:
            print("[OK] Subset IDs correctly belong to split.")

        # Leakage into wrong splits
        leaks = {
            "TRAIN": subset_ids & train_ids if split != "train" else set(),
            "VAL":   subset_ids & val_ids   if split != "val"   else set(),
            "TEST":  subset_ids & test_ids  if split != "test"  else set(),
        }

        leaked = False
        for name, leak in leaks.items():
            if leak:
                leaked = True
                print(f"[ERROR] {len(leak)} subset IDs leak into {name}!")
                print(list(leak)[:10])

        if not leaked:
            print("[OK] No subset leakage detected.")

    print("\n========== CONSISTENCY CHECK COMPLETE ==========\n")
