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
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return set()

    df = pd.read_csv(csv_path)
    col = "slide_id" if "slide_id" in df.columns else "subject_organ_UID"
    return {norm(x) for x in df[col].astype(str)}


def load_feature_ids(feature_dir):
    """Return all normalized .pt feature IDs in the directory."""
    feature_dir = Path(feature_dir)
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

    # Normalize metadata fields
    if "slide_id" in df.columns:
        df["slide_id"] = df["slide_id"].apply(norm)
    if "subject_organ_UID" in df.columns:
        df["subject_organ_UID"] = df["subject_organ_UID"].apply(norm)

    ids           = {norm(x) for x in data["ids"]}

    # ---- IMPORTANT: directories have changed ----
    raw_slide_dir = Path(data["slide_dir"])      # raw tile bags (not used for checking)
    slide_feat_dir = Path(data["features_dir"])  # processed slide features (D,)
    animal_feat_dir = Path(data.get("animal_dir", ""))  # processed animal features

    split     = data["split"]

    subset_csv= data.get("subset_csv", None)
    train_csv = data.get("train_csv", None)
    val_csv   = data.get("val_csv", None)
    test_csv  = data.get("test_csv", None)

    print(f"[INFO] Split: {split}")
    print(f"[INFO] Feature type: {ftype}")
    print(f"[INFO] Num loaded IDs: {len(ids)}")

    # ------------------------------
    # 1) metadata consistency
    # ------------------------------
    print("\n--- Checking metadata dataframe ---")

    df_col = "slide_id" if ftype == "slide" else "subject_organ_UID"
    df_ids = set(df[df_col])

    missing_in_df = ids - df_ids
    if missing_in_df:
        print(f"[ERROR] {len(missing_in_df)} IDs missing from metadata df!")
        print(list(missing_in_df)[:10])
    else:
        print("[OK] All dataset IDs found in metadata.")

    # ------------------------------
    # 2) processed slide features
    # ------------------------------
    if ftype == "slide":
        print("\n--- Checking processed slide feature files ---")
        processed_slide_ids = load_feature_ids(slide_feat_dir)
        print(f"[INFO] Processed slide features available: {len(processed_slide_ids)} in {slide_feat_dir}")

        missing_slide = ids - processed_slide_ids
        if missing_slide:
            print(f"[ERROR] Missing {len(missing_slide)} processed slide features!")
            print(list(missing_slide)[:10])
        else:
            print("[OK] All processed slide features present.")

    # ------------------------------
    # 3) processed animal features
    # ------------------------------
    if ftype == "animal":
        print("\n--- Checking processed animal feature files ---")
        processed_animals = load_feature_ids(animal_feat_dir)
        print(f"[INFO] Animal features available: {len(processed_animals)} in {animal_feat_dir}")

        missing_animal = ids - processed_animals
        if missing_animal:
            print(f"[ERROR] Missing {len(missing_animal)} animal-level features!")
            print(list(missing_animal)[:10])
        else:
            print("[OK] All animal-level features present.")

    # ------------------------------
    # 4) Subset leakage
    # ------------------------------
    if subset_csv is not None:
        print("\n--- Checking subset + split leakage ---")

        subset_ids_raw = load_ids(subset_csv)

        if ftype == "slide":
            subset_slide_ids = subset_ids_raw
        else:
            subset_slide_ids = set(df[df["subject_organ_UID"].isin(subset_ids_raw)]["slide_id"])

        train_ids = load_ids(train_csv) if train_csv else set()
        val_ids   = load_ids(val_csv)   if val_csv else set()
        test_ids  = load_ids(test_csv)  if test_csv else set()

        expected = (
            train_ids if split == "train" else
            val_ids   if split == "val"   else
            test_ids
        )

        missing_from_split = subset_slide_ids - expected
        if missing_from_split:
            print(f"[ERROR] {len(missing_from_split)} subset IDs do NOT belong to split '{split}'!")
            print(list(missing_from_split)[:10])
        else:
            print("[OK] Subset IDs correctly belong to split.")

        # leakage checks
        leaks = {
            "TRAIN": subset_slide_ids & train_ids if split != "train" else set(),
            "VAL":   subset_slide_ids & val_ids   if split != "val"   else set(),
            "TEST":  subset_slide_ids & test_ids  if split != "test"  else set(),
        }

        leaked = False
        for name, leak in leaks.items():
            if leak:
                leaked = True
                print(f"[ERROR] {len(leak)} subset IDs leak into {name}!")
        if not leaked:
            print("[OK] No subset leakage detected.")

    print("\n========== CONSISTENCY CHECK COMPLETE ==========\n")
