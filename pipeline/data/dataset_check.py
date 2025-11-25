import pandas as pd
from pathlib import Path


def load_ids(csv_path):
    """Load IDs from CSV (supports slide or animal mode)."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return set()

    df = pd.read_csv(csv_path)
    col = "slide_id" if "slide_id" in df.columns else "subject_organ_UID"
    return set(df[col].astype(str))


def load_feature_ids(feature_dir):
    """Return all .pt feature IDs in given directory."""
    feature_dir = Path(feature_dir)
    if not feature_dir.exists():
        print(f"[ERROR] Feature directory missing: {feature_dir}")
        return set()

    return {p.stem for p in feature_dir.glob("*.pt")}


def check_subset_consistency(prepared):
    """
    Validate:
    - IDs exist in metadata
    - Slide/animal feature files exist
    - Subset IDs match the correct split and do not leak to others
    """

    print("\n========== CHECKING SUBSET CONSISTENCY ==========")

    data = prepared["data"]   

    # Extract fields
    df        = data["df"]
    ids       = set(data["ids"])
    slide_dir = Path(data["slide_dir"])
    feat_dir  = Path(data["features_dir"])
    split     = data["split"]
    ftype     = data["features_type"]
    subset_csv= data.get("subset_csv", None)
    train_csv = data.get("train_csv", None)
    val_csv   = data.get("val_csv", None)
    test_csv  = data.get("test_csv", None)


    print(f"[INFO] Split: {split}")
    print(f"[INFO] Feature type: {ftype}")
    print(f"[INFO] Num loaded IDs: {len(ids)}")

    # step 1: Metadata check
    print("\n--- Checking metadata dataframe ---")
    df_col = "slide_id" if ftype == "slide" else "subject_organ_UID"
    df_ids = set(df[df_col].astype(str))

    missing_in_df = ids - df_ids
    if missing_in_df:
        print(f"[ERROR] {len(missing_in_df)} IDs missing from metadata df!")
        print(list(missing_in_df)[:10])
    else:
        print("[OK] All dataset IDs found in metadata.")

    # step 2: Slide feature check
    print("\n--- Checking slide feature files ---")
    slide_features = load_feature_ids(slide_dir)
    print(f"[INFO] Slide features available: {len(slide_features)} in {slide_dir}")

    # convert animal → slide IDs if needed
    if ftype == "slide":
        ids_slide = ids
    else:
        ids_slide = set(
            df[df["subject_organ_UID"].astype(str).isin(ids)]["slide_id"].astype(str)
        )

    missing_slide = ids_slide - slide_features
    if missing_slide:
        print(f"[ERROR] Missing {len(missing_slide)} slide-level features!")
        print(list(missing_slide)[:10])
    else:
        print("[OK] All slide-level features present.")

    # step 3: Animal feature check (if needed)
    print("\n--- Checking animal feature files ---")
    missing_animal = None
    if ftype == "animal":
        animal_feats = load_feature_ids(feat_dir)
        print(f"[INFO] Animal features available: {len(animal_feats)} in {feat_dir}")

        missing_animal = ids - animal_feats
        if missing_animal:
            print(f"[ERROR] Missing {len(missing_animal)} animal-level features!")
            print(list(missing_animal)[:10])
        else:
            print("[OK] All animal-level feature files present.")

    # step 4: subset + split consistency
    if subset_csv is not None:
        print("\n--- Checking subset + split leakage ---")

        # Load actual subset IDs
        subset_ids_raw = load_ids(subset_csv)

        # Convert subset animal IDs → slide IDs if needed
        if ftype == "slide":
            subset_slide_ids = subset_ids_raw
        else:
            subset_slide_ids = set(
                df[df["subject_organ_UID"].astype(str).isin(subset_ids_raw)]["slide_id"].astype(str)
            )

        # Load full split definitions
        train_ids = load_ids(train_csv) if train_csv else set()
        val_ids   = load_ids(val_csv)   if val_csv   else set()
        test_ids  = load_ids(test_csv)  if test_csv  else set()

        expected = (
            train_ids if split == "train" else
            val_ids   if split == "val"   else
            test_ids
        )

        # Check subset belongs to correct split
        missing_from_split = subset_slide_ids - expected
        if missing_from_split:
            print(f"[ERROR] {len(missing_from_split)} subset IDs do NOT belong to split '{split}'!")
            print(list(missing_from_split)[:10])
        else:
            print("[OK] Subset matches the correct split.")

        # Leakage into other splits
        leak_train = subset_slide_ids & train_ids if split != "train" else set()
        leak_val   = subset_slide_ids & val_ids   if split != "val"   else set()
        leak_test  = subset_slide_ids & test_ids  if split != "test"  else set()

        if leak_train:
            print(f"[ERROR] {len(leak_train)} subset IDs leak into TRAIN!")
        if leak_val:
            print(f"[ERROR] {len(leak_val)} subset IDs leak into VAL!")
        if leak_test:
            print(f"[ERROR] {len(leak_test)} subset IDs leak into TEST!")

        if not (leak_train or leak_val or leak_test):
            print("[OK] No subset leakage detected.")

    print("\n========== CONSISTENCY CHECK COMPLETE ==========\n")
