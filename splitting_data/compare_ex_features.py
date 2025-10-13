import pandas as pd
from pathlib import Path

def compare_ids_extracted_features(subset_csv, extracted_features_dir, export_missing_csv=None):
    subset_df = pd.read_csv(Path(subset_csv))
    if "FILE_LOCATION" not in subset_df.columns:
        raise KeyError("The CSV MUST contain a 'FILE_LOCATION' column.")
    
    #  Extract WSI IDs from CSV 
    wsi_paths_subset = subset_df["FILE_LOCATION"].dropna().tolist()
    wsi_ids_subset = [Path(p).stem for p in wsi_paths_subset]
    subset_df["slide_id"] = wsi_ids_subset

    # Extract IDs from directory containing feature files 
    extracted_features_ids = [p.stem for p in Path(extracted_features_dir).rglob("*.pt")]

    #  Compare sets 
    missing_feature_ids = set(wsi_ids_subset) - set(extracted_features_ids)
    extra_feature_ids = set(extracted_features_ids) - set(wsi_ids_subset)

    # Print summary 
    print(f"Total slides in CSV: {len(wsi_ids_subset)}")
    print(f"Total extracted feature files: {len(extracted_features_ids)}")
    print(f"Slides in CSV but missing in features: {len(missing_feature_ids)}")
    print(f"Slides in features but not in CSV: {len(extra_feature_ids)}")

    #  Export missing FILE_LOCATION paths 
    if export_missing_csv and missing_feature_ids:
        missing_rows = subset_df[subset_df["slide_id"].isin(missing_feature_ids)]
        missing_wsi_paths = missing_rows[["FILE_LOCATION"]]

        export_path = Path(export_missing_csv)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        missing_wsi_paths.to_csv(export_path, index=False)

        print(f"\n Exported {len(missing_wsi_paths)} missing WSI paths to {export_path}")
    else:
        print("\n No missing features or export skipped.")

    return missing_feature_ids, extra_feature_ids


if __name__ == "__main__":
    subset_csv = "/home/mikaklepper/temporary/repos/oaks_project/splitting_data/Subsets/val_balanced_subset.csv"
    extracted_features_dir = "/data/temporary/toxicology/TG-GATES/liver/UNI/UNI/features"
    export_missing_csv = "/home/mikaklepper/temporary/repos/oaks_project/splitting_data/missing_val_wsi_paths.csv"

    compare_ids_extracted_features(subset_csv, extracted_features_dir, export_missing_csv)
