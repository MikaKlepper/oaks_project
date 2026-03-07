import pandas as pd
from pathlib import Path

# Input CSV files
# files = {
#     "min": "/data/temporary/mika/repos/oaks_project/pipeline/outputs/eval/backup_old/min/min_benchmark_results.csv",
#     "mean": "/data/temporary/mika/repos/oaks_project/pipeline/outputs/eval/backup_old/mean/mean_benchmark_results.csv",
#     "max": "/data/temporary/mika/repos/oaks_project/pipeline/outputs/eval/backup_old/max/max_benchmark_results.csv",
# }

# files = {
#     "min": "/data/temporary/mika/repos/oaks_project/pipeline/outputs/eval/min_benchmark_results.csv",
#     "mean": "/data/temporary/mika/repos/oaks_project/pipeline/outputs/eval/mean_benchmark_results_without_full_train.csv",
#     "max": "/data/temporary/mika/repos/oaks_project/pipeline/outputs/eval/max_benchmark_results.csv",
# }
files ={"mean": "/data/temporary/mika/repos/oaks_project/pipeline/outputs/eval/tggates/mean_benchmark_results_without_full_training.csv",
        "max": "/data/temporary/mika/repos/oaks_project/pipeline/outputs/eval/tggates/max_benchmark_results.csv",
        "min": "/data/temporary/mika/repos/oaks_project/pipeline/outputs/eval/tggates/min_benchmark_results.csv",
        "MIL": "/data/temporary/mika/repos/oaks_project/pipeline/outputs/eval/tggates/mil_benchmark_results_without_full_training.csv"
        }


results = []

for name, path in files.items():
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    if "roc_auc" not in df.columns:
        raise ValueError(f"'roc_auc' column not found in {path}")

    roc_auc = df["roc_auc"]

    results.append({
        "setting": name,
        "mean_roc_auc": roc_auc.mean(),
        "median_roc_auc": roc_auc.median(),
        "std_roc_auc": roc_auc.std(),
        "num_rows": len(roc_auc),
    })

# Create summary table
summary_df = pd.DataFrame(results)

# Save output
output_path = Path(
    "/data/temporary/mika/repos/oaks_project/pipeline/outputs/eval/miccai/tggates/roc_auc_summary.csv"
)
summary_df.to_csv(output_path, index=False)

print("ROC-AUC summary saved to:")
print(output_path)
print()
print(summary_df)


# file= "/data/temporary/mika/repos/oaks_project/splitting_data/FewShotCompoundBalanced/train_fewshot_k100.csv"
# val_df = pd.read_csv(file)
# # compute number of positive and negative samples
# num_positive = val_df['HasHypertrophy'].sum()
# num_negative = len(val_df) - num_positive
# print(f"Number of positive samples: {num_positive}")
# print(f"Number of negative samples: {num_negative}")
# print(f"Total samples: {len(val_df)}")

# # compute severity distribution
# severity_counts = val_df['Severity'].value_counts().sort_index()
# print("Severity distribution:")
# for severity, count in severity_counts.items():
#     print(f"  Severity {severity}: {count} samples")

# # compute number of compounds for positive and negative samples
# positive_compounds = val_df[val_df['HasHypertrophy'] == 1]['compound_name_clean'].nunique()
# negative_compounds = val_df[val_df['HasHypertrophy'] == 0]['compound_name_clean'].nunique()
# print(f"Number of unique compounds in positive samples: {positive_compounds}")
# print(f"Number of unique compounds in negative samples: {negative_compounds}")
# print(f"Total unique compounds: {val_df['compound_name_clean'].nunique()}")



# # Load test set
# test_file = "/data/temporary/mika/repos/oaks_project/splitting_data/Splits/train.csv"
# test_df = pd.read_csv(test_file)

# required_severities = {"minimal", "slight", "moderate", "severe"}

# # Normalize severity
# test_df["Severity_norm"] = test_df["Severity"].str.lower()

# # Find compounds with all severities
# compound_severities = (
#     test_df.groupby("compound_name_clean")["Severity_norm"]
#     .apply(set)
# )

# valid_compounds = compound_severities[
#     compound_severities.apply(lambda s: required_severities.issubset(s))
# ].index

# print(f"Number of compounds with all 4 severity levels: {len(valid_compounds)}")

# if len(valid_compounds) == 0:
#     print("No compound found with all 4 severity levels.")
# else:
#     print("\nOne file path per severity for each compound:\n")

#     for compound in valid_compounds:
#         subset = test_df[test_df["compound_name_clean"] == compound]

#         print(f"Compound: {compound}")

#         for severity in ["minimal", "slight", "moderate", "severe"]:
#             row = subset[subset["Severity_norm"] == severity].iloc[0]
#             file_path = row["FILE_LOCATION"]

#             print(f"  {severity:8s}: {file_path}")

#         print("-" * 60)
