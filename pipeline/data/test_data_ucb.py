import ast
import pandas as pd
from pathlib import Path
from types import SimpleNamespace
from prepare_dataset import (
    _extract_hypertrophy_location_severity,
    _normalize_severity,
)

# --- fake cfg object (only what _load_metadata needs) ---
cfg = SimpleNamespace()
cfg.data = SimpleNamespace()
cfg.data.metadata_csv = (
    "/data/temporary/mika/repos/oaks_project/splitting_data/UCB/ucb_test.csv"
)
cfg.data.organ = "Liver"

# --- run loader ---
df = pd.read_csv(cfg.data.metadata_csv)

# print("Columns:", df.columns.tolist())
print("Rows:", len(df))

# apply hypertrophy parsing
parsed = df.apply(_extract_hypertrophy_location_severity, axis=1)

df["HasHypertrophy"] = parsed.apply(lambda x: x[0])
df["Location"] = parsed.apply(lambda x: x[1])
df["Severity_raw"] = parsed.apply(lambda x: x[2])
df["Severity"] = df["Severity_raw"].apply(_normalize_severity)

print(df[["HasHypertrophy", "Location", "Severity"]].head())
print("Positive hypertrophy count:", df["HasHypertrophy"].sum())

print("\n=== RAW FINDINGS SAMPLE ===")

if "findings" in df.columns:
    print("Using column: findings")
    print(df["findings"].dropna().head(5).tolist())

elif "liver_findings_microscopy" in df.columns:
    print("Using column: liver_findings_microscopy")
    print(df["liver_findings_microscopy"].dropna().head(5).tolist())

else:
    print("⚠️ No known pathology column found")

print("\n================ SEVERITY DISTRIBUTION ================")
print(df["Severity"].value_counts(dropna=False).sort_index())

print("Positive hypertrophy count:", df["HasHypertrophy"].sum())
print(df[df["HasHypertrophy"] == 1][["Location", "Severity_raw", "Severity"]].head(10))

print("\n=== UNIQUE RAW SEVERITY VALUES ===")
print(
    df["Severity_raw"]
    .astype(str)
    .value_counts(dropna=False)
)
