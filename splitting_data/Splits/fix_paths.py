from pathlib import Path
import pandas as pd

csv_files = [
    "/data/temporary/mika/repos/oaks_project/splitting_data/Splits/train.csv",
    "/data/temporary/mika/repos/oaks_project/splitting_data/Splits/val.csv",
    "/data/temporary/mika/repos/oaks_project/splitting_data/Splits/test.csv",
]

old_prefix = "/data/RBS_PA_CPGARCHIVE"
new_prefix = "/data/pa_cpgarchive"
source_col = "FILE_LOCATION"
new_col = "wsi_path"

def add_normalized_column(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f" File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if source_col not in df.columns:
        print(f" Column '{source_col}' not found in {csv_path.name}")
        return
    df[new_col] = df[source_col].astype(str).str.replace(old_prefix, new_prefix, regex=False)
    df.to_csv(csv_path, index=False)
    print(f" Overwrote {csv_path.name} with new column '{new_col}' ({len(df)} rows).")

for f in csv_files:
    add_normalized_column(f)
