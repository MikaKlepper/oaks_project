import shutil
from pathlib import Path
import pandas as pd

# config
csv_path = "/data/temporary/mika/repos/oaks_project/splitting_data/FewShotCompoundBalanced/train_fewshot_k100_paths.csv"
output_dir = Path("/data/temporary/mika/repos/oaks_project/wsis/liver/train")
output_dir.mkdir(parents=True, exist_ok=True)

# reading in the csv file
df = pd.read_csv(csv_path)
if "wsi_path" not in df.columns:
    raise ValueError("CSV must contain a 'wsi_path' column")

missing = []
copied = 0
total = len(df)

print(f" Starting WSI transfer to {output_dir} ...")

# copy loop 
for i, src in enumerate(df["wsi_path"], start=1):
    src = Path(src)
    dst = output_dir / src.name

    try:
        if not src.exists():
            missing.append(str(src))
            print(f"[{i}/{total}]  NOT FOUND → {src.name}")
            continue

        shutil.copyfile(src, dst)
        copied += 1
        print(f"[{i}/{total}]  Copied → {src.name}")

    except PermissionError:
        print(f"[{i}/{total}]  Permission denied → {src}")
        missing.append(str(src))

    except Exception as e:
        print(f"[{i}/{total}] ⚠️ Failed ({e}) → {src}")
        missing.append(str(src))

# === SUMMARY ===
print("\n DONE!")
print(f"  → Copied: {copied}")
print(f"  → Missing/Failed: {len(missing)}")

if missing:
    print("\n Missing or failed files (first 10 shown):")
    for m in missing[:10]:
        print("  -", m)
    if len(missing) > 10:
        print("  ... (more not shown)")

