import json
import ast
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# --------------------------------------------------------
# PARSE findings LIST SAFELY (supports multiple lesions)
# --------------------------------------------------------
def extract_hypertrophy_info(findings_str):
    """
    findings_str example:
      "[['Hypertrophy','Centrilobular','slight',False]]"
      "[['Ground glass',...], ['Hypertrophy','Centrilobular','minimal',False]]"

    Returns (Location, Severity) for hypertrophy or (None, None)
    """

    if not isinstance(findings_str, str):
        return None, None

    try:
        findings = ast.literal_eval(findings_str)
    except:
        return None, None

    if not isinstance(findings, list):
        return None, None

    for entry in findings:
        # entry example = ['Hypertrophy','Centrilobular','slight',False]
        if isinstance(entry, list) and len(entry) >= 3:
            if isinstance(entry[0], str) and entry[0].lower() == "hypertrophy":
                location = entry[1]
                severity = entry[2]
                return location, severity

    return None, None


# --------------------------------------------------------
# LOAD METADATA (corrected hypertrophy parsing)
# --------------------------------------------------------
def load_metadata(metadata_csv, organ):
    df = pd.read_csv(metadata_csv)

    df = df[df["ORGAN"].str.lower() == organ.lower()].copy()

    # Binary label
    df["HasHypertrophy"] = df["findings"].str.contains("Hypertrophy", na=False).astype(int)

    # New robust parsing
    hypertrophy_info = df["findings"].apply(extract_hypertrophy_info)
    df["Location"] = hypertrophy_info.apply(lambda x: x[0])
    df["Severity"] = hypertrophy_info.apply(lambda x: x[1])

    df["subject_organ_UID"] = df["subject_organ_UID"].astype(str)

    print(f"[INFO] Loaded metadata: {len(df)} rows")
    print(df[["subject_organ_UID", "HasHypertrophy", "Location", "Severity"]].head())

    return df


# --------------------------------------------------------
# ENRICH MISCLASSIFIED.JSON WITH METADATA
# --------------------------------------------------------
def enrich_misclassified_with_metadata(mis_path, df_metadata, out_path):
    mis = json.load(open(mis_path))

    wrong_ids = mis["wrong_ids"]
    y_true = mis["y_true_wrong"]
    y_pred = mis["y_pred_wrong"]

    enriched = []

    for slide_id, yt, yp in zip(wrong_ids, y_true, y_pred):

        match = df_metadata[df_metadata["subject_organ_UID"] == slide_id]

        if match.empty:
            enriched.append({
                "id": slide_id,
                "location": None,
                "severity": None,
                "found_in_metadata": False,
                "y_true": yt,
                "y_pred": yp
            })
            continue

        row = match.iloc[0]

        enriched.append({
            "id": slide_id,
            "location": row["Location"],
            "severity": row["Severity"],
            "found_in_metadata": True,
            "y_true": yt,
            "y_pred": yp
        })

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(enriched, f, indent=2)

    print(f"[INFO] Saved enriched misclassified JSON → {out_path}")

    return enriched


# --------------------------------------------------------
# HISTOGRAM GENERATOR
# --------------------------------------------------------
def plot_misclassification_histograms(enriched, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(enriched)
    df = df[df["found_in_metadata"]]   # keep only matched rows

    if df.empty:
        print("[WARN] No rows with metadata match — no histograms generated.")
        return

    # Identify FP vs FN categories
    df["type"] = df.apply(
        lambda r: "FP" if (r.y_true == 0 and r.y_pred == 1)
        else "FN" if (r.y_true == 1 and r.y_pred == 0)
        else None,
        axis=1
    )
    df = df[df["type"].notna()]

    # ---------------- Severity histogram ----------------
    if df["severity"].notna().any():
        plt.figure(figsize=(8, 5))
        df["severity"].dropna().value_counts().plot(kind="bar")
        plt.title("Severity distribution among misclassified")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_dir / "severity_histogram.png", dpi=200)
        plt.close()
    else:
        print("[INFO] No severity values found — skipping severity histogram.")

    # ---------------- Location histogram ----------------
    if df["location"].notna().any():
        plt.figure(figsize=(8, 5))
        df["location"].dropna().value_counts().plot(kind="bar")
        plt.title("Location distribution among misclassified")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_dir / "location_histogram.png", dpi=200)
        plt.close()
    else:
        print("[INFO] No location values found — skipping location histogram.")

    # ---------------- FP vs FN × Severity ----------------
    if df["severity"].notna().any():
        plt.figure(figsize=(10, 6))
        df.groupby(["type", "severity"]).size().unstack(fill_value=0).plot(kind="bar")
        plt.title("FP vs FN by Severity")
        plt.tight_layout()
        plt.savefig(out_dir / "severity_fp_fn.png", dpi=200)
        plt.close()

    # ---------------- FP vs FN × Location ----------------
    if df["location"].notna().any():
        plt.figure(figsize=(10, 6))
        df.groupby(["type", "location"]).size().unstack(fill_value=0).plot(kind="bar")
        plt.title("FP vs FN by Location")
        plt.tight_layout()
        plt.savefig(out_dir / "location_fp_fn.png", dpi=200)
        plt.close()

    print(f"[INFO] Histograms saved → {out_dir}")


# --------------------------------------------------------
# MAIN ENTRY POINT
# --------------------------------------------------------
def run_analysis(
    misclassified_json,
    metadata_csv,
    organ,
    output_json,
    output_plots_dir
):
    df_meta = load_metadata(metadata_csv, organ)

    enriched = enrich_misclassified_with_metadata(
        misclassified_json,
        df_meta,
        output_json
    )

    plot_misclassification_histograms(
        enriched,
        output_plots_dir
    )


# --------------------------------------------------------
# EXAMPLE MAIN
# --------------------------------------------------------
if __name__ == "__main__":

    EXP_ROOT = Path("outputs/experiments_benchmark/mean/CONCH/linear/k100")

    misclassified_json = EXP_ROOT / "eval/metrics/misclassified.json"

    metadata_csv = (
        "/data/pa_cpgarchive/archives/toxicology/open-tg-gates/"
        "metadata_SD/open_tggates_master_list.csv"
    )

    organ = "Liver"

    output_json = EXP_ROOT / "eval/metrics/misclassified_with_metadata.json"
    output_plots_dir = EXP_ROOT / "eval/metrics/misclassified_plots"

    run_analysis(
        misclassified_json=misclassified_json,
        metadata_csv=metadata_csv,
        organ=organ,
        output_json=output_json,
        output_plots_dir=output_plots_dir
    )
