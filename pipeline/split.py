import pandas as pd
import os
import random
from collections import Counter
import csv
import yaml


def group_labels_per_compound(metadata_df, organ):
    """
    Groups abnormal labels per drug.
    """
    df = metadata_df[metadata_df["ORGAN"] == organ].copy()
    df["abnormal"]= df["abnormal"].astype(int)
    df = df.rename(columns={"compound_name_clean": "drug_name"})
    df = df[["drug_name", "abnormal"]]
    labels_by_drug_df = df.groupby("drug_name").agg({
        "abnormal": list
    }).reset_index()

    labels_by_drug = [
        (row["drug_name"], row["abnormal"])
        for _, row in labels_by_drug_df.iterrows()
    ]
    return labels_by_drug, labels_by_drug_df


def evaluate_partition(A, B):
    """
    Computes imbalance between two splits.
    """
    count_A, count_B = Counter(), Counter()

    for _, (abn_list) in A:
        count_A.update(f"abn_{h}" for h in abn_list)

    for _, (abn_list) in B:
        count_B.update(f"abn_{h}" for h in abn_list)

    return sum(
        abs(count_A[label] - count_B[label])
        for label in set(count_A) | set(count_B)
    )


def greedy_partition(findings_by_drug):
    """
    Greedy split into two balanced partitions.
    """
    A, B = [], []
    count_A, count_B = Counter(), Counter()

    random.shuffle(findings_by_drug)

    for drug, abn_list in findings_by_drug:
        finding_counter = Counter(f"abn_{h}" for h in abn_list)

        imbalance_A = sum(
            abs((count_A[l] + finding_counter[l]) - count_B[l])
            for l in finding_counter
        )

        imbalance_B = sum(
            abs((count_B[l] + finding_counter[l]) - count_A[l])
            for l in finding_counter
        )

        if imbalance_A < imbalance_B:
            A.append((drug, abn_list))
            count_A.update(finding_counter)
        else:
            B.append((drug, abn_list))
            count_B.update(finding_counter)

    return A, B, count_A, count_B


def repeat_partitions(findings_by_drug, num_repeats=1000):
    """
    Runs multiple splits and selects best one.
    """
    best_score = float("inf")

    for seed in range(num_repeats):
        random.seed(seed)

        A, B, count_A, count_B = greedy_partition(findings_by_drug)
        score = evaluate_partition(A, B)

        if score < best_score:
            best_score = score
            best_A, best_B = A, B
            best_count_A, best_count_B = count_A, count_B

    print(f"[Splitting] Best imbalance score: {best_score}")

    A_drugs = [d for d, _ in best_A]
    B_drugs = [d for d, _ in best_B]

    return A_drugs, B_drugs, best_A, best_B, best_count_A, best_count_B


def prepare_splits_files(master_df, organ, train_drugs, val_drugs, test_drugs, output_dir):
    """
    Writes train/val/test CSV files.
    """
    print(f"[Splitting] Overwriting splits in: {output_dir}")

    master_df = master_df[master_df["ORGAN"] == organ].copy()
    master_df["HasAbnormal"] = master_df["abnormal"].astype(int)

    master_df = master_df.rename(columns={"FILE_location": "wsi_path"})

    train_df = master_df[master_df["compound_name_clean"].isin(train_drugs)]
    val_df = master_df[master_df["compound_name_clean"].isin(val_drugs)]
    test_df = master_df[master_df["compound_name_clean"].isin(test_drugs)]

    os.makedirs(output_dir, exist_ok=True)

    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/val.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)

    print(f"[Splitting] Saved:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")

    return train_df, val_df, test_df


def summarize_three_splits(counter_train, counter_val, counter_test, output_path):
    """
    Saves summary CSV of class distribution.
    """
    all_labels = set(counter_train) | set(counter_val) | set(counter_test)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Label", "Train", "Val", "Test"])

        for lbl in sorted(all_labels):
            writer.writerow([
                lbl,
                counter_train.get(lbl, 0),
                counter_val.get(lbl, 0),
                counter_test.get(lbl, 0)
            ])

    print(f"[Splitting] Summary saved to: {output_path}")


def generate_abnormality_splits(metadata_csv, organ, output_dir, num_repeats=1000):
    """
    Generate train/val/test splits for abnormality classification and save them.
    """
    output_dir = str(output_dir)
    df = pd.read_csv(metadata_csv)

    labels_by_drug, _ = group_labels_per_compound(df, organ)

    (
        _train_val_drugs,
        test_drugs,
        train_val_groups,
        _test_groups,
        _counter_train_val,
        counter_test,
    ) = repeat_partitions(labels_by_drug, num_repeats=num_repeats)
    (
        train_drugs,
        val_drugs,
        _train_groups,
        _val_groups,
        counter_train,
        counter_val,
    ) = repeat_partitions(train_val_groups, num_repeats=num_repeats)

    prepare_splits_files(
        df,
        organ,
        train_drugs,
        val_drugs,
        test_drugs,
        output_dir=output_dir,
    )
    summarize_three_splits(
        counter_train,
        counter_val,
        counter_test,
        os.path.join(output_dir, "summary.csv"),
    )


def main(config_path="configs/base_config.yaml"):
    """
    Main entry point.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    generate_abnormality_splits(
        config["data"]["metadata_csv"],
        config["data"]["organ"],
        config["data"]["splits_dir"],
        num_repeats=config["splitting"]["num_repeats"],
    )

def find_missing_feature_slides(metadata_df, organ, encoder="H_OPTIMUS_1"):
    """
    Finds slides missing feature files and returns full metadata rows.
    """

    print("[Check] Filtering metadata...")
    df = metadata_df[metadata_df["ORGAN"] == organ].copy()

    # --- Create and normalize slide_id ---
    if "slide_id" not in df.columns:
        df["slide_id"] = df["FILE_LOCATION"].apply(
            lambda x: os.path.splitext(os.path.basename(x))[0]
        )

    # extract numeric ID to match .pt filenames
    df["slide_id"] = df["slide_id"].astype(str).str.extract(r"(\d+)")[0]

    print(f"[Check] Total slides in metadata: {len(df)}")

    # --- Feature directories ---
    base_path = "/data/temporary/toxicology/TG-GATES"
    dirs_to_check = [
        f"{base_path}/Trainings_FM/{encoder}/features",
        f"{base_path}/Validations_FM/{encoder}/features",
        f"{base_path}/Tests_FM/{encoder}/features",
    ]

    existing_slide_ids = set()

    print("[Check] Scanning feature directories...")

    for d in dirs_to_check:
        if not os.path.exists(d):
            print(f"[Warning] Directory not found: {d}")
            continue

        extracted_ids = {
            # normalize feature filenames → numeric string
            os.path.splitext(f)[0]
            for f in os.listdir(d)
            if f.endswith(".pt")
        }

        print(f"[Check] {d} → {len(extracted_ids)} files")

        # ensure string type
        existing_slide_ids |= {str(x) for x in extracted_ids}

    print(f"[Check] Total unique extracted slides: {len(existing_slide_ids)}")

    # --- Find missing ---
    missing_df = df[~df["slide_id"].isin(existing_slide_ids)].copy()

    print(f"[Check] Missing slides: {len(missing_df)}")

    print("\n[DEBUG] Available columns:")
    print(missing_df.columns.tolist())
    # Save useful info
    missing_df[["slide_id", "FILE_LOCATION"]].to_csv(
        "missing_slides.csv", index=False
    )
    # save wsi paths for potential reprocessing
    missing_wsi_paths = missing_df["FILE_LOCATION"].rename("wsi_path")
    missing_wsi_paths = missing_wsi_paths.str.replace(
    "/data/RBS_PA_CPGARCHIVE",
    "/data/pa_cpgarchive"
)
    missing_wsi_paths.to_csv("missing_wsi_paths.csv", index=False)


    return missing_df

if __name__ == "__main__":
    main()
