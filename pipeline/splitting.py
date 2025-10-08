import pandas as pd
import os
import random
from collections import Counter
import csv
import yaml  # for reading config files

# Step 1: Prepare hypertrophy labels
def group_labels_per_compound(metadata_df, organ):
    df = metadata_df[metadata_df["ORGAN"] == organ] 
    df["HasHypertrophy"] = df["findings"].str.contains("Hypertrophy", na=False).astype(int)  # make a new hypertrophy column, binary value
    df[["Location", "Severity"]] = df["findings"].str.extract(r"\['Hypertrophy'\s*,\s*'([^']+)'\s*,\s*'([^']+)'")  # extract location and severity if needed
    df["Location"] = df["Location"].where(df["HasHypertrophy"] == 1, None)
    df["Severity"] = df["Severity"].where(df["HasHypertrophy"] == 1, None)

    df = df.rename(columns={"compound_name_clean": "drug_name"})
    df = df[["drug_name", "HasHypertrophy", "Location", "Severity"]]
    
    # Group by drug: collect all slide labels into lists
    labels_by_drug_df = df.groupby("drug_name").agg({
        "HasHypertrophy": list,
        "Location": list,
        "Severity": list
    }).reset_index()
    # Convert to list of tuples
    labels_by_drug = [
        (row["drug_name"], (row["HasHypertrophy"], row["Location"], row["Severity"]))
        for _, row in labels_by_drug_df.iterrows()
    ]

    return labels_by_drug, labels_by_drug_df


# Step 2: Evaluate partitions for class balance
def evaluate_partition(A, B):
    """
    Evaluates the quality of a partition by computing class balance difference.
    Lower values indicate better balance.
    
    Parameters:
    A, B (list of tuples): Partitioned subgroups (drug, (hyps, locs, sevs)).
    
    Returns:
    int: A total imbalance score.
    """
    count_A, count_B = Counter(), Counter()

    for _, (hyps, locs, sevs) in A:
        count_A.update([f"hyp_{h}" for h in hyps])                 # hypertrophy flags
        count_A.update([f"loc_{l}" for l in locs if l])            # only non-None locations
        count_A.update([f"sev_{s}" for s in sevs if s])            # only non-None severities

    for _, (hyps, locs, sevs) in B:
        count_B.update([f"hyp_{h}" for h in hyps])
        count_B.update([f"loc_{l}" for l in locs if l])
        count_B.update([f"sev_{s}" for s in sevs if s])

    imbalance = sum(
        abs(count_A[label] - count_B[label]) 
        for label in set(count_A.keys()).union(set(count_B.keys()))
    )
    return imbalance


def greedy_partition(findings_by_drug):
    """
    Greedy algorithm to split subgroups into two sets while maintaining class balance.
    
    Parameters:
    findings_by_drug (list of tuples): Each subgroup is a tuple (drug name, (hyps, locs, sevs)).
    
    Returns:
    tuple: (A, B, count_A, count_B) where A and B are lists of subgroups,
           and count_A / count_B are Counters of label distributions.
    """
    A, B = [], []
    count_A, count_B = Counter(), Counter()
    
    # shuffle to avoid bias in input ordering
    random.shuffle(findings_by_drug)
    
    for drug, (hyps, locs, sevs) in findings_by_drug:
        # Count occurrences of each class in the subgroup
        finding_counter = (
            Counter([f"hyp_{h}" for h in hyps]) +
            Counter([f"loc_{l}" for l in locs if l]) +
            Counter([f"sev_{s}" for s in sevs if s])
        )
        
        # Calculate imbalance if added to A or B
        imbalance_A = 0
        imbalance_B = 0
        for label in finding_counter:
            imbalance_A += abs((count_A[label] + finding_counter[label]) - count_B[label])
            imbalance_B += abs((count_B[label] + finding_counter[label]) - count_A[label])
        
        # Assign to the set that maintains better balance (if equal, send to B for richness)
        if imbalance_A < imbalance_B:
            A.append((drug, (hyps, locs, sevs)))
            count_A.update(finding_counter)
        else:
            B.append((drug, (hyps, locs, sevs)))
            count_B.update(finding_counter)
    
    return A, B, count_A, count_B


def repeat_partitions(findings_by_drug, output_dir=None, num_repeats=1000):
    """
    Repeats the greedy partitioning multiple times to find the best class-balanced split.
    
    Parameters:
    findings_by_drug (list of tuples): Each subgroup is a tuple (drug name, (hyps, locs, sevs)).
    output_dir (str): Directory to save the results CSV. If None, results are not saved.
    num_repeats (int): Number of random initializations to try.
    
    Returns:
    tuple: (A_drugs, B_drugs, best_A, best_B, best_count_A, best_count_B)
    """
    best_A, best_B = None, None
    best_count_A, best_count_B = None, None
    best_seed = None
    best_score = float('inf')
    results = []

    for seed in range(num_repeats):
        random.seed(seed)
        A, B, count_A, count_B = greedy_partition(findings_by_drug)
        score = evaluate_partition(A, B)
        results.append((seed, score, A, B, count_A, count_B))

        if score < best_score:
            best_score = score
            best_A, best_B = A, B
            best_count_A, best_count_B = count_A, count_B
            best_seed = seed

    # Print results
    print(f"Best score: {best_score} at seed {best_seed}")

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        # 1) Save all runs imbalance scores
        i = 0
        while os.path.exists(f"{output_dir}/partition_scores_{i}.csv"):
            i += 1
        scores_path = f"{output_dir}/partition_scores_{i}.csv"

        # Collect all labels across runs
        all_labels = set()
        for _, _, _, _, count_A, count_B in results:
            all_labels.update(count_A.keys())
            all_labels.update(count_B.keys())
        all_labels = sorted(all_labels)

        with open(scores_path, "w") as f:
            header = "seed,score," + ",".join([f"A_{lbl}" for lbl in all_labels]) + "," + ",".join([f"B_{lbl}" for lbl in all_labels]) + "\n"
            f.write(header)
            for seed, score, _, _, count_A, count_B in results:
                row = [str(seed), str(score)]
                row += [str(count_A.get(lbl, 0)) for lbl in all_labels]
                row += [str(count_B.get(lbl, 0)) for lbl in all_labels]
                f.write(",".join(row) + "\n")

        print(f"Saved all partition scores to {scores_path}")

        # 2) Save summary for best partition
        summary_path = f"{output_dir}/best_partition_summary_{i}.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Label", "Count_A", "Count_B", "Difference"])

            for lbl in all_labels:
                a_val = best_count_A.get(lbl, 0)
                b_val = best_count_B.get(lbl, 0)
                diff = abs(a_val - b_val)
                writer.writerow([lbl, a_val, b_val, diff])

        print(f"Saved best partition summary to {summary_path}")

    # Extract drug names from best splits
    A_drugs = [drug for drug, _ in best_A]
    B_drugs = [drug for drug, _ in best_B]

    return A_drugs, B_drugs, best_A, best_B, best_count_A, best_count_B


def prepare_splits_files(master_df, organ, train_drugs, val_drugs, test_drugs, output_dir,subset_of_slide_ids=None,skip_incorrect_ids=True):

    master_df = master_df[master_df['ORGAN'] == organ]
    master_df["HasHypertrophy"] = master_df["findings"].str.contains("Hypertrophy", na=False).astype(int)  # make a new hypertrophy column, binary value
    master_df[["Location", "Severity"]] = master_df["findings"].str.extract(r"\['Hypertrophy'\s*,\s*'([^']+)'\s*,\s*'([^']+)'")  # extract location and severity if needed
    master_df["Location"] = master_df["Location"].where(master_df["HasHypertrophy"] == 1, None)
    master_df["Severity"] = master_df["Severity"].where(master_df["HasHypertrophy"] == 1, None)
    master_df = master_df.rename(columns={"FILE_location": "wsi_path"})

    # Remove the rows with incorrect UIDs
    if skip_incorrect_ids:
        master_df = master_df[~master_df['bad_UID']]

    # If a subset of slide IDs is provided, filter the master_df
    if subset_of_slide_ids is not None:
        master_df = master_df[master_df['slide_id'].isin(subset_of_slide_ids)]

    # Get the df for each set
    train_df = master_df[master_df['compound_name_clean'].isin(train_drugs)]
    val_df = master_df[master_df['compound_name_clean'].isin(val_drugs)]
    test_df = master_df[master_df['compound_name_clean'].isin(test_drugs)]

    os.makedirs(output_dir, exist_ok=True)
    # Save the train, val, and test dataframes to CSV files
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/val.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)


    # Check if any of the the case IDs in one set are in the other sets
    # This is to ensure that the split is correct
    if any(train_df['subject_organ_UID'].isin(val_df['subject_organ_UID'])):
        raise ValueError("Train and val sets have overlapping case IDs.")
    
    if any(train_df['subject_organ_UID'].isin(test_df['subject_organ_UID'])):
        raise ValueError("Train and test sets have overlapping case IDs.")
    
    if any(val_df['subject_organ_UID'].isin(test_df['subject_organ_UID'])):
        raise ValueError("Val and test sets have overlapping case IDs.")
    
    return train_df, val_df, test_df


def summarize_three_splits(counter_train, counter_val, counter_test, output_path):
    """
    Save a CSV summary comparing Train, Val, and Test distributions.
    """
    # collect all labels across splits
    all_labels = set(counter_train.keys()) | set(counter_val.keys()) | set(counter_test.keys())
    ordered_labels = sorted(all_labels)

    
    # write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Label", "Train", "Val", "Test"])
        for lbl in ordered_labels:
            writer.writerow([
                lbl,
                counter_train.get(lbl, 0),
                counter_val.get(lbl, 0),
                counter_test.get(lbl, 0)
            ])
    print(f"Saved 3-way split summary to {output_path}")


# main function to run the splitting process
def main(config_path="configs/configs.yaml"):

    # load config yaml file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # loading metadata from open TG-GATES
    df = pd.read_csv(config["data"]["metadata_csv"])

    # group data by drug and prepare hypertrophy labels
    labels_by_drug, _ = group_labels_per_compound(df, organ=config["data"]["organ"])

    # split data into train, val, test using greedy partitioning to ensure class balance across splits  and no compound overlap between splits
    # We do this in two steps:

    # First split: train+val vs test (50-50)
    train_val_drugs, test_drugs, train_val_groups, test_groups, counter_train_val, counter_test = repeat_partitions(
        labels_by_drug,
        output_dir=config["splitting"]["partitions_train_val_test"],
        num_repeats=config["splitting"].get("num_repeats", 1000)
    )

    #  Second split: train vs val (50-50 of train_val)
    train_drugs, val_drugs, train_groups, val_groups, counter_train, counter_val = repeat_partitions(
        train_val_groups,
        output_dir=config["splitting"]["partitions_train_val"],
        num_repeats=config["splitting"].get("num_repeats", 1000)
    )

    # prepare and save the final CSV files for each split: so training, validation, test
    prepare_splits_files(
        df,
        config["data"]["organ"],
        train_drugs, val_drugs, test_drugs,
        output_dir=config["data"]["splits_dir"],
        subset_of_slide_ids=None,
        skip_incorrect_ids=True
    )

    # finally save a summary comparing all three splits
    summarize_three_splits(
        counter_train, counter_val, counter_test,
        config["splitting"]["summary_csv"]
    )

# allow running the script directly
if __name__ == "__main__":
    main()
