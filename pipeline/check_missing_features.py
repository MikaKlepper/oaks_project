import pandas as pd
import ast
from collections import Counter
import matplotlib.pyplot as plt


def extract_abnormalities(df):
    """
    Extract all abnormality types per dataframe.
    """
    all_abn = []

    for x in df["findings"].dropna():
        try:
            parsed = ast.literal_eval(x)
            for entry in parsed:
                all_abn.append(entry[0])
        except:
            continue

    return Counter(all_abn)


def compute_distribution(split_dir):
    splits = ["train", "val", "test"]

    results = {}

    for split in splits:
        df = pd.read_csv(f"{split_dir}/{split}.csv")
        counter = extract_abnormalities(df)

        total = sum(counter.values())

        # convert to percentages
        percentages = {k: v / total for k, v in counter.items()}

        results[split] = {
            "counts": counter,
            "percentages": percentages
        }

        print(f"\n=== {split.upper()} ===")
        print("Top abnormalities (counts):")
        print(counter.most_common(10))

    return results


def create_comparison_table(results, output_path="abnormality_distribution.csv"):
    all_labels = set()

    for split in results:
        all_labels |= set(results[split]["counts"].keys())

    rows = []

    for label in sorted(all_labels):
        row = {
            "abnormality": label,
        }

        for split in ["train", "val", "test"]:
            row[f"{split}_count"] = results[split]["counts"].get(label, 0)
            row[f"{split}_pct"] = results[split]["percentages"].get(label, 0)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(f"\nSaved → {output_path}")
    return df


def plot_top_abnormalities(df, top_k=10):
    """
    Plot top-K abnormalities across splits.
    """
    df["total"] = df["train_count"] + df["val_count"] + df["test_count"]
    df = df.sort_values("total", ascending=False).head(top_k)

    labels = df["abnormality"]

    x = range(len(labels))

    plt.figure()
    plt.plot(x, df["train_pct"], label="train")
    plt.plot(x, df["val_pct"], label="val")
    plt.plot(x, df["test_pct"], label="test")

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("percentage")
    plt.title("Top abnormality distribution across splits")
    plt.legend()
    plt.tight_layout()
    plt.savefig("abnormality_distribution.png", dpi=300)
    print("Saved plot → abnormality_distribution.png")

def compute_overall_abnormality(split_dir):
    splits = ["train", "val", "test"]

    print("\n=== OVERALL ABNORMALITY RATE ===")

    for split in splits:
        df = pd.read_csv(f"{split_dir}/{split}.csv")

        total = len(df)
        abnormal = df["abnormal"].sum()

        pct = abnormal / total

        print(f"{split.upper()}:")
        print(f"  Total slides: {total}")
        print(f"  Abnormal:    {abnormal}")
        print(f"  Percentage:  {pct:.4f} ({pct*100:.2f}%)\n")

if __name__ == "__main__":
    split_dir = "/data/temporary/mika/repos/oaks_project/splitting_data/Splits/latest"

    results = compute_distribution(split_dir)
    df = create_comparison_table(results)

    plot_top_abnormalities(df, top_k=15)
    compute_overall_abnormality(split_dir)