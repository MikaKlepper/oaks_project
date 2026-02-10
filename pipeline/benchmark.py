import subprocess
import sys
from pathlib import Path
import itertools
import time
import pandas as pd

from plot_benchmarks import run_all_plots


BASE_CONFIG = "configs/base_config.yaml"

# ENCODERS = [
#     "CONCH",
#     "H_OPTIMUS_0",
#     "H_OPTIMUS_1",
#     "UNI",
#     "UNI_2",
#     "VIRCHOW2",
#     "KAIKO",
#     "PHIKON",
#     "PHIKON_V2",
#     "MIDNIGHT12K",
#     "PRISM",
#     "RESNET50",
#     "HIBOU_B",
#     "HIBOU_L",
#     "PROV_GIGAPATH_224_SLIDE",
#     "PROV_GIGAPATH_256_SLIDE",
#     "PROV_GIGAPATH_224_TILE",
#     "PROV_GIGAPATH_256_TILE",
# ]
ENCODERS =["H_OPTIMUS_1"]  # TEMPORARY LIMIT FOR TESTING

PROBES = ["linear", "mlp", "logreg", "knn", "svm_linear", "svm_rbf"]
# K_VALUES = [2953,100, 80, 40, 20, 10, 5, 1]
K_VALUES = [100, 80, 40, 20, 10, 5, 1]
# AGGREGATION_METHODS = ["mean", "max", "min"]
# AGGREGATION_METHODS = ["mean", "max", "min"]  # TEMPORARY LIMIT FOR TESTING
AGGREGATION_METHODS = ["mean"]  # TEMPORARY LIMIT FOR TESTING
EPOCHS = 100


# ============================================================
# Check if experiment already exists (via benchmark CSV)
# ============================================================
def experiment_exists(model, probe, k, agg, stage="eval"):
    benchmark_file = Path("outputs") / stage / f"{agg}_benchmark_results.csv"
    if not benchmark_file.exists():
        return False

    df = pd.read_csv(benchmark_file)

    match = (
        (df["encoder"] == model) &
        (df["probe"] == probe) &
        (df["k_shot"] == k)
    )

    return match.any()


# ============================================================
# Run one experiment
# ============================================================
def run_experiment(model, probe, k, agg):
    cmd = [
        sys.executable,
        "main.py",
        "--config", BASE_CONFIG,
        "--model", model,
        "--probe", probe,
        "--k", str(k),
        "--agg", agg,
        "--epochs", str(EPOCHS),
        "--stage", "all",
    ]

    print("\n====================================================")
    print(f"[BENCHMARK] MODEL={model} | PROBE={probe} | k={k} | AGG={agg}")
    print("====================================================")
    print(" ".join(cmd))
    print("----------------------------------------------------")

    subprocess.run(cmd, check=True)


# ============================================================
# Main benchmark loop (with progress counter)
# ============================================================
def run_benchmark():
    combos = list(itertools.product(ENCODERS, PROBES, K_VALUES, AGGREGATION_METHODS))
    total = len(combos)

    print(f"[BENCHMARK] Total experiments (theoretical): {total}")

    for idx, (model, probe, k, agg) in enumerate(combos, start=1):
        print(f"\n[PROGRESS] {idx}/{total} → MODEL={model} PROBE={probe} k={k} agg={agg}")

        if probe == "knn" and k == 1:
            print(f"[SKIP] knn cannot use k=1")
            continue

        if experiment_exists(model, probe, k, agg, stage="eval") and \
            experiment_exists(model, probe, k, agg, stage="test"):
            print(f"[SKIP] Already benchmarked.")
            continue

        try:
            run_experiment(model, probe, k, agg)
        except subprocess.CalledProcessError as err:
            print(f"[ERROR] MODEL={model} PROBE={probe} k={k} agg={agg}")
            print(err)
            time.sleep(3)
            continue

        print(f"[DONE] MODEL={model} PROBE={probe} k={k} agg={agg}")
        time.sleep(1)

    # ===================================================
    # Generate plots for eval and test
    # ===================================================
    print("\n==============================================")
    print("  GENERATING BENCHMARK PLOTS")
    print("==============================================\n")

    for agg in AGGREGATION_METHODS:
        run_all_plots(agg, stage="eval")
        run_all_plots(agg, stage="test")

    print("\n==============================================")
    print("        ALL BENCHMARKING COMPLETE!")
    print("==============================================\n")


if __name__ == "__main__":
    run_benchmark()
