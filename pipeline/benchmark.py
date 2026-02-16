import subprocess
import sys
from pathlib import Path
import itertools
import time
import pandas as pd


from plot_benchmarks import run_all_plots, combine_mil_and_mean, run_all_plots_combined


BASE_CONFIG = "configs/base_config.yaml"

# # Full encoder list (can be reduced for testing)
ENCODERS = [
    "CONCH",
    "H_OPTIMUS_0",
    "H_OPTIMUS_1",
    "UNI",
    "UNI_2",
    "VIRCHOW2",
    "KAIKO",
    "PHIKON",
    "PHIKON_V2",
    "MIDNIGHT12K",
    "PRISM",
    "RESNET50",
    "HIBOU_B",
    "HIBOU_L",
    "PROV_GIGAPATH_224_SLIDE",
    "PROV_GIGAPATH_256_SLIDE",
    "PROV_GIGAPATH_224_TILE",
    "PROV_GIGAPATH_256_TILE",
]

# TEMPORARY LIMIT FOR TESTING
# ENCODERS = ["H_OPTIMUS_1"]

# PROBES = [
#     "linear",
#     "mlp",
#     "logreg",
#     "knn",
#     "svm_linear",
#     "svm_rbf",
#     "abmil",
#     "clam",
# ]
PROBES = ["abmil", "clam", "dsmil"]  # TEMPORARY LIMIT FOR TESTING
# K_VALUES = [2953, 100, 80, 40, 20, 10, 5, 1]
K_VALUES = [100, 80, 40, 20, 10, 5, 1]  # TEMPORARY LIMIT FOR TESTING


# AGGREGATION_METHODS = ["mean", "max", "min"]
AGGREGATION_METHODS = ["mean"]


EPOCHS = 100

MIL_PROBES = {"abmil", "clam", "dsmil"}
    

# check if experiment already exists in eval and test results
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


# function to run a single experiment
def run_experiment(model, probe, k, agg):
    cmd = [
        sys.executable,
        "main.py",
        "--config", BASE_CONFIG,
        "--model", model,
        "--probe", probe,
        "--k", str(k),
        "--epochs", str(EPOCHS),
        "--stage", "all",
    ]

    # ONLY pass --agg for non-MIL probes
    if probe not in MIL_PROBES:
        cmd += ["--agg", agg]

    print("\n====================================================")
    print(f"[BENCHMARK] MODEL={model} | PROBE={probe} | k={k} | AGG={agg}")
    print("====================================================")
    print(" ".join(cmd))
    print("----------------------------------------------------")

    subprocess.run(cmd, check=True)


# main function to run the full benchmark
def run_benchmark():
    combos = []

    for model in ENCODERS:
        for probe in PROBES:
            for k in K_VALUES:
                if probe in MIL_PROBES:
                    # MIL has no aggregation – use a fixed tag for bookkeeping
                    combos.append((model, probe, k, "MIL"))
                else:
                    for agg in AGGREGATION_METHODS:
                        combos.append((model, probe, k, agg))

    total = len(combos)
    print(f"[BENCHMARK] Total experiments (theoretical): {total}")

    for idx, (model, probe, k, agg) in enumerate(combos, start=1):
        print(
            f"\n[PROGRESS] {idx}/{total} -> "
            f"MODEL={model} PROBE={probe} k={k} agg={agg}"
        )

        if probe == "knn" and k == 1:
            print("[SKIP] knn cannot use k=1")
            continue

        if experiment_exists(model, probe, k, agg, stage="eval"):
        # and \
        #    experiment_exists(model, probe, k, agg, stage="test"):
            print("[SKIP] Already benchmarked.")
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

    print("\n==============================================")
    print("  GENERATING BENCHMARK PLOTS")
    print("==============================================\n")

    # pooled aggregations
    for agg in AGGREGATION_METHODS:
        run_all_plots(agg, stage="eval")
        # run_all_plots(agg, stage="test")

    # MIL plots (logged under MIL)
    run_all_plots("MIL", stage="eval")
    # run_all_plots("MIL", stage="test")

    combine_mil_and_mean(stage="eval")
    # combine_mil_and_mean(stage="test")

    run_all_plots_combined(stage="eval")
    # run_all_plots_combined(stage="test")

    print("\n==============================================")
    print("        ALL BENCHMARKING COMPLETE!")
    print("==============================================\n")


if __name__ == "__main__":
    run_benchmark()
