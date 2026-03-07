import subprocess
import sys
from pathlib import Path
import time
import pandas as pd

from plot_benchmarks import (
    run_all_plots,
    combine_mil_and_mean,
    run_all_plots_combined,
)

BASE_CONFIG = "configs/base_config.yaml"
TEST_ONLY_DATASETS = {"ucb"}

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
ENCODERS =["H_OPTIMUS_1"]  # for quick testing, focus on one encoder]

PROBES = [
    "clam",
]
# K_VALUES = [100, 80, 40, 20, 10, 5, 1]
K_VALUES =[2953]  # for quick testing, use all training samples for tggates
AGGREGATION_METHODS = ["mean"]
MIL_PROBES = {"abmil", "clam", "dsmil"}
TEST_ENCODERS = {"H_OPTIMUS_1"}
TEST_K_VALUES = {100, 2953}  # 2953 = all training samples for tggates
TEST_NON_MIL_AGGS = {"mean"}
EPOCHS = 100

DATASETS = {
    "tggates": None,
    "ucb": "/data/temporary/mika/repos/oaks_project/splitting_data/UCB/ucb_test.csv",
}


def stages_for_dataset(dataset):
    """Return stages that should be run for a dataset."""
    return ["test"] if dataset in TEST_ONLY_DATASETS else ["eval", "test"]


def experiment_exists(model, probe, k, agg, dataset, stage):
    benchmark_file = (
        Path("outputs")
        / stage
        / dataset
        / f"{agg}_benchmark_results.csv"
    )

    if not benchmark_file.exists():
        return False

    df = pd.read_csv(benchmark_file)

    match = (
        (df["dataset"] == dataset) &
        (df["encoder"] == model) &
        (df["probe"] == probe) &
        (df["k_shot"] == k)
    )

    return match.any()


def run_experiment(model, probe, k, agg, dataset, subset_csv, stage):
    """
    Run a single experiment for a specific stage.
    """
    main_stage = "test" if stage == "test" else "all"

    cmd = [
        sys.executable,
        "main.py",
        "--config", BASE_CONFIG,
        "--dataset", dataset,
        "--model", model,
        "--probe", probe,
        "--k", str(k),
        "--epochs", str(EPOCHS),
        "--stage", main_stage,
    ]

    if subset_csv is not None:
        cmd += ["--test_subset_csv", subset_csv]

    if probe not in MIL_PROBES:
        cmd += ["--agg", agg]

    print("\n====================================================")
    print(
        f"[BENCHMARK] DATASET={dataset} | STAGE={stage} | "
        f"MODEL={model} | PROBE={probe} | k={k} | AGG={agg}"
    )
    print("====================================================")
    print(" ".join(cmd))
    print("----------------------------------------------------")

    subprocess.run(cmd, check=True)


def run_benchmark():
    combos = []

    for dataset, subset_csv in DATASETS.items():
        for stage in stages_for_dataset(dataset):

            if stage == "test":
                encoders = TEST_ENCODERS
                k_values = TEST_K_VALUES
            else:
                encoders = ENCODERS
                k_values = K_VALUES

            for model in encoders:
                for probe in PROBES:
                    for k in k_values:

                        if probe in MIL_PROBES:
                            combos.append(
                                (dataset, subset_csv, stage, model, probe, k, "MIL")
                            )
                        else:
                            aggs = (
                                TEST_NON_MIL_AGGS if stage == "test"
                                else AGGREGATION_METHODS
                            )
                            for agg in aggs:
                                combos.append(
                                    (dataset, subset_csv, stage, model, probe, k, agg)
                                )

    total = len(combos)
    print(f"[BENCHMARK] Total experiments: {total}")

    for idx, (dataset, subset_csv, stage, model, probe, k, agg) in enumerate(combos, start=1):
        print(
            f"\n[PROGRESS] {idx}/{total} -> "
            f"DATASET={dataset} STAGE={stage} MODEL={model} "
            f"PROBE={probe} k={k} agg={agg}"
        )

        if probe == "knn" and k == 1:
            print("[SKIP] knn cannot use k=1")
            continue

        if experiment_exists(model, probe, k, agg, dataset, stage):
            print(f"[SKIP] Already benchmarked ({stage}).")
            continue

        try:
            run_experiment(model, probe, k, agg, dataset, subset_csv, stage)
        except subprocess.CalledProcessError as err:
            print(
                f"[ERROR] DATASET={dataset} STAGE={stage} "
                f"MODEL={model} PROBE={probe} k={k} agg={agg}"
            )
            print(err)
            time.sleep(3)
            continue

        print(
            f"[DONE] DATASET={dataset} STAGE={stage} "
            f"MODEL={model} PROBE={probe} k={k} agg={agg}"
        )
        time.sleep(1)

    print("\n==============================================")
    print("  GENERATING BENCHMARK PLOTS")
    print("==============================================\n")

    for dataset in DATASETS.keys():
        for stage in ["eval", "test"]:
            path = Path("outputs") / stage / dataset
            if not path.exists():
                continue

            print(f"[PLOTS] Dataset = {dataset} (stage={stage})")

            for agg in AGGREGATION_METHODS:
                run_all_plots(agg, stage=stage, dataset=dataset)

            run_all_plots("MIL", stage=stage, dataset=dataset)
            combine_mil_and_mean(stage=stage, dataset=dataset)
            run_all_plots_combined(stage=stage, dataset=dataset)

    print("\n==============================================")
    print("        ALL BENCHMARKING COMPLETE!")
    print("==============================================\n")


if __name__ == "__main__":
    run_benchmark()
