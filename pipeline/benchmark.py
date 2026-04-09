import os
import subprocess
import sys
from pathlib import Path
import time
import pandas as pd

from utils.experiment_registry import experiment_run_exists

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
TARGETS = ["liver_hypertrophy", "any_abnormality"]
CALIBRATION_SAMPLES = [5, 10, 25, 50]

# PROBES = [
#     "linear",
#     "mlp",
#     "logreg",
#     "knn",
#     "svm_linear",
#     "svm_rbf",
#     "abmil",
#     "clam",
#     "dsmil",
#     "flow",
# ]
PROBES = ["linear", "flow"]

# K_VALUES = [100, 80, 40, 20, 10, 5, 1]
# K_VALUES =[2953]  # for quick testing, use all training samples for tggates
# K_VALUES = [2953, 100, 80, 40, 20, 10, 5, 1]
K_VALUES = [100]  # for quick testing, skip the all-sample setting
# AGGREGATION_METHODS = ["mean","max","min"]
AGGREGATION_METHODS = ["mean","max","min"]
MIL_PROBES = {"abmil", "clam", "dsmil", "flow"}

TEST_ENCODERS = {"H_OPTIMUS_1"}
# TEST_K_VALUES = {100, 2953}  # 2953 = all training samples for tggates
TEST_K_VALUES = {100}  # 2953 = all training samples for tggates
TEST_NON_MIL_AGGS = {"mean"}
DATASETS = {
    "tggates": None,
    "ucb": "/data/temporary/mika/repos/oaks_project/splitting_data/UCB/ucb_test.csv",
}
SMOKE_MODE = os.getenv("BENCHMARK_SMOKE", "0").lower() in {"1", "true", "yes"}
SKIP_PLOTS = os.getenv("BENCHMARK_SKIP_PLOTS", "0").lower() in {"1", "true", "yes"}


def load_plot_functions():
    from plot_benchmarks import (
        run_all_plots,
        combine_mil_and_mean,
        run_all_plots_combined,
    )
    return run_all_plots, combine_mil_and_mean, run_all_plots_combined


def build_variants(probe, dataset, target):
    if probe == "linear":
        variants = [
            {
                "tag": "linear_baseline",
                "cli": {
                    "epochs": 10,
                    "lr": 1e-4,
                    "batch_size": 16,
                },
            }
        ]
        if dataset == "ucb" and target == "liver_hypertrophy":
            for n in CALIBRATION_SAMPLES:
                variants.append(
                    {
                        "tag": f"linear_calibration_n{n}",
                        "cli": {
                            "epochs": 10,
                            "lr": 1e-4,
                            "batch_size": 16,
                            "calibrate": True,
                            "calibration_samples": n,
                            "calibration_seed": 7,
                            "calibration_source_csv": "/data/temporary/mika/repos/oaks_project/splitting_data/UCB/ucb_test.csv",
                        },
                    }
                )
        if SMOKE_MODE and dataset == "ucb" and target == "liver_hypertrophy":
            return [v for v in variants if v["tag"] == "linear_calibration_n5"]
        return variants

    if probe == "flow":
        variants = [
            {
                "tag": "flow_baseline",
                "cli": {
                    "epochs": 10,
                    "lr": 1e-4,
                    "batch_size": 16,
                    "flow_input_dim": 32,
                    "flow_layers": 8,
                    "flow_hidden": 128,
                    "flow_train_max_tiles": 5000,
                    "flow_topk_frac": 0.4,
                    "flow_tau_percentile": 95,
                },
            }
        ]
        if dataset == "ucb" and target == "liver_hypertrophy":
            for n in CALIBRATION_SAMPLES:
                variants.append(
                    {
                        "tag": f"flow_calibration_n{n}",
                        "cli": {
                            "epochs": 10,
                            "lr": 1e-4,
                            "batch_size": 16,
                            "flow_input_dim": 32,
                            "flow_layers": 8,
                            "flow_hidden": 128,
                            "flow_train_max_tiles": 5000,
                            "flow_topk_frac": 0.4,
                            "flow_tau_percentile": 95,
                            "calibrate": True,
                            "calibration_samples": n,
                            "calibration_seed": 7,
                            "calibration_source_csv": "/data/temporary/mika/repos/oaks_project/splitting_data/UCB/ucb_test.csv",
                        },
                    }
                )
        if SMOKE_MODE and dataset == "ucb" and target == "liver_hypertrophy":
            return [v for v in variants if v["tag"] == "flow_calibration_n5"]
        return variants

    return [{"tag": f"{probe}_default", "cli": {}}]


def stages_for_dataset(dataset):
    """Return stages that should be run for a dataset."""
    return ["test"] if dataset in TEST_ONLY_DATASETS else ["eval", "test"]


def experiment_exists(model, probe, target, experiment_tag, k, agg, dataset, stage):
    calibration_enabled = False
    calibration_samples = None
    calibration_seed = None
    registry_exists = experiment_run_exists(
        stage=stage,
        dataset=dataset,
        target_task=target,
        experiment_tag=experiment_tag,
        encoder=model,
        probe=probe,
        k_shot=k,
        aggregation=agg,
        calibration_enabled=calibration_enabled,
        calibration_samples=calibration_samples,
        calibration_seed=calibration_seed,
        feature_type="animal",
    )
    if registry_exists:
        return True

    benchmark_file = (
        Path("outputs")
        / stage
        / dataset
        / f"{agg}_benchmark_results.csv"
    )

    if not benchmark_file.exists():
        return False

    df = pd.read_csv(benchmark_file)
    if "target_task" not in df.columns:
        df["target_task"] = "liver_hypertrophy"
    if "experiment_tag" not in df.columns:
        df["experiment_tag"] = "legacy"

    match = (
        (df["dataset"] == dataset) &
        (df["target_task"] == target) &
        (df["experiment_tag"] == experiment_tag) &
        (df["encoder"] == model) &
        (df["probe"] == probe) &
        (df["k_shot"] == k)
    )

    return match.any()


def experiment_exists_for_variant(model, probe, target, variant, k, agg, dataset, stage):
    calibration_enabled = bool(variant["cli"].get("calibrate", False))
    calibration_samples = variant["cli"].get("calibration_samples")
    calibration_seed = variant["cli"].get("calibration_seed")

    if experiment_run_exists(
        stage=stage,
        dataset=dataset,
        target_task=target,
        experiment_tag=variant["tag"],
        encoder=model,
        probe=probe,
        k_shot=k,
        aggregation=agg,
        calibration_enabled=calibration_enabled,
        calibration_samples=calibration_samples,
        calibration_seed=calibration_seed,
        feature_type="animal",
    ):
        return True

    return experiment_exists(model, probe, target, variant["tag"], k, agg, dataset, stage)


def run_experiment(model, probe, target, variant, k, agg, dataset, subset_csv, stage):
    """
    Run a single experiment for a specific stage.
    """
    calibration_enabled = bool(variant["cli"].get("calibrate", False))
    if stage == "test" and dataset in TEST_ONLY_DATASETS and calibration_enabled:
        main_stage = "all"
    else:
        main_stage = "test" if stage == "test" else "all"

    cmd = [
        sys.executable,
        "main.py",
        "--config", BASE_CONFIG,
        "--dataset", dataset,
        "--model", model,
        "--probe", probe,
        "--target", target,
        "--experiment_tag", variant["tag"],
        "--k", str(k),
        "--stage", main_stage,
    ]

    for key, value in variant["cli"].items():
        if isinstance(value, bool):
            if value:
                cmd += [f"--{key}"]
            continue
        if value is None:
            continue
        cmd += [f"--{key}", str(value)]

    if subset_csv is not None:
        cmd += ["--test_subset_csv", subset_csv]

    if probe not in MIL_PROBES:
        cmd += ["--agg", agg]

    print("\n====================================================")
    print(
        f"[BENCHMARK] DATASET={dataset} | STAGE={stage} | "
        f"TARGET={target} | VARIANT={variant['tag']} | MODEL={model} | PROBE={probe} | k={k} | AGG={agg}"
    )
    print("====================================================")
    print(" ".join(cmd))
    print("----------------------------------------------------")

    subprocess.run(cmd, check=True)


def run_benchmark():
    combos = []
    active_datasets = {"ucb": DATASETS["ucb"]} if SMOKE_MODE else DATASETS
    active_probes = ["linear"] if SMOKE_MODE else PROBES
    active_targets = ["liver_hypertrophy"] if SMOKE_MODE else TARGETS

    if SMOKE_MODE:
        print("[BENCHMARK] Smoke mode enabled -> running only UCB liver hypertrophy linear calibration n=5.")

    for dataset, subset_csv in active_datasets.items():
        for stage in stages_for_dataset(dataset):

            if stage == "test":
                encoders = TEST_ENCODERS
                k_values = TEST_K_VALUES
            else:
                encoders = ENCODERS
                k_values = K_VALUES

            for model in encoders:
                for probe in active_probes:
                    for target in active_targets:
                        for variant in build_variants(probe, dataset, target):
                            for k in k_values:

                                if probe in MIL_PROBES:
                                    combos.append(
                                        (dataset, subset_csv, stage, model, probe, target, variant, k, "MIL")
                                    )
                                else:
                                    aggs = (
                                        TEST_NON_MIL_AGGS if stage == "test"
                                        else AGGREGATION_METHODS
                                    )
                                    for agg in aggs:
                                        combos.append(
                                            (dataset, subset_csv, stage, model, probe, target, variant, k, agg)
                                        )

    total = len(combos)
    print(f"[BENCHMARK] Total experiments: {total}")

    for idx, (dataset, subset_csv, stage, model, probe, target, variant, k, agg) in enumerate(combos, start=1):
        print(
            f"\n[PROGRESS] {idx}/{total} -> "
            f"DATASET={dataset} STAGE={stage} MODEL={model} "
            f"TARGET={target} VARIANT={variant['tag']} PROBE={probe} k={k} agg={agg}"
        )

        if probe == "knn" and k == 1:
            print("[SKIP] knn cannot use k=1")
            continue

        if experiment_exists_for_variant(model, probe, target, variant, k, agg, dataset, stage):
            print(f"[SKIP] Already benchmarked ({stage}).")
            continue

        try:
            run_experiment(model, probe, target, variant, k, agg, dataset, subset_csv, stage)
        except subprocess.CalledProcessError as err:
            print(
                f"[ERROR] DATASET={dataset} STAGE={stage} "
                f"TARGET={target} VARIANT={variant['tag']} MODEL={model} PROBE={probe} k={k} agg={agg}"
            )
            print(err)
            time.sleep(3)
            continue

        print(
            f"[DONE] DATASET={dataset} STAGE={stage} "
            f"TARGET={target} VARIANT={variant['tag']} MODEL={model} PROBE={probe} k={k} agg={agg}"
        )
        time.sleep(1)

    if SKIP_PLOTS or SMOKE_MODE:
        print("\n[BENCHMARK] Skipping plot generation.")
        print("\n==============================================")
        print("        ALL BENCHMARKING COMPLETE!")
        print("==============================================\n")
        return

    run_all_plots, combine_mil_and_mean, run_all_plots_combined = load_plot_functions()

    print("\n==============================================")
    print("  GENERATING BENCHMARK PLOTS")
    print("==============================================\n")

    for dataset in active_datasets.keys():
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
