import subprocess
import sys
from pathlib import Path
import time

from utils.experiment_registry import experiment_run_exists

BASE_CONFIG = "configs/base_config.yaml"
TEST_ONLY_DATASETS = {"ucb"}
MIL_PROBES = {"abmil", "clam", "dsmil", "flow"}

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
CALIBRATION_ONLY = True

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
PROBES = ["linear", "mlp", "logreg", "knn", "svm_linear", "svm_rbf", "flow"]

# K_VALUES = [100, 80, 40, 20, 10, 5, 1]
# K_VALUES = [2953]  # for quick testing, use all training samples for tggates
# K_VALUES = [2953, 100, 80, 40, 20, 10, 5, 1]
K_VALUES = ["full", 100]  # full train split + k-shot
# AGGREGATION_METHODS = ["mean","max","min"]
AGGREGATION_METHODS = ["mean", "max", "min"]

TEST_ENCODERS = {"H_OPTIMUS_1"}
# TEST_K_VALUES = {100, 2953}  # 2953 = all training samples for tggates
TEST_K_VALUES = {"full", 100}  # full train split + k-shot
TEST_NON_MIL_AGGS = {"mean"}
DATASETS = {
    "tggates": None,
    "ucb": "/data/temporary/mika/repos/oaks_project/splitting_data/UCB/ucb_test.csv",
}


def build_variants(probe, dataset, target):
    base_cli = {"epochs": 10, "lr": 1e-4, "batch_size": 16}
    if probe == "flow":
        base_cli |= {
            "flow_input_dim": 32,
            "flow_layers": 8,
            "flow_hidden": 128,
            "flow_train_max_tiles": 5000,
            "flow_topk_frac": 0.4,
            "flow_tau_percentile": 95,
        }

    variants = [] if CALIBRATION_ONLY else [{"tag": "default", "cli": base_cli}]
    if dataset in {"ucb", "tggates"} and target in {"liver_hypertrophy", "any_abnormality"}:
        if dataset == "ucb":
            calibration_source_csv = "/data/temporary/mika/repos/oaks_project/splitting_data/UCB/ucb_test.csv"
        else:
            calibration_source_csv = "/data/temporary/mika/repos/oaks_project/splitting_data/TG-GATES/Splits/test.csv"
        for n in CALIBRATION_SAMPLES:
            calibration_cli = {
                **base_cli,
                "calibrate": True,
                "calibration_samples": n,
                "calibration_seed": 42,
            }
            variants.append(
                {
                    "tag": f"calibration_n{n}",
                    "cli": calibration_cli,
                }
            )
    return variants


def stages_for_dataset(dataset):
    """Return stages that should be run for a dataset."""
    return ["test"] if dataset in TEST_ONLY_DATASETS else ["eval", "test"]

def _normalize_k(k):
    return None if k in (None, "full", "all") else int(k)


def experiment_exists_for_variant(model, probe, target, variant, k, agg, dataset, stage):
    calibration_enabled = bool(variant["cli"].get("calibrate", False))
    calibration_samples = variant["cli"].get("calibration_samples")
    calibration_seed = variant["cli"].get("calibration_seed")

    return experiment_run_exists(
        stage=stage,
        dataset=dataset,
        target_task=target,
        experiment_tag=variant["tag"],
        encoder=model,
        probe=probe,
        k_shot=_normalize_k(k),
        aggregation=agg,
        calibration_enabled=calibration_enabled,
        calibration_samples=calibration_samples,
        calibration_seed=calibration_seed,
        feature_type="animal",
    )


def run_experiment(model, probe, target, variant, k, agg, dataset, subset_csv, stage):
    """
    Run a single experiment for a specific stage.
    """
    calibration_enabled = bool(variant["cli"].get("calibrate", False))
    if stage == "test" and dataset in TEST_ONLY_DATASETS and calibration_enabled:
        main_stage = "all"
    else:
        main_stage = "test" if stage == "test" else "all"

    k_arg = _normalize_k(k)
    k_label = "full" if k_arg is None else str(k_arg)
    cmd = [
        sys.executable,
        "main.py",
        "--config", BASE_CONFIG,
        "--dataset", dataset,
        "--model", model,
        "--probe", probe,
        "--target", target,
        "--experiment_tag", variant["tag"],
        "--stage", main_stage,
    ]
    if k_arg is not None:
        cmd += ["--k", str(k_arg)]

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
        f"TARGET={target} | VARIANT={variant['tag']} | MODEL={model} | PROBE={probe} | k={k_label} | AGG={agg}"
    )
    print("====================================================")
    print(" ".join(cmd))
    print("----------------------------------------------------")

    subprocess.run(cmd, check=True)


def build_combos():
    combos = []
    active_datasets = DATASETS
    active_probes = PROBES
    active_targets = TARGETS

    for dataset, subset_csv in active_datasets.items():
        for stage in stages_for_dataset(dataset):
            encoders = TEST_ENCODERS if stage == "test" else ENCODERS
            k_values = TEST_K_VALUES if stage == "test" else K_VALUES
            for model in encoders:
                for probe in active_probes:
                    aggs = ["MIL"] if probe in MIL_PROBES else (
                        TEST_NON_MIL_AGGS if stage == "test" else AGGREGATION_METHODS
                    )
                    for target in active_targets:
                        for variant in build_variants(probe, dataset, target):
                            for k in k_values:
                                for agg in aggs:
                                    combos.append(
                                        (dataset, subset_csv, stage, model, probe, target, variant, k, agg)
                                    )
    return combos, active_datasets


def run_benchmark():
    combos, active_datasets = build_combos()

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

    from plot_benchmarks import (
        run_all_plots,
        combine_mil_and_mean,
        run_all_plots_combined,
    )

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
