import subprocess
import sys
from pathlib import Path
import itertools
import time

# ============================================================
# USER CONFIG
# ============================================================

BASE_CONFIG = "configs/base_config.yaml"

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

PROBES = ["svm_linear"]

K_VALUES = [100, 80, 40, 20, 5, 1]

AGGREGATION_METHODS = ["mean", "max", "min"]  # add/remove as you want

EPOCHS = 100


# ============================================================
# Helpers
# ============================================================

def experiment_exists(model, probe, k, agg):
    """
    Checks whether an experiment has finished already:
    metrics.json must exist (TRAIN + EVAL done)
    """
    exp_root = Path(f"outputs/experiments_benchmark/{agg}/{model}/{probe}/k{k}")
    metrics_file = exp_root / "eval" / "metrics" / "metrics.json"
    return metrics_file.exists()


def run_experiment(model, probe, k, agg):
    """
    Launch main.py with aggregation override.
    """
    cmd = [
        sys.executable,
        "main.py",
        "--config", BASE_CONFIG,
        "--model", model,
        "--probe", probe,
        "--k", str(k),
        "--agg", agg,            # <--- NEW aggregation override
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
# MAIN BENCHMARK LOOP
# ============================================================

def run_benchmark():
    total = len(ENCODERS) * len(PROBES) * len(K_VALUES) * len(AGGREGATION_METHODS)
    print(f"[BENCHMARK] Total experiments to run: {total}")

    for model, probe, k, agg in itertools.product(ENCODERS, PROBES, K_VALUES, AGGREGATION_METHODS):

        if experiment_exists(model, probe, k, agg):
            print(f"[SKIP] MODEL={model} PROBE={probe} k={k} agg={agg} already done.")
            continue

        try:
            run_experiment(model, probe, k, agg)
        except subprocess.CalledProcessError as err:
            print(f"[ERROR] MODEL={model} PROBE={probe} k={k} agg={agg}")
            print(err)
            time.sleep(3)
            continue

        print(f"[DONE] MODEL={model} PROBE={probe} k={k} agg={agg}\n")
        time.sleep(1)


if __name__ == "__main__":
    run_benchmark()
