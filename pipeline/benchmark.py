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

PROBES = [
    "linear",
    "mlp",
    "logreg",
    "knn",
    "svm_linear",
    "svm_rbf",
]

K_VALUES = [100, 80, 40, 20, 5, 1]

EPOCHS = 100  # You can override this easily


# ============================================================
# Helpers
# ============================================================

def experiment_exists(model, probe, k):
    """
    Checks whether an experiment is already finished 
    (metrics.json exists → evaluation completed).
    """
    exp_root = Path(f"outputs/experiments_benchmark/{model}/{probe}/k{k}")
    metrics_file = exp_root / "eval" / "metrics" / "metrics.json"
    return metrics_file.exists()


def run_experiment(model, probe, k):
    """
    Launch main.py for TRAIN → EVAL.
    """
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

    print("\n====================================================")
    print(f"[BENCHMARK] Running MODEL={model} | PROBE={probe} | k={k}")
    print("====================================================")
    print(" ".join(cmd))
    print("----------------------------------------------------")

    subprocess.run(cmd, check=True)


# ============================================================
# MAIN BENCHMARK LOOP
# ============================================================

def run_benchmark():
    total = len(ENCODERS) * len(PROBES) * len(K_VALUES)
    print(f"[BENCHMARK] Total experiments to run: {total}")

    for model, probe, k in itertools.product(ENCODERS, PROBES, K_VALUES):

        if experiment_exists(model, probe, k):
            print(f"[SKIP] MODEL={model} PROBE={probe} k={k} already done.")
            continue

        try:
            run_experiment(model, probe, k)
        except subprocess.CalledProcessError as err:
            print(f"[ERROR] Failed: MODEL={model} PROBE={probe} k={k}")
            print(err)
            time.sleep(3)
            continue

        print(f"[DONE] MODEL={model} PROBE={probe} k={k}\n")
        time.sleep(1)  # Optional pacing


if __name__ == "__main__":
    run_benchmark()
