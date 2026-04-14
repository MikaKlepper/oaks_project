#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=7-00:00:00
#SBATCH --job-name="benchmark-all-models"
#SBATCH --output=/data/pathology/projects/mika/repos/oaks_project/logs/slurm-%j-benchmark.out
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/pathology/projects:/data/temporary
#SBATCH --container-image="dockerdex.umcn.nl:5005#clemsgrs/slide2vec:v1.3.0"
#SBATCH --qos=vram
#SBATCH --requeue

set -euo pipefail

echo "============================"
echo "   STARTING BENCHMARK RUN   "
echo "============================"

ENCODER="H_OPTIMUS_1"

REPO_DIR="/data/temporary/mika/repos/oaks_project/pipeline"
SMOKE_MODE="${BENCHMARK_SMOKE:-0}"

pip3 install seaborn torchmil normflows --quiet

export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

cd "$REPO_DIR"

if [ "$SMOKE_MODE" = "1" ]; then
    export BENCHMARK_SMOKE=1
    export BENCHMARK_SKIP_PLOTS=1
    echo "[INFO] Running benchmark smoke test (UCB liver hypertrophy linear calibration n=5)..."
else
    echo "[INFO] Running benchmark..."
fi

python benchmark.py

echo "============================"
echo "      BENCHMARK DONE        "
echo "============================"
