#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=7-00:00:00
#SBATCH --job-name="benchmark-all-models"
#SBATCH --output=/data/pathology/projects/mika/repos/oaks_project/logs/slurm-%j-l-h_optimus_1_benchmark.out
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/pathology/projects:/data/temporary
#SBATCH --container-image="dockerdex.umcn.nl:5005#clemsgrs/slide2vec:v1.3.0"
#SBATCH --qos=low
#SBATCH --requeue

pip3 install seaborn
pip3 install torchmil

echo "============================"
echo "   STARTING BENCHMARK RUN   "
echo "============================"

echo "[INFO] Hostname: $(hostname)"
echo "[INFO] SLURM Job ID: $SLURM_JOB_ID"


REPO_DIR="/data/temporary/mika/repos/oaks_project/pipeline"
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
echo "[INFO] PYTHONPATH → $PYTHONPATH"

cd "$REPO_DIR" || exit 1
echo "[INFO] Working directory: $(pwd)"

echo "[INFO] Running benchmark.py..."
python benchmark.py

echo "============================"
echo "      BENCHMARK DONE        "
echo "============================"
