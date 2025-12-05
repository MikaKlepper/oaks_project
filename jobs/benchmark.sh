#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=7-00:00:00
#SBATCH --job-name="benchmark-all-models"
#SBATCH --output=/data/temporary/mika/repos/oaks_project/logs/slurm-%j-l-benchmark.out
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/temporary:/data/temporary
#SBATCH --container-image="dockerdex.umcn.nl:5005#clemsgrs/slide2vec:v1.3.0"
#SBATCH --qos=low
#SBATCH --requeue


echo "============================"
echo "   STARTING BENCHMARK RUN   "
echo "============================"

echo "[INFO] Hostname: $(hostname)"
echo "[INFO] SLURM Job ID: $SLURM_JOB_ID"

# ----------------------------------------------------------
# Set repository path & PYTHONPATH
# ----------------------------------------------------------
REPO_DIR="/data/temporary/mika/repos/oaks_project/pipeline"
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
echo "[INFO] PYTHONPATH → $PYTHONPATH"

cd "$REPO_DIR" || exit 1
echo "[INFO] Working directory: $(pwd)"

# ----------------------------------------------------------
# Run the benchmarking script
# ----------------------------------------------------------
echo "[INFO] Running benchmark.py..."
python benchmark.py

echo "============================"
echo "      BENCHMARK DONE        "
echo "============================"
