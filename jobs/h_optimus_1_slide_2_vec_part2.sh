#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=7-00:00:00
#SBATCH --job-name="l-h_optimus_1_missing_slides_part_2"
#SBATCH --output=/data/pathology/projects/mika/repos/oaks_project/logs/slurm-%j-l-h_optimus_1_missing_slides_part_2.out
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/pathology/projects:/data/temporary
#SBATCH --container-image="dockerdex.umcn.nl:5005#clemsgrs/slide2vec:v1.3.0"
#SBATCH --qos=vram
#SBATCH --requeue

echo "RUNNING SLIDE2VEC WITH H-OPTIMUS-1 ON TG-GATES DATASET (missing slides PART 2)"


REPO_DIR="/data/temporary/mika/repos/oaks_project/slide_2_vec"
CONFIG_PATH="$REPO_DIR/yaml_configs/liver_h_optimus_1_part2.yaml"

SCRATCH_BASE="/scratch_mikaklepper_tg_gates"
SCRATCH_OUTPUT_DIR="$SCRATCH_BASE/outputs/H_OPTIMUS_1"
FINAL_OUTPUT_DIR="/data/temporary/toxicology/TG-GATES/Missing_slides_FM/features_part_2/H_OPTIMUS_1"

echo "Creating required directories..."
mkdir -p "$SCRATCH_BASE" "$SCRATCH_OUTPUT_DIR" "$FINAL_OUTPUT_DIR"

# --------------------------------------------------
# HuggingFace cache on scratch
# --------------------------------------------------

export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
if [ -z "$HF_TOKEN" ]; then
  echo "HF_TOKEN is not set. Export HF_TOKEN or HUGGINGFACE_HUB_TOKEN before running."
  exit 1
fi

export HOME="$SCRATCH_BASE"
export HF_HOME="$SCRATCH_BASE/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export XDG_CACHE_HOME="$HF_HOME"

mkdir -p "$HF_HOME"

export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# --------------------------------------------------
# Move into Slide2Vec repository
# --------------------------------------------------

cd "$REPO_DIR" || exit 1

echo "Installing required backends..."
python3 -m pip install --quiet openslide-bin
pip3 install --quiet \
    git+https://github.com/lilab-stanford/MUSK.git \
    git+https://github.com/Mahmoodlab/CONCH.git


(
    while true; do
        sleep 1800
        echo "[SYNC] Copying outputs from scratch → final directory..."
        mkdir -p "$FINAL_OUTPUT_DIR"
        cp -ur "$SCRATCH_OUTPUT_DIR"/* "$FINAL_OUTPUT_DIR"/ 2>/dev/null
        echo "[SYNC] Incremental backup completed at $(date)"
    done
) &
SYNC_PID=$!

# --------------------------------------------------
# Run Slide2Vec
# --------------------------------------------------

echo "Running Slide2Vec with config:"
echo "  $CONFIG_PATH"

env HF_HOME="$HF_HOME" TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
    HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE" XDG_CACHE_HOME="$XDG_CACHE_HOME" \
    python3 -m slide2vec.main --config "$CONFIG_PATH"

# --------------------------------------------------
# Final sync + cleanup
# --------------------------------------------------

echo "Stopping background sync..."
kill $SYNC_PID 2>/dev/null

echo "Performing final sync..."
mkdir -p "$FINAL_OUTPUT_DIR"
cp -ur "$SCRATCH_OUTPUT_DIR"/* "$FINAL_OUTPUT_DIR"/ 2>/dev/null

echo "Slide2Vec H-OPTIMUS-1 missing slides part2 job completed successfully."
