#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=7-00:00:00
#SBATCH --job-name="l-h_optimus_1"
#SBATCH --output=/data/temporary/mika/repos/oaks_project/logs/slurm-%j-l-h_optimus_1.out
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/temporary:/data/temporary
#SBATCH --container-image="dockerdex.umcn.nl:5005#clemsgrs/slide2vec:v1.3.0"
#SBATCH --qos=vram
#SBATCH --requeue

echo "RUNNING SLIDE2VEC WITH H-OPTIMUS-1 ON TEST SET"

# --------------------------------------------------
# Paths
# --------------------------------------------------
REPO_DIR="/data/temporary/mika/repos/oaks_project/slide_2_vec"
CONFIG_PATH="$REPO_DIR/yaml_configs/liver_h_optimus_1.yaml"

# WSIs are NOT copied — they are read from test_wsi.csv
# located under: /data/temporary/mika/repos/oaks_project/wsis/liver/test/test_wsi.csv

SCRATCH_BASE="/scratch_mikaklepper_test"
SCRATCH_OUTPUT_DIR="$SCRATCH_BASE/outputs/H_OPTIMUS_1"
FINAL_OUTPUT_DIR="/data/temporary/toxicology/TG-GATES/liver/Tests_FM/H_OPTIMUS_1"

echo "Creating necessary directories..."
mkdir -p "$SCRATCH_BASE" "$SCRATCH_OUTPUT_DIR" "$FINAL_OUTPUT_DIR"

# --------------------------------------------------
# HuggingFace cache on scratch (much faster)
# --------------------------------------------------
export HF_TOKEN="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"

export HOME="$SCRATCH_BASE"
mkdir -p "$HOME/hf_cache"
export HF_HOME="$HOME/hf_cache"
export TRANSFORMERS_CACHE="$HOME/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HOME/hf_cache"
export XDG_CACHE_HOME="$HOME/hf_cache"

export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# --------------------------------------------------
# Move into Slide2Vec repository
# --------------------------------------------------
cd "$REPO_DIR"

echo "Installing necessary packages inside container..."
python3 -m pip install openslide-bin
pip3 install git+https://github.com/lilab-stanford/MUSK.git git+https://github.com/Mahmoodlab/CONCH.git
# --------------------------------------------------
# Background syncing (every 30 minutes)
# --------------------------------------------------
(
    while true; do
        sleep 1800
        echo "[SYNC] Copying outputs from scratch → temporary..."
        mkdir -p "$FINAL_OUTPUT_DIR"
        cp -ur "$SCRATCH_OUTPUT_DIR"/* "$FINAL_OUTPUT_DIR"/ 2>/dev/null
        echo "[SYNC] Incremental backup completed at $(date)"
    done
) &
SYNC_PID=$!

# --------------------------------------------------
# Run Slide2Vec
# --------------------------------------------------
echo "Running Slide2Vec with config: $CONFIG_PATH"

env HF_HOME="$HF_HOME" TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
    HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE" XDG_CACHE_HOME="$XDG_CACHE_HOME" \
    python3 -m slide2vec.main --config "$CONFIG_PATH"

# --------------------------------------------------
# Final sync + cleanup
# --------------------------------------------------
echo "Stopping background sync..."
kill $SYNC_PID 2>/dev/null

echo "Final sync of outputs..."
mkdir -p "$FINAL_OUTPUT_DIR"
cp -ur "$SCRATCH_OUTPUT_DIR"/* "$FINAL_OUTPUT_DIR"/ 2>/dev/null

echo "Slide2Vec H-OPTIMUS-1 job completed successfully."
