#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=7-00:00:00
#SBATCH --job-name="l-titan"
#SBATCH --output=/data/temporary/mika/repos/oaks_project/logs/slurm-%j-l-titan.out
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/temporary:/data/temporary
#SBATCH --container-image="dockerdex.umcn.nl:5005#clemsgrs/slide2vec:v1.3.0"
#SBATCH --qos=high
#SBATCH --requeue


echo "RUNNING SLIDE2VEC WITH TITAN ON TRAINING SET"

# paths
REPO_DIR="/data/temporary/mika/repos/oaks_project/slide_2_vec"
CONFIG_PATH="$REPO_DIR/yaml_configs/liver_titan.yaml"
SRC_WSI_DIR="/data/temporary/mika/repos/oaks_project/wsis/liver/train"
SCRATCH_BASE="/scratch_mikaklepper_train"
SCRATCH_WSI_DIR="$SCRATCH_BASE/wsis/liver/train"
SCRATCH_OUTPUT_DIR="$SCRATCH_BASE/outputs/TITAN"
FINAL_OUTPUT_DIR="/data/temporary/toxicology/TG-GATES/liver/Trainings_FM/TITAN"

# ensure base scratch directory exists
echo "Ensure base scratch directory exists ..."
mkdir -p "$SCRATCH_BASE"

# create all directories if they don't exist
echo "Ensure all directories exist ..."
mkdir -p "$SRC_WSI_DIR" "$SCRATCH_WSI_DIR" "$SCRATCH_OUTPUT_DIR" "$FINAL_OUTPUT_DIR" "$REPO_DIR"

# setup environment
export HF_TOKEN="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"

# redirect Hugging Face cache to scratch (prevents race condition + quota errors)
export HOME="$SCRATCH_BASE"
mkdir -p "$HOME/hf_cache"
export HF_HOME="$HOME/hf_cache"
export TRANSFORMERS_CACHE="$HOME/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HOME/hf_cache"
export XDG_CACHE_HOME="$HOME/hf_cache"

export PYTHONPATH="/data/temporary/mika/repos/oaks_project/slide_2_vec:$PYTHONPATH"

# copy all WSIs to scratch if needed
echo "Check whether WSIs are transferred to scratch space ..."
if [ ! -d "$SCRATCH_WSI_DIR" ] || [ $(ls -1 "$SCRATCH_WSI_DIR" 2>/dev/null | wc -l) -eq 0 ]; then
    echo " Scratch WSIs missing — copying from /data/temporary..."
    cp -u "$SRC_WSI_DIR"/* "$SCRATCH_WSI_DIR"/
else
    echo "Found $(ls -1 "$SCRATCH_WSI_DIR" | wc -l) WSIs in scratch."
fi

# move to directory needed to run slide2vec
cd "$REPO_DIR"

echo "Install openslide as backend"
python3 -m pip install openslide-bin

# optional: pre-download TITAN model to shared cache (prevents multi-GPU race)
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("MahmoodLab/TITAN", cache_dir="/scratch_mikaklepper_val/hf_cache")
PY

# start backend shell, to regulary transfer outputs to temporary
# is done every 30 minutes
(
    while true; do
    sleep 1800
    echo "copying outputs to temporary from scratch"
    mkdir -p "$FINAL_OUTPUT_DIR"
    cp -ur "$SCRATCH_OUTPUT_DIR"/* "$FINAL_OUTPUT_DIR"/ 2>/dev/null
    echo "Incremental backup complete at $(date)" 
    done
) &
SYNC_PID=$!

echo " Running Slide2Vec with config: $CONFIG_PATH"
env HF_HOME="$HF_HOME" TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
    HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE" XDG_CACHE_HOME="$XDG_CACHE_HOME" \
    python3 -m slide2vec.main --config "$CONFIG_PATH"

# Stop background sync and perform final copy 
echo "Stopping background sync..."
kill $SYNC_PID 2>/dev/null

echo " Performing final sync to $FINAL_OUTPUT_DIR ..."
mkdir -p "$FINAL_OUTPUT_DIR"
cp -ur "$SCRATCH_OUTPUT_DIR"/* "$FINAL_OUTPUT_DIR"/ 2>/dev/null

echo " Slide2Vec TITAN job completed successfully."
