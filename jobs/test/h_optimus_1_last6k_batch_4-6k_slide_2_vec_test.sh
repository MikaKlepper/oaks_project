#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=7-00:00:00
#SBATCH --job-name="l-h_optimus_1_last6k_batch_4-6k"
#SBATCH --output=/data/pathology/projects/mika/repos/oaks_project/logs/slurm-%j-l-h_optimus_1_last6k_batch_4-6k.out
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/pathology/projects:/data/temporary
#SBATCH --container-image="dockerdex.umcn.nl:5005#clemsgrs/slide2vec:v1.3.0"
#SBATCH --qos=low
#SBATCH --requeue

echo "RUNNING SLIDE2VEC WITH H-OPTIMUS-1 ON LAST 6K (BATCH 4-6K) SET"

REPO_DIR="/data/temporary/mika/repos/oaks_project/slide_2_vec"
CONFIG_PATH="$REPO_DIR/yaml_configs/liver_h_optimus_1_last6k_batch_4-6k.yaml"

SCRATCH_BASE="/scratch_mikaklepper_test"
SCRATCH_OUTPUT_DIR="$SCRATCH_BASE/outputs/H_OPTIMUS_1_last6k/batch_4-6k"
FINAL_OUTPUT_DIR="/data/temporary/toxicology/TG-GATES/liver/Tests_FM/H_OPTIMUS_1_last6k/batch_4-6k"

mkdir -p "$SCRATCH_BASE" "$SCRATCH_OUTPUT_DIR" "$FINAL_OUTPUT_DIR"

export HF_TOKEN="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"

export HOME="$SCRATCH_BASE"
mkdir -p "$HOME/hf_cache"
export HF_HOME="$HOME/hf_cache"
export TRANSFORMERS_CACHE="$HOME/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HOME/hf_cache"
export XDG_CACHE_HOME="$HOME/hf_cache"

export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

cd "$REPO_DIR"

python3 -m pip install openslide-bin
pip3 install git+https://github.com/lilab-stanford/MUSK.git git+https://github.com/Mahmoodlab/CONCH.git

(
    while true; do
        sleep 1800
        cp -ur "$SCRATCH_OUTPUT_DIR"/* "$FINAL_OUTPUT_DIR"/ 2>/dev/null
    done
) &
SYNC_PID=$!

python3 -m slide2vec.main --config "$CONFIG_PATH"

kill $SYNC_PID 2>/dev/null

cp -ur "$SCRATCH_OUTPUT_DIR"/* "$FINAL_OUTPUT_DIR"/ 2>/dev/null

echo "Slide2Vec H-OPTIMUS-1 last6k job completed successfully."
