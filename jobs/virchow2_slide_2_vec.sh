#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=7-00:00:00
#SBATCH --job-name="l-virchow2"
#SBATCH --output=/data/temporary/mika/repos/oaks_project/logs/slurm-%j-l-virchow2.out
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/temporary:/data/temporary
#SBATCH --container-image="dockerdex.umcn.nl:5005#clemsgrs/slide2vec:v1.3.0"
#SBATCH --qos=high
#SBATCH --requeue

echo "RUNNING SLIDE2VEC WITH VIRCHOW2 ON VALIDATION SET"

# setup environment
export HF_TOKEN="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"
export HF_HOME="/tmp/hf_cache"
export PYTHONPATH="/data/temporary/mika/repos/oaks_project/slide_2_vec:$PYTHONPATH"

# paths
REPO_DIR="/data/temporary/mika/repos/oaks_project/slide_2_vec"
CONFIG_PATH="$REPO_DIR/yaml_configs/liver_virchow2.yaml"
SRC_WSI_DIR="/data/temporary/mika/repos/oaks_project/wsis/liver/val"
SCRATCH_WSI_DIR="/scratch_mikaklepper_val/wsis/liver/val"
SCRATCH_OUTPUT_DIR="/scratch_mikaklepper_val/outputs/liver_virchow2"
FINAL_OUTPUT_DIR="/data/temporary/toxicology/TG-GATES/liver/Validations_FM/VIRCHOW2"

# create all directories if they don't exist
echo "Ensure all directories exist ... "
mkdir -p "$SRC_WSI_DIR" "$SCRATCH_WSI_DIR" "$SCRATCH_OUTPUT_DIR" "$FINAL_OUTPUT_DIR" "$REPO_DIR"

# copy all WSIs to to scratch if needed
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
python3 -m slide2vec.main --config "$CONFIG_PATH"

# Stop background sync and perform final copy 
echo "Stopping background sync..."
kill $SYNC_PID 2>/dev/null

echo " Performing final sync to $FINAL_OUTPUT_DIR ..."
mkdir -p "$FINAL_OUTPUT_DIR"
cp -ur "$SCRATCH_OUTPUT_DIR"/* "$FINAL_OUTPUT_DIR"/ 2>/dev/null

echo " Slide2Vec job completed successfully."
