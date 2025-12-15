#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=7-00:00:00
#SBATCH --job-name="hopt_last6k_${2}"
#SBATCH --output=/data/temporary/mika/repos/oaks_project/logs/slurm-%j-hopt_last6k_${2}.out
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/temporary:/data/temporary
#SBATCH --container-image="dockerdex.umcn.nl:5005#clemsgrs/slide2vec:v1.3.0"
#SBATCH --qos=low
#SBATCH --requeue

CSV_FILE="$1"
BATCH_ID="$2"

REPO_DIR="/data/temporary/mika/repos/oaks_project/slide_2_vec"
TEMPLATE="$REPO_DIR/yaml_configs/liver_h_optimus_1_template.yaml"

SCRATCH_BASE="/scratch_mikaklepper_test"
SCRATCH_OUTPUT_DIR="$SCRATCH_BASE/outputs/H_OPTIMUS_1_last6k/batch_${BATCH_ID}"
FINAL_OUTPUT_DIR="/data/temporary/toxicology/TG-GATES/liver/Tests_FM/H_OPTIMUS_1_last6k/batch_${BATCH_ID}"

CONFIG="/scratch_mikaklepper_test/config_last6k_batch_${BATCH_ID}.yaml"

mkdir -p "$SCRATCH_OUTPUT_DIR" "$FINAL_OUTPUT_DIR" "$SCRATCH_BASE"

export CSV_FILE SCRATCH_OUTPUT_DIR FINAL_OUTPUT_DIR

# build configuration
envsubst < "$TEMPLATE" > "$CONFIG"

cd "$REPO_DIR"

# run inference
python3 -m slide2vec.main --config "$CONFIG"

# sync back
cp -ur "$SCRATCH_OUTPUT_DIR"/* "$FINAL_OUTPUT_DIR"/ 2>/dev/null

echo "[DONE] Batch ${BATCH_ID}"
