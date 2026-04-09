#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=30G
#SBATCH --time=7-00:00:00
#SBATCH --job-name="toxicology_archives"
#SBATCH --output=/data/pathology/projects/mika/repos/oaks_project/logs/slurm-%j-archives.out
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/pathology/projects:/data/temporary
#SBATCH --container-image="dockerdex.umcn.nl:5005#clemsgrs/slide2vec:v1.3.0"
#SBATCH --qos=high
#SBATCH --requeue

set -euo pipefail

echo "Starting toxicology feature archiving job"

# --------------------------------------------------
# Config
# --------------------------------------------------

DATASETS=("TG-GATES" "UCB")
SPLITS=("Trainings_FM" "Validations_FM" "Tests_FM")

BASE_DATA_ROOT="/data/temporary/toxicology"
BASE_ARCHIVE_ROOT="/data/temporary/toxicology_archives"

PARALLEL_JOBS=8

# --------------------------------------------------
# Compression function
# --------------------------------------------------

compress_encoder() {

    DATASET=$1
    SPLIT=$2
    ENC_DIR=$3

    ENC_NAME=$(basename "$ENC_DIR")

    SRC_DIR="$BASE_DATA_ROOT/$DATASET/$SPLIT"
    FINAL_SPLIT="$BASE_ARCHIVE_ROOT/$DATASET/$SPLIT"

    mkdir -p "$FINAL_SPLIT"

    ARCHIVE="$FINAL_SPLIT/${ENC_NAME}.tar"

    if [ -f "$ARCHIVE" ]; then
        echo "Skipping $DATASET/$SPLIT/$ENC_NAME (already archived)"
        return
    fi

    echo "Creating archive $DATASET/$SPLIT/$ENC_NAME"

    tar -cf "$ARCHIVE" -C "$SRC_DIR" "$ENC_NAME"

    echo "Finished $DATASET/$SPLIT/$ENC_NAME"
}

export -f compress_encoder
export BASE_DATA_ROOT BASE_ARCHIVE_ROOT

# --------------------------------------------------
# Run compression
# --------------------------------------------------

for DATASET in "${DATASETS[@]}"; do

    echo ""
    echo "============================"
    echo "Processing dataset: $DATASET"
    echo "============================"

    for SPLIT in "${SPLITS[@]}"; do

        SRC_DIR="$BASE_DATA_ROOT/$DATASET/$SPLIT"

        if [ ! -d "$SRC_DIR" ]; then
            echo "Skipping missing split: $SRC_DIR"
            continue
        fi

        echo ""
        echo "Processing split: $DATASET/$SPLIT"

        find "$SRC_DIR" -mindepth 1 -maxdepth 1 -type d | \
        xargs -P $PARALLEL_JOBS -I {} bash -c "compress_encoder $DATASET $SPLIT {}"

    done

done

echo ""
echo "All archives completed successfully."