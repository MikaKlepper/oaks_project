#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=7-00:00:00
#SBATCH --job-name="benchmark-all-models"
#SBATCH --output=/data/pathology/projects/mika/repos/oaks_project/logs/slurm-%j-benchmark.out
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/pathology/projects:/data/temporary
#SBATCH --container-image="dockerdex.umcn.nl:5005#clemsgrs/slide2vec:v1.3.0"
#SBATCH --qos=low
#SBATCH --requeue

set -euo pipefail

has_pt_features() {
    local feature_dir="$1"
    find "$feature_dir" -maxdepth 1 -name '*.pt' -print -quit | grep -q .
}

echo "============================"
echo "   STARTING BENCHMARK RUN   "
echo "============================"

ENCODER="H_OPTIMUS_1"

REPO_DIR="/data/temporary/mika/repos/oaks_project/pipeline"
ARCHIVE_ROOT="/data/temporary/toxicology_archives"
NODE_FEATURE_ROOT="/local/features"
LOCAL_ARCHIVE_ROOT="/local/tmp_archives"
SMOKE_MODE="${BENCHMARK_SMOKE:-0}"

mkdir -p "$NODE_FEATURE_ROOT"
mkdir -p "$LOCAL_ARCHIVE_ROOT"

echo "[INFO] Copying and extracting $ENCODER feature archives..."

for DATASET in TG-GATES UCB; do
    for SPLIT in Trainings_FM Validations_FM Tests_FM; do

        SRC="$ARCHIVE_ROOT/$DATASET/$SPLIT"

        if [ ! -d "$SRC" ]; then
            continue
        fi

        DST="$NODE_FEATURE_ROOT/$DATASET/$SPLIT"
        mkdir -p "$DST"

        REMOTE_ARCHIVE="$SRC/${ENCODER}.tar"

        if [ ! -f "$REMOTE_ARCHIVE" ]; then
            echo "[INFO] No archive for $DATASET/$SPLIT/$ENCODER"
            continue
        fi

        TARGET_DIR="$DST/$ENCODER/features"

        if [ -d "$TARGET_DIR" ]; then
            echo "[INFO] $DATASET/$SPLIT/$ENCODER already extracted"
            if ! has_pt_features "$TARGET_DIR"; then
                echo "[ERROR] $TARGET_DIR exists but contains no .pt feature files"
                exit 1
            fi
            continue
        fi

        LOCAL_DATASET_ARCHIVE_DIR="$LOCAL_ARCHIVE_ROOT/$DATASET/$SPLIT"
        mkdir -p "$LOCAL_DATASET_ARCHIVE_DIR"

        LOCAL_ARCHIVE="$LOCAL_DATASET_ARCHIVE_DIR/${ENCODER}.tar"

        SIZE=$(du -h "$REMOTE_ARCHIVE" | cut -f1)

        echo ""
        echo "[INFO] Copying $DATASET/$SPLIT/$ENCODER ($SIZE) to local disk..."

        COPY_START=$(date +%s)
        cp "$REMOTE_ARCHIVE" "$LOCAL_ARCHIVE"
        COPY_END=$(date +%s)
        COPY_ELAPSED=$((COPY_END - COPY_START))

        echo "[INFO] Finished copy in ${COPY_ELAPSED}s"

        echo "[INFO] Extracting $DATASET/$SPLIT/$ENCODER/features locally..."

        EXTRACT_START=$(date +%s)
        tar -xf "$LOCAL_ARCHIVE" -C "$DST" "$ENCODER/features"
        EXTRACT_END=$(date +%s)
        EXTRACT_ELAPSED=$((EXTRACT_END - EXTRACT_START))

        echo "[INFO] Finished extraction in ${EXTRACT_ELAPSED}s"

        echo "[INFO] Removing local tar..."
        rm -f "$LOCAL_ARCHIVE"

        if ! has_pt_features "$TARGET_DIR"; then
            echo "[ERROR] Extraction completed but no .pt feature files were found in $TARGET_DIR"
            exit 1
        fi

    done
done

echo "[INFO] Feature copy + extraction finished"

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
