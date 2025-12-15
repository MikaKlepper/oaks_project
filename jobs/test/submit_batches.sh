#!/bin/bash

BATCH_DIR="/data/temporary/mika/repos/oaks_project/splitting_data/Splits/last6k_batches"

for batch_csv in "$BATCH_DIR"/last6k_batch_*.csv; do
    BID=$(basename "$batch_csv" .csv | sed 's/last6k_batch_//')
    echo "Submitting batch $BID → $batch_csv"
    sbatch run_batch.sh "$batch_csv" "$BID"
done
