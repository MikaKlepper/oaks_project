#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --job-name=trident-uni
#SBATCH --output=/data/temporary/mika/repos/oaks_project/logs/slurm-%j-trident-uni.out
#SBATCH --container-image="dockerdex.umcn.nl:5005#dnschouten/crossmodal-alignment:v1.0_trident"
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/temporary:/data/temporary
#SBATCH --qos=low

# -----------------------------
# Environment setup
# -----------------------------
export HF_HOME=/tmp
export HF_TOKEN="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"

# Move to Trident's working directory inside the container
cd /home/user/trident

# -----------------------------
# Run feature extraction
# -----------------------------
echo "🚀 Starting Trident UNI feature extraction..."

python3 run_batch_of_slides.py \
    --task feat \
    --wsi_dir /data/pa_cpgarchive/archives/toxicology/open-tg-gates/images \
    --custom_list_of_wsis /data/temporary/mika/repos/oaks_project/splitting_data/Subsets/val_balanced_subset_paths.csv \
    --job_dir /data/temporary/mika/repos/oaks_project/trident_output/uni_features \
    --patch_encoder uni_v1 \
    --mag 20 \
    --patch_size 256 \
    --max_workers 16 \

echo "✅ Trident UNI feature extraction completed successfully."

