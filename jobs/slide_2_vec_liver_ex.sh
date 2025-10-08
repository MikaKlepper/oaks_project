#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=30G
#SBATCH --time=7-00:00:00
#SBATCH --job-name="l-uni"
#SBATCH --output=/data/temporary/mika/repos/oaks_project/logs/slurm-%j-l-uni.out
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/temporary:/data/temporary
#SBATCH --container-image="dockerdex.umcn.nl:5005#clemsgrs/slide2vec:v1.2.3"
#SBATCH --qos=high
#SBATCH --requeue



# === Environment setup ===
export HF_TOKEN="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"
export HF_HOME="/tmp"
export PYTHONPATH="/data/temporary/mika/repos/slide_2_vec:${PYTHONPATH}"

cd /data/temporary/mika/repos/slide_2_vec

# === Ensure openslide is available (if not in container) ===
echo "Installing openslide..."
pip install --no-cache-dir openslide-python openslide-bin


# === Install CUDA 11 compatible PyTorch stack ===
echo "Installing CUDA 11 compatible PyTorch stack for GTX 1080 Ti..."
pip install --no-cache-dir torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# === Remove incompatible xFormers (built for PyTorch 2.8) ===
pip uninstall -y xformers >/dev/null 2>&1 || true


sed -i 's/torchvision.disable_beta_transforms_warning()/if hasattr(torchvision, "disable_beta_transforms_warning"): torchvision.disable_beta_transforms_warning()/' /data/temporary/mika/repos/slide_2_vec/slide2vec/embed.py


# === Hugging Face login (non-interactive) ===
echo "Logging in to Hugging Face..."
huggingface-cli login --token "$HF_TOKEN"

# === Run Slide2Vec job ===
CONFIG_PATH="/home/mikaklepper/temporary/repos/slide_2_vec/liver_uni.yaml"
echo "Running Slide2Vec with config: ${CONFIG_PATH}"
python3 -m slide2vec.main --config-file "${CONFIG_PATH}"

echo " Slide2Vec job completed."

