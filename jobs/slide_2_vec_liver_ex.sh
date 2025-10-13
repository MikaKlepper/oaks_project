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
export PYTHONPATH="/data/temporary/mika/repos/oaks_project/slide_2_vec:${PYTHONPATH}:$(pwd)"

# === Go to correct working directory ===
cd /data/temporary/mika/repos/oaks_project/slide_2_vec

# === Make sure Python can always find the local slide2vec package ===
SITE_DIR=$(python3 -c "import site; print(site.getsitepackages()[0])")
echo "/data/temporary/mika/repos/oaks_project/slide_2_vec" | tee "${SITE_DIR}/slide2vec_path.pth"
echo "✅ Added slide2vec path to ${SITE_DIR}/slide2vec_path.pth"

# === Verify path + import before running ===
echo "=== DEBUG PYTHONPATH ==="
python3 -c "import sys, os; print('Python executable:', sys.executable); print('Python path entries:'); [print(' -', p) for p in sys.path]; print('Current directory:', os.getcwd())"
python3 -c "import slide2vec; print('✅ slide2vec imported from:', slide2vec.__file__)"

# === Ensure openslide is available ===
echo "Installing openslide..."
pip install --no-cache-dir openslide-python openslide-bin

# === Install CUDA 11 compatible PyTorch stack ===
echo "Installing CUDA 11 compatible PyTorch stack for GTX 1080 Ti..."
pip install --no-cache-dir torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# === Remove incompatible xFormers (built for PyTorch 2.8) ===
pip uninstall -y xformers >/dev/null 2>&1 || true

# === Hugging Face login (non-interactive) ===
echo "Logging in to Hugging Face..."
huggingface-cli login --token "$HF_TOKEN"

# === Run Slide2Vec job ===
CONFIG_PATH="/data/temporary/mika/repos/oaks_project/slide_2_vec/liver_uni.yaml"
echo "Running Slide2Vec with config: ${CONFIG_PATH}"
python3 -m slide2vec.main --config-file "${CONFIG_PATH}"

echo "Slide2Vec job completed."
