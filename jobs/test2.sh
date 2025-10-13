#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=30G
#SBATCH --time=7-00:00:00
#SBATCH --job-name="l-uni-local"
#SBATCH --output=/data/temporary/mika/repos/oaks_project/logs/slurm-%j-l-uni-local.out
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/temporary:/data/temporary
#SBATCH --container-image="dockerdex.umcn.nl:5005#clemsgrs/slide2vec:v1.2.3"
#SBATCH --qos=high
#SBATCH --requeue

# === Environment ===
export HF_TOKEN="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"
export HF_HOME="/tmp"

# ✅ Use your own repository version instead of /opt/app
export PYTHONPATH="/data/temporary/mika/repos/oaks_project/slide_2_vec:$PYTHONPATH"
cd /data/temporary/mika/repos/oaks_project/slide_2_vec

echo "🔧 Using Mika's local Slide2Vec repository"
cp -r /opt/app/slide2vec/data /data/temporary/mika/repos/oaks_project/slide_2_vec/slide2vec/

# === Debug Python paths ===
echo "=== DEBUG: Which Slide2Vec will Python use? ==="
python3 - <<'EOF'
import sys, slide2vec, os
print("Python executable:", sys.executable)
print("Search path order:")
for p in sys.path: print(" -", p)
print("\nslide2vec imported from:", slide2vec.__file__)
cfg_path = os.path.join(os.path.dirname(slide2vec.__file__), "configs", "default.yaml")
print("Default config exists:", os.path.exists(cfg_path), "->", cfg_path)
EOF

# === Run Slide2Vec from your repo ===
echo "Running Slide2Vec from your local repo with config: liver_uni.yaml"
python3 -m slide2vec.main --config /data/temporary/mika/repos/oaks_project/slide_2_vec/liver_uni.yaml

echo "✅ Slide2Vec job completed (local repo version)."
