#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=60G
#SBATCH --time=7-00:00:00
#SBATCH --job-name="s2vnew"
#SBATCH --output=/data/temporary/mika/repos/oaks_project/logs/slurm-%j-l-titan.out
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/temporary:/data/temporary
#SBATCH --container-image="waticlems/slide2vec:prism"
#SBATCH --qos=vram
#SBATCH --requeue

export HF_HOME="/tmp"
export PYTHONPATH="${PYTHONPATH}:/data/temporary/ivan/cloned_tools/slide2vec/"

REPO_DIR="/data/temporary/mika/repos/oaks_project/slide_2_vec"
CONFIG_PATH="$REPO_DIR/yaml_configs/liver_titan.yaml"

export HF_TOKEN="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"

pip3 install h5py

# mkdir -p /tmp/modules/transformers_modules/MahmoodLab/TITAN/d3eb67f26b9256b617f84dbb9b2978d70a538ff7
# cp -r /data/temporary/ivan/tmp/modules/transformers_modules/MahmoodLab/TITAN/  /tmp/modules/transformers_modules/MahmoodLab/

. /data/temporary/ivan/cpgscriptbackup/keys.sh
cd /data/temporary/ivan/cloned_tools/slide2vec/
python3 slide2vec/main.py --config-file "$CONFIG_PATH"
 