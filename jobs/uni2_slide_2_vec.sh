#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=7-00:00:00
#SBATCH --job-name="l-uni2"
#SBATCH --output=/data/temporary/mika/repos/oaks_project/logs/slurm-%j-l-uni2.out
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/temporary:/data/temporary
#SBATCH --container-image="dockerdex.umcn.nl:5005#clemsgrs/slide2vec:v1.3.0"
#SBATCH --qos=low
#SBATCH --requeue


export HF_TOKEN="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"
export HF_HOME="/tmp"
export PYTHONPATH="/data/temporary/mika/repos/oaks_project/slide_2_vec:$PYTHONPATH"

cd /data/temporary/mika/repos/oaks_project/slide_2_vec

python3 -m pip install openslide-bin

CONFIG_PATH="/data/temporary/mika/repos/oaks_project/slide_2_vec/yaml_configs/liver_uni2.yaml"
echo " Running Slide2Vec with config: $CONFIG_PATH"
python3 -m slide2vec.main --config "$CONFIG_PATH"


echo " Slide2Vec job completed successfully."
