#!/bin/bash
#SBATCH --qos=low
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=7-00:00:00
#SBATCH --job-name="l-TEST"
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/pathology/projects:/data/temporary
#SBATCH --output=/data/pathology/projects/mika/repos/oaks_project/logs/slurm-%j-l-TEST.out
#SBATCH --container-image="dockerdex.umcn.nl:5005#clemsgrs/slide2vec:v1.3.0"
#SBATCH --requeue



export HF_TOKEN="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"
export HF_HOME="/tmp"
export PYTHONPATH="/data/temporary/mika/repos/oaks_project/slide_2_vec:$PYTHONPATH"

cd /data/temporary/mika/repos/oaks_project/slide_2_vec

PORT=8789
NODE=$(hostname)
USERNAME=$(whoami)
SSH_ID_RSA_FOLDER=/Users/mikaklepper/.ssh/id_ed25519

echo "vscode://vscode-remote/ssh-remote+${USERNAME}@${NODE}:${PORT}/home/${USERNAME}?ssh=${SSH_ID_RSA_FOLDER}"

echo "Started SSH on port $PORT"
/usr/sbin/sshd -D -p $PORT
