#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=72:00:00
#SBATCH --job-name=trident-uni
#SBATCH --output=/data/temporary/mika/repos/oaks_project/logs/slurm-%j-trident-uni.out
#SBATCH --container-image="dockerdex.umcn.nl:5005#dnschouten/crossmodal-alignment:v1.0_trident"
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/temporary:/data/temporary
#SBATCH --qos=high

export HF_TOKEN="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"
export HF_HOME="/tmp"

PORT=8693
NODE=$(hostname)
USERNAME=$(whoami)
SSH_ID_RSA_FOLDER=/Users/mikaklepper/.ssh/id_ed25519

echo "vscode://vscode-remote/ssh-remote+${USERNAME}@${NODE}:${PORT}/home/${USERNAME}"

echo "Started SSH on port $PORT"
/usr/sbin/sshd -D -p $PORT
