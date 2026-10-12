#!/bin/bash
#SBATCH --account=c2016159
#SBATCH --constraint=MI250
#SBATCH --job-name=pair_attack_eval
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

# Load your environment (adjust to your cluster)
source $WORKDIR/envs/strong_env/bin/activate

python pair_attack_evaluation.py
