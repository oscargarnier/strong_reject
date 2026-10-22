#!/bin/bash
#SBATCH --account=c2016159
#SBATCH --constraint=MI250
#SBATCH --job-name=sample_attack
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --threads-per-core=1
#SBATCH --time=03:00:00
#SBATCH --export=ALL
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out


# Load your environment (adjust to your cluster)
source $WORKDIR/envs/strong_env/bin/activate

export HF_HOME=$WORKDIR/hf_cache     # must match the download step
export HF_HUB_OFFLINE=1
export HF_HUB_VERBOSITY=debug
export TRANSFORMERS_OFFLINE=1
export LITELLM_LOCAL_MODEL_COST_MAP="True"
#Check for GPU visibility
export AMD_SERIALIZE_KERNEL=3 
export HIP_LAUNCH_BLOCKING=1

python run_attacks_and_evaluation.py \
    --attack sample_attack \
    --dataset_file sample_dataset \
    --victim_model sample_model \
    --evaluator harmbench_finetuned

