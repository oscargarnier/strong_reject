#!/bin/bash
#SBATCH --account=c2016159
#SBATCH --constraint=MI250
#SBATCH --job-name=vanilla_evaluation_3
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --threads-per-core=1
#SBATCH --time=02:00:00
#SBATCH --export=ALL
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out


# Load your environment (adjust to your cluster)
source $WORKDIR/envs/strong_env/bin/activate

export HF_HOME=$WORKDIR/hf_cache HF_HUB_OFFLINE=1 HF_HUB_VERBOSITY=debug TRANSFORMERS_OFFLINE=1 LITELLM_LOCAL_MODEL_COST_MAP="True"
#Check for GPU visibility
export AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1

python run_attack_and_evaluation.py \
    --attack none \
    --dataset_file data/strong_reject3.csv \
    --victim_model llama2-7 \
    --evaluator strongreject_finetuned \
    --max_tokens 500 \
    --full_reset
