#!/bin/bash
#SBATCH --account=c2016159
#SBATCH --constraint=MI250
#SBATCH --job-name=sample_model_evaluation_sample_dataset
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --threads-per-core=1
#SBATCH --time=sample_time
#SBATCH --export=ALL
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out


echo "Host:                 $(hostname)"
echo "Job ID:               $SLURM_JOB_ID"
echo "Slurm GPU allocation: ${SLURM_JOB_GPUS:-none}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"


# Load your environment (adjust to your cluster)
source $WORKDIR/envs/strong_env/bin/activate

export HF_HOME=$WORKDIR/hf_cache HF_HUB_OFFLINE=1 HF_HUB_VERBOSITY=debug TRANSFORMERS_OFFLINE=1 LITELLM_LOCAL_MODEL_COST_MAP="True"
#Check for GPU visibility
export AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1

python run_attack_and_evaluation.py \
    --attack none \
    --dataset_file data/sample_dataset.csv \
    --victim_model sample_model \
    --batch_size 128 \
    --evaluator strongreject_finetuned \
    --max_tokens 1000 \
    --full_reset
