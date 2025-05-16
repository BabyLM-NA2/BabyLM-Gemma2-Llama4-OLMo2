#!/bin/bash

#SBATCH --job-name=olmo2_10M
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --output=./log/train_olmo2_10M_%j.log
#SBATCH --mail-user=wratthapoom1@sheffield.ac.uk
#SBATCH --mail-type=ALL

# Load required modules
module load Anaconda3/2024.02-1
module load CUDA/12.4.0
module load GCC/12.2.0

# Activate conda environment
source activate babylm2

# Set critical environment variables
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_VISIBLE_DEVICES=0  # Force single GPU usage

# Run script
python run.py \
    --data_folder=train_10M \
    --model=olmo2 \
    --vocab_size=32000 \
    --seq_length=256 \
    --batch_size=32 \
    --hidden_size=256 \
    --epoch=10 \
    --num_hidden_layers=8

echo "Job completed"
