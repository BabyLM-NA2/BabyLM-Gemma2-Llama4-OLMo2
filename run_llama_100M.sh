#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --output=./log/train_model_%j.log
#SBATCH --mail-user=wratthapoom1@sheffield.ac.uk
#SBATCH --mail-type=ALL

# Load required modules
module load Anaconda3/2024.02-1
module load CUDA/12.4.0
module load GCC/12.2.0

# Check if the conda environment 'babylm' exists
if conda env list | grep -q 'babylm2'; then
    echo "Conda environment 'babylm2' already exists."
else
    echo "Conda environment 'babylm2' does not exist. Creating it from environment.yml..."
    conda env create -f environment.yml -n babylm2
fi

# Activate conda environment
source activate babylm2
conda env export > environment.yml

# Set critical environment variables
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_VISIBLE_DEVICES=0  # Force single GPU usage

python run.py \
  --data_folder=train_100M \
  --model=llama \
  --vocab_size=48000 \
  --seq_length=1024 \
  --batch_size=64 \
  --hidden_size=1024 \
  --num_hidden_layers=10

echo "Job completed"
