#!/bin/bash
#SBATCH --job-name=train_llama
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --mem=64G
#SBATCH --output=./log/train_llama_%j.log
#SBATCH --mail-user=wratthapoom1@sheffield.ac.uk
#SBATCH --mail-type=ALL

# Load required modules
module load Anaconda3/2024.02-1
module load CUDA/12.4.0
module load GCC/12.2.0

# Check if the conda environment 'babylm' exists
if conda env list | grep -q 'babylm1'; then
    echo "Conda environment 'babylm1' already exists."
else
    echo "Conda environment 'babylm1' does not exist. Creating it from environment.yml..."
    conda env create -f environment.yml -n babylm1
fi

# Activate conda environment
source activate babylm1

# Skip the environment export which was causing permission denied errors
# conda env export > environment.yml

# Set environment variables for better performance
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1  # For multi-GPU setups
export TORCH_EXTENSIONS_DIR=$HOME/.cache/torch_extensions

# Fix for expandable segments (avoid memory fragmentation)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"
export BNB_CUDA_VERSION=124
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.4/lib64

# Monitor GPU status before running
echo "GPU status before execution:"
nvidia-smi || echo "nvidia-smi not found, continuing anyway"

# Run script with distributed launcher
echo "Running LLaMA training with torchrun..."
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
torchrun --standalone --nproc_per_node=$NUM_GPUS --log_dir=my_logs/torch_distributed_logs run.py --data_folder=train_10M --model=llama --vocab_size=200000 --seq_length=128 --batch_size=32
# Check execution status
if [ $? -eq 0 ]; then
    echo "Execution completed successfully!"
else
    echo "LLaMA training failed with error code $?"
fi

# Monitor GPU status after running
echo "GPU status after execution:"
nvidia-smi || echo "nvidia-smi not found, continuing anyway"

echo "Job completed"