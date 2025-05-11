#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --mem=64G
#SBATCH --output=./log/train_model_%j.log
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
conda env export > environment.yml

# Set environment variables for better performance
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1  # For multi-GPU setups
export TORCH_EXTENSIONS_DIR=$HOME/.cache/torch_extensions

# Fix for expandable segments (avoid memory fragmentation)
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export BNB_CUDA_VERSION=124
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.4/lib64

# Configure NCCL for better multi-GPU performance
export NCCL_DEBUG=INFO
# Use multiple possible interface names instead of just eth0
export NCCL_SOCKET_IFNAME=en,eth,em,bond,ib
# Additional performance settings
export NCCL_IB_TC=128
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=22
export NCCL_TIMEOUT=1800
export NCCL_P2P_LEVEL=NVL 
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000

# Monitor GPU status before running
echo "GPU status before execution:"
nvidia-smi

# Run script without distributed launcher (using built-in model parallelism)
echo "Running run.py with native model parallelism..."
NUM_GPUS=$(nvidia-smi -L | wc -l)
# python run.py --data_folder=train_10M --model=rwkv --vocab_size=200000 --seq_length=128
torchrun --standalone --nproc_per_node=$NUM_GPUS run.py \
  --data_folder=train_10M \
  --model=rwkv \
  --vocab_size=200000 \
  --seq_length=128 \
  --batch_size=16
# deepspeed --num_gpus=$NUM_GPUS run.py --data_folder=train_10M --model=rwkv --vocab_size=200000 --seq_length=128

# Check execution status
if [ $? -eq 0 ]; then
    echo "Execution completed successfully!"
else
    echo "Execution failed with error code $?"
fi

# Monitor GPU status after running
echo "GPU status after execution:"
nvidia-smi

echo "Job completed"
