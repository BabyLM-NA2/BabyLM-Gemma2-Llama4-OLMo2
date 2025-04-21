#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-node=2
#SBATCH --mem=64G
#SBATCH --output=./output/train_model_%j.log
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
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"
export BNB_CUDA_VERSION=124
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.4/lib64

# Monitor GPU status before running
echo "GPU status before execution:"
nvidia-smi

# Run script without distributed launcher (using built-in model parallelism)
echo "Running run.py with native model parallelism..."
NUM_GPUS=$(nvidia-smi -L | wc -l)
# python run.py --data_folder=train_10M --model=rwkv --vocab_size=200000 --seq_length=128
torchrun --standalone --nproc_per_node=$NUM_GPUS --log_dir=./log/torch_distributed_logs run.py --data_folder=train_10M --model=rwkv --vocab_size=200000 --seq_length=128
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
