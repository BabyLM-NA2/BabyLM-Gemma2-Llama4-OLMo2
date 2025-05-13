#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-node=2
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

export TORCH_USE_CUDA_DSA=1  # Enable device-side assertions

# Monitor GPU status before running
echo "GPU status before execution:"
nvidia-smi

# Run script without distributed launcher (using built-in model parallelism)
echo "Running run.py with native model parallelism..."
NUM_GPUS=$(nvidia-smi -L | wc -l)

torchrun --standalone --nproc_per_node=$NUM_GPUS run.py \
  --data_folder=train_100M \
  --model=rwkv \
  --vocab_size=48000 \
  --seq_length=2048 \
  --batch_size=256 \
  --hidden_size=1536 \
  --num_hidden_layers=12

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
