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
if conda env list | grep -q 'babylm'; then
    echo "Conda environment 'babylm' already exists."
else
    echo "Conda environment 'babylm' does not exist. Creating it from environment.yml..."
    conda env create -f environment.yml -n babylm
fi

# Activate conda environment
source activate babylm

# Set environment variables for better performance
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1  # For multi-GPU setups
export TORCH_EXTENSIONS_DIR=$HOME/.cache/torch_extensions
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Verify PyTorch installation and print version information
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Monitor GPU status before running
echo "GPU status before execution:"
nvidia-smi

# Run script without distributed launcher (using built-in model parallelism)
echo "Running main.py with native model parallelism..."
python -m run

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
