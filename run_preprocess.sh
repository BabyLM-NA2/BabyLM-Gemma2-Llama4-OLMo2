#!/bin/bash
#SBATCH --job-name=preprocessing
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --output=./log/preprocessing_%j.log
#SBATCH --mail-user=wratthapoom1@sheffield.ac.uk
#SBATCH --mail-type=ALL

# Select Data Folder to clean
export DATA_FOLDER=train_100M

# Load required modules
module load Anaconda3/2024.02-1

# Check if the conda environment 'babylm' exists
if conda env list | grep -q 'babylm'; then
    echo "Conda environment 'babylm' already exists."
else
    echo "Conda environment 'babylm' does not exist. Creating it from environment.yml..."
    conda env create -f environment.yml -n babylm
fi

# Activate conda environment
source activate babylm

# Specify the folder containing Python scripts
SCRIPT_FOLDER="./preprocessing"

# Check if the folder exists
if [ -d "$SCRIPT_FOLDER" ]; then
    echo "Running all Python scripts in the folder: $SCRIPT_FOLDER"
    
    # Loop through each .py file in the folder
    for script in "$SCRIPT_FOLDER"/*.py; do
        # Check if the file exists (to avoid errors if no .py files are found)
        if [ -f "$script" ]; then
            echo "Running $script..."
            python "$script"
        else
            echo "No Python scripts found in $SCRIPT_FOLDER."
        fi
    done
else
    echo "Folder $SCRIPT_FOLDER does not exist. Please check the path."
fi

echo "Job completed"
