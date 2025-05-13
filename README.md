# Language Model Training Framework

A comprehensive framework for training small language models from scratch using the BabyLM dataset. This project supports multiple modern language model architectures (Gemma2, Llama4, Olmo2) and includes optimized training routines for efficient fine-tuning.

## Features

- **Multiple Model Architectures**: Train Gemma2, Llama4, or Olmo2 models
- **Custom Tokenizer Training**: BPE tokenizer with configurable vocabulary size
- **Memory-Optimized Training**: Includes CUDA memory management and gradient checkpointing
- **Robust Error Handling**: Safe data collation and automatic checkpoint saving on errors
- **Flexible Configuration**: Extensive command-line options for model and training parameters

## Installation

```bash
# Clone the repository
git clone https://github.com/BabyLM-NA2/BabyLM2025.git
cd BabyLM2025

# Install dependencies
conda env create -n babylm2 -f environment.yml
```

## Data

This project is designed to work with the BabyLM dataset which can be downloaded at [this OSF directory](https://osf.io/ad7qg/files/osfstorage#). The expected directory structure is:

```
data/
├── train_10M/
├── train_100M/
├── dev/
└── test/
```

## Project Structure

- **babylm_dataset.py**: Dataset and tokenizer handling
- **train.py**: Training utilities and optimization techniques
- **run.py**: Main execution script with CLI interface
- **run_preprocess.sh**: Bash script for data preprocessing

## Usage

### Basic Training

```bash
bash run_gemma_10M.sh
```

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_folder` | Data folder to process (train_10M, train_100M, dev, test) | train_10M |
| `--model` | Model architecture (gemma2, llama4, olmo2) | llama4 |
| `--vocab_size` | Size of vocabulary for tokenizer | 48000 |
| `--seq_length` | Token context length | 128 |
| `--batch_size` | Training batch size | 16 |
| `--epoch` | Number of training epochs | 3 |
| `--hidden_size` | Model hidden dimension size | 768 |
| `--num_hidden_layers` | Number of transformer layers | 12 |
| `--num_attention_heads` | Number of attention heads (for LLaMA) | None |

## Training Process

1. **Data Cleaning**: Preprocesses the BabyLM dataset using the `run_preprocess.sh` script
2. **Tokenizer Training**: Trains a custom BPE tokenizer on the cleaned data
3. **Dataset Creation**: Creates PyTorch datasets for training and evaluation
4. **Model Configuration**: Sets up the selected model architecture with specified parameters
5. **Training**: Trains the model with optimization techniques like gradient checkpointing and mixed precision

## Memory Optimization

The framework includes several memory optimization techniques for training larger models on limited hardware:

- Gradient checkpointing to reduce memory usage
- Regular CUDA cache clearing with the `GradientCallback`
- Safe data handling to prevent out-of-bounds indices
- Mixed precision training with fp16