import argparse
import gc
import subprocess
from babylm_dataset import BabylmDataset, train_tokenizer
from training.train import train_model
from transformers import (
    Gemma2Config,
    Gemma2ForCausalLM,
    Llama4TextConfig,
    Llama4ForCausalLM,
    Olmo2Config,
    Olmo2ForCausalLM
)
import torch
import os

# Set CUDA environment variables
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Force synchronous CUDA operations
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='Preprocessing script')
parser.add_argument('--data_folder', type=str, required=False,
                   default='train_10M',
                   choices=['train_10M', 'train_100M', 'dev', 'test'],
                   help='The data folder to process inside the ./data folder')
parser.add_argument('--model', type=str, required=False,
                   default='llama4',
                   choices=['gemma2', 'llama4', 'olmo2'],
                   help='Model to be trained')
parser.add_argument('--vocab_size', type=int, required=False,
                   default=48000,
                   help='Define the size of vocaburary for tokenizer.')
parser.add_argument('--seq_length', type=int, required=False,
                   default=128,
                   help='Defines how many tokens (words or subwords) a model processes at once.')
parser.add_argument('--batch_size', type=int, required=False,
                   default=16,
                   help='Batch Size for Training')
parser.add_argument('--epoch', type=int, required=False,
                   default=3,
                   help='Number of epoch for training')
parser.add_argument('--hidden_size', type=int, required=False,
                   default=768,
                   help='Hidden size for model architecture')
parser.add_argument('--num_hidden_layers', type=int, required=False,
                   default=12,
                   help='Number of layers in the model')
parser.add_argument('--num_attention_heads', type=int, required=False,
                   default=None,
                   help='Number of attention heads (for LLaMA)')
args = parser.parse_args()

def clean_data(data_folder: str = args.data_folder) -> None:
    """Execute Bash Script for Data Cleaning"""
    print(f"Processing data from: {data_folder}")
    bash_script = './run_preprocess.sh'
    # Run the bash script with the data_folder argument
    try:
        subprocess.run(['chmod', '+x', bash_script])
        result = subprocess.run([bash_script, data_folder],
                               check=True,
                               text=True,
                               capture_output=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        print(e.stderr)

if __name__ == "__main__":
    # Set GPU memory fraction to avoid OOM
    if torch.cuda.is_available():
        print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
        # Limit GPU memory usage
        torch.cuda.empty_cache()
        gc.collect()
    else:
        print("CUDA is not available. Training on CPU.")
        
    SEQ_LENGTH = args.seq_length
    
    # Clean Training set
    clean_data(data_folder=args.data_folder)
    
    # Clean Validation set
    clean_data(data_folder='dev')
    
    # Train Tokenizer
    tokenizer = train_tokenizer(data_folder=args.data_folder, vocab_size=args.vocab_size)
    
    # Double-check tokenizer vocab size
    actual_vocab_size = len(tokenizer.get_vocab())
    print(f"Actual tokenizer vocabulary size: {actual_vocab_size}")
    
    # Create Dataset
    train_dataset = BabylmDataset(f'./data/{args.data_folder}_cleaned',
                                seq_length=SEQ_LENGTH,
                                tokenizer=tokenizer,
                                vocab_size=actual_vocab_size,  # Use actual vocab size
                                random_chunk=True)
    
    eval_dataset = BabylmDataset('./data/dev_cleaned',
                                seq_length=SEQ_LENGTH,
                                tokenizer=tokenizer,
                                vocab_size=actual_vocab_size,  # Use actual vocab size
                                offset=0)
    
    # Load Configurations
    if args.model == 'gemma2':
        config = Gemma2Config(
            vocab_size=actual_vocab_size,  # Use actual tokenizer vocab size
            context_length=SEQ_LENGTH,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            attention_hidden_size=args.hidden_size,
            intermediate_size=4 * args.hidden_size,
            layer_norm_epsilon=1e-5,
            bos_token_id=0,
            eos_token_id=0,
            rescale_every=6,
            tie_word_embeddings=True,
            use_cache=False
        )
        
        model = Gemma2ForCausalLM(config)
        
    elif args.model == 'llama4':
        config = Llama4TextConfig(
            vocab_size=actual_vocab_size,  # Use actual tokenizer vocab size
            context_length=SEQ_LENGTH,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            attention_hidden_size=args.hidden_size,
            num_attention_heads=args.num_attention_heads,
            intermediate_size=4 * args.hidden_size,
            layer_norm_epsilon=1e-5,
            bos_token_id=0,
            eos_token_id=0,
            rescale_every=6,
            tie_word_embeddings=True,
            use_cache=False,
        )
        
        model = Llama4ForCausalLM(config)
    
    elif args.model == 'olmo2':
        config = Olmo2Config(
            vocab_size=actual_vocab_size,  # Use actual tokenizer vocab size
            context_length=SEQ_LENGTH,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            attention_hidden_size=args.hidden_size,
            intermediate_size=4 * args.hidden_size,
            layer_norm_epsilon=1e-5,
            bos_token_id=0,
            eos_token_id=0,
            rescale_every=6,
            tie_word_embeddings=True,
            use_cache=False,
        )
        
        model = Olmo2ForCausalLM(config)
    
    output_dir = f"./models/{args.model}_{args.data_folder}"
    
    # Train Model
    trainer, model = train_model(model=model,
                              tokenizer=tokenizer,
                              train_dataset=train_dataset,
                              eval_dataset=eval_dataset,
                              per_device_train_batch_size=args.batch_size,
                              per_device_eval_batch_size=args.batch_size,
                              output_dir=output_dir,
                              num_train_epochs=args.epoch,
                              fp16=True)
    
    print("✅ Training complete!")
