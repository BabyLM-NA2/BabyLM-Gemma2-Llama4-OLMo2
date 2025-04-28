import argparse
import subprocess
try:
    from ds_compat import get_accelerator, HAS_DEEPSPEED
    HAS_DEEPSPEED = True
except ImportError:
    import torch
    HAS_DEEPSPEED = False
    
    # Create a mock accelerator
    class MockAccelerator:
        def empty_cache(self):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_accelerator():
        return MockAccelerator()

from babylm_dataset import load_olmo_tokenizer, BabylmDataset
from training.train import train_rwkv_with_pretokenized_data, generate_text, train_llama_with_pretokenized_data
from model.rwkv import RWKVConfig
from model.llama import LlamaConfig, LlamaForCausalLM



parser = argparse.ArgumentParser(description='Preprocessing script')

parser.add_argument('--data_folder', type=str, required=False, 
                    default='train_10M', 
                    choices=['train_10M', 'train_100M', 'dev', 'test'],
                    help='The data folder to process inside the ./data folder')
parser.add_argument('--model', type=str, required=False,
                    default='rwkv')
parser.add_argument('--vocab_size', type=int, required=False, 
                    default=200000, 
                    help='Define the size of vocaburary for tokenizer.')
parser.add_argument('--seq_length', type=int, required=False, 
                    default=128, 
                    help='Defines how many tokens (words or subwords) a model processes at once.')
parser.add_argument('--batch_size', type=int, required=False, 
                    default=8, 
                    help='Batch Size for Training')
# Add the new arguments here
parser.add_argument('--hidden_size', type=int, required=False, 
                   default=768,
                   help='Hidden size for model architecture')
parser.add_argument('--num_hidden_layers', type=int, required=False,
                   default=12,
                   help='Number of layers in the model')
parser.add_argument('--num_attention_heads', type=int, required=False,
                   default=12,
                   help='Number of attention heads (for LLaMA)')
# Add the new arguments here


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
    SEQ_LENGTH = args.seq_length
    # Clean Training set
    clean_data(data_folder=args.data_folder)
    # Clean Validation set
    clean_data(data_folder='dev')
    # Train Tokenizer
    tokenizer = load_olmo_tokenizer()
    # Create Dataset
    train_dataset = BabylmDataset(f'./data/{args.data_folder}_cleaned', SEQ_LENGTH, tokenizer=tokenizer, random_chunk=True)
    full_eval_dataset = BabylmDataset('./data/dev_cleaned', SEQ_LENGTH, tokenizer=tokenizer, offset=0)
    
    # Create configuration
    config = RWKVConfig(
        vocab_size=args.vocab_size,
        context_length=1024,
        hidden_size=768,
        num_hidden_layers=12,
    )
    
    if args.model == 'rwkv':
        # Training example
        trainer, model = train_rwkv_with_pretokenized_data(
            model_config=config,
            train_file=f"./data/{args.data_folder}_cleaned/tokenized_OLMo2SuperBPE.pt",
            val_file=f"./data/dev_cleaned/tokenized_OLMo2SuperBPE.pt",
            output_dir=f"./output/rwkv-trained-model-{args.data_folder}",
            batch_size=args.batch_size
        )
    elif args.model == 'llama':
        # Use hardcoded values instead of command-line args
        hidden_size = 768
        num_hidden_layers = 12
        num_attention_heads = 12
        # Create LLaMA configuration
        config = LlamaConfig(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            max_position_embeddings=1024
        )
        
        # Training LLaMA
        trainer, model = train_llama_with_pretokenized_data(
            model_config=config,
            train_file=f"./data/{args.data_folder}_cleaned/tokenized_OLMo2SuperBPE.pt",
            val_file=f"./data/dev_cleaned/tokenized_OLMo2SuperBPE.pt",
            output_dir=f"./output/llama-trained-model-{args.data_folder}",
            batch_size=args.batch_size
        )
    
    # Generate text example
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt="How many r in the word 'strawberry'",
        max_length=100,
        temperature=0.7,
        use_rnn_mode=True
    )
    
    print(generated_text)