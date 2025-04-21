import argparse
import subprocess
from babylm_dataset import load_olmo_tokenizer, BabylmDataset
from training.train import train_rwkv_with_pretokenized_data, generate_text
from model.rwkv import RWKVConfig


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
    # clean_data(data_folder=args.data_folder)
    # Clean Validation set
    # clean_data(data_folder='dev')
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
            output_dir="./output/rwkv-trained-model",
            # context_length=128,
            # hub_model_id="your-username/rwkv-custom-model"
        )
    elif args.model == 'llama':
        pass
    
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