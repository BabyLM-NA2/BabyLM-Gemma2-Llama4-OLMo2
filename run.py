import argparse
import subprocess
from babylm_dataset import train_tokenizer, BabylmDataset


parser = argparse.ArgumentParser(description='Preprocessing script')
parser.add_argument('--data_folder', type=str, required=False, 
                    default='train_10M', 
                    choices=['train_10M', 'train_100M', 'dev', 'test'],
                    help='The data folder to process inside the ./data folder')
parser.add_argument('--vocab_size', type=int, required=False, 
                    default=16000, 
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
    clean_data(data_folder=args.data_folder)
    # Clean Validation set
    clean_data(data_folder='dev')
    # Train Tokenizer
    gpt2_tokenizer = train_tokenizer(data_folder=args.data_folder, vocab_size=16000)
    # # Create Dataset
    train_dataset = BabylmDataset(f'./data/{args.data_folder}_cleaned', SEQ_LENGTH, tokenizer=gpt2_tokenizer, random_chunk=True)
    full_eval_dataset = BabylmDataset('./data/dev_cleaned', SEQ_LENGTH, tokenizer=gpt2_tokenizer, offset=0)