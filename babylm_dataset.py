import os
import torch
from torch.utils.data import Dataset
from random import randrange
from pathlib import Path
from transformers import AutoTokenizer
from tokenizers import (Tokenizer, decoders, models, pre_tokenizers,
                        processors, trainers)
from tokenizers.normalizers import NFKC
from transformers import GPT2TokenizerFast


def load_olmo_tokenizer():
    """Load the pre-trained OLMo2-8B-SuperBPE tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained("UW/OLMo2-8B-SuperBPE-t180k")
    vocab_size = len(tokenizer.get_vocab())
    return tokenizer, vocab_size

def train_tokenizer(data_folder: str, vocab_size: int = 25000):
    """Train Tokenizer on Training Datasets"""
    
    # Init Tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.normalizer = NFKC()

    # Train Tokenizer
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=["<pad>", "<s>", "</s>"])
    
    data_dir = Path(f"data/{data_folder}_cleaned")
    paths = [str(f) for f in data_dir.glob("*") if f.is_file() and f.suffix in [".train"]]
    tokenizer.train(paths, trainer)

    # Save Tokenizer File
    tokenizer_path =  f"tokenizer-{vocab_size}.json"
    tokenizer.save(str(tokenizer_path), pretty=True)
    
    # Embedding
    gpt2_tokenizer = GPT2TokenizerFast(tokenizer_file= str(tokenizer_path))
    gpt2_tokenizer.bos_token = "<s>"
    gpt2_tokenizer.eos_token = "</s>"
    gpt2_tokenizer.pad_token = "<pad>"
    
    return gpt2_tokenizer

class BabylmDataset(Dataset):
    def __init__(self, data_dir: str, seq_length: int, tokenizer, offset: int=0, random_chunk: bool=False):
        self.seq_length = seq_length
        self.offset = offset
        self.tokenizer = tokenizer
        self.random_chunk = random_chunk

        # Change the tokenizer naming to reflect OLMo2
        tokenizer_name = "GPT2TokenizerFast_16000"
        tokenized_file = Path(os.path.join(data_dir, f"tokenized_{tokenizer_name}.pt"))

        if tokenized_file.exists():
            print(f"Loading data from {tokenized_file}")
            self.data = torch.load(tokenized_file)
        else:
            data = []
            src_files = [str(f) for f in Path(data_dir).glob("**/*")
                         if f.is_file() and not f.name.endswith(".DS_Store") and f.suffix in [".train", ".dev"]]

            for src_file in src_files:
                text = Path(src_file).read_text(encoding="utf-8")
                encoded = self.tokenizer.encode(text)
                print("🔥", src_file, "len:", len(encoded))
                data.extend(encoded)

            self.data = torch.tensor(data)

            # Save tokenized data
            print(f"Saving data to {tokenized_file}")
            torch.save(self.data, tokenized_file)

    def __len__(self):
        if self.random_chunk:
            return len(self.data) // self.seq_length - 1
        else:
            return (len(self.data) - self.offset) // self.seq_length

    def __getitem__(self, i):
        if self.random_chunk:
            offset = randrange(self.seq_length)
            tokens = self.data[i*self.seq_length+offset:(i+1)*self.seq_length+offset]
        else:
            tokens = self.data[i*self.seq_length+self.offset:(i+1)*self.seq_length+self.offset]

        # Add validation to prevent CUDA errors
        max_token_id = tokens.max().item()
        vocab_size = len(self.tokenizer.get_vocab())
        if max_token_id >= vocab_size:
            print(f"Warning: Found token ID {max_token_id} exceeds vocab size {vocab_size}")
            # Clip token IDs to prevent CUDA errors
            tokens = torch.clamp(tokens, max=vocab_size-1)

        return {
            "input_ids": tokens,
            "labels": tokens.clone()
        }