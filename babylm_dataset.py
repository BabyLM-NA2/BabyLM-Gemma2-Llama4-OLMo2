import os
import torch
from torch.utils.data import Dataset
from random import randrange
from pathlib import Path
from tokenizers import (Tokenizer, decoders, models, pre_tokenizers,
                       processors, trainers)
from tokenizers.normalizers import NFKC
from transformers import GPT2TokenizerFast

def train_tokenizer(data_folder: str, vocab_size: int = 25000):
    """Train Tokenizer on Training Datasets"""
    # Init Tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.normalizer = NFKC()
    
    # Train Tokenizer
    # Using proper special tokens instead of empty strings
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, 
                                 special_tokens=["<|endoftext|>", "<|pad|>", "<|unk|>"])
    
    data_dir = Path(f"data/{data_folder}_cleaned")
    paths = [str(f) for f in data_dir.glob("*") if f.is_file() and f.suffix in [".train"]]
    
    if not paths:
        raise ValueError(f"No training files found in {data_dir}. Make sure data is cleaned.")
        
    tokenizer.train(paths, trainer)
    
    # Save Tokenizer File
    tokenizer_path = f"tokenizer-{vocab_size}.json"
    tokenizer.save(str(tokenizer_path), pretty=True)
    
    # Embedding
    gpt2_tokenizer = GPT2TokenizerFast(tokenizer_file=str(tokenizer_path))
    gpt2_tokenizer.bos_token = "<|endoftext|>"
    gpt2_tokenizer.eos_token = "<|endoftext|>"
    gpt2_tokenizer.pad_token = "<|pad|>"
    
    # Verify vocab size
    actual_vocab = len(gpt2_tokenizer.get_vocab())
    print(f"Tokenizer created with vocab size: {actual_vocab} (requested: {vocab_size})")
    
    return gpt2_tokenizer

class BabylmDataset(Dataset):
    def __init__(self, data_dir: str, vocab_size: int, seq_length: int, tokenizer, offset: int=0, random_chunk: bool=False):
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.offset = offset
        self.tokenizer = tokenizer
        self.random_chunk = random_chunk
        
        # Check if the actual tokenizer vocab size matches the provided vocab size
        actual_vocab_size = len(tokenizer.get_vocab())
        if actual_vocab_size != vocab_size:
            print(f"Warning: Tokenizer vocab size ({actual_vocab_size}) different from provided ({vocab_size})")
            # Use the actual vocab size to prevent index errors
            self.vocab_size = actual_vocab_size
        
        # Set pad token ID for use in getitem
        self.pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None else 0
        
        tokenizer_name = f"GPT2TokenizerFast_{self.vocab_size}"
        tokenized_file = Path(os.path.join(data_dir, f"tokenized_{tokenizer_name}.pt"))
        
        if tokenized_file.exists():
            print(f"Loading data from {tokenized_file}")
            self.data = torch.load(tokenized_file)
            
            # Verify loaded data doesn't exceed vocab size
            max_token = self.data.max().item()
            min_token = self.data.min().item()
            if max_token >= self.vocab_size or min_token < 0:
                print(f"Warning: Loaded data contains token IDs from {min_token} to {max_token}, but valid range is [0, {self.vocab_size-1}]")
                print("Clamping token IDs to prevent CUDA errors...")
                self.data = torch.clamp(self.data, min=0, max=self.vocab_size-1)
        else:
            data = []
            src_files = [str(f) for f in Path(data_dir).glob("**/*") 
                        if f.is_file() and not f.name.endswith(".DS_Store") and f.suffix in [".train", ".dev"]]
            
            if not src_files:
                raise ValueError(f"No source files found in {data_dir}")
            
            for src_file in src_files:
                text = Path(src_file).read_text(encoding="utf-8")
                encoded = self.tokenizer.encode(text)
                print("🔥", src_file, "len:", len(encoded))
                data.extend(encoded)
            
            self.data = torch.tensor(data)
            
            # Check for out-of-bound token IDs before saving
            max_token = self.data.max().item()
            min_token = self.data.min().item()
            if max_token >= self.vocab_size or min_token < 0:
                print(f"Warning: Generated data contains token IDs from {min_token} to {max_token}, but valid range is [0, {self.vocab_size-1}]")
                print("Clamping token IDs to prevent CUDA errors...")
                self.data = torch.clamp(self.data, min=0, max=self.vocab_size-1)
            
            # Save tokenized data
            print(f"Saving data to {tokenized_file}")
            torch.save(self.data, tokenized_file)
    
    def __len__(self):
        if self.random_chunk:
            return max(0, len(self.data) // self.seq_length - 1)  # Ensure non-negative length
        else:
            return max(0, (len(self.data) - self.offset) // self.seq_length)
    
    def __getitem__(self, i):
        try:
            # Get tokens safely with bounds checking
            if self.random_chunk:
                offset = randrange(self.seq_length) if self.seq_length > 1 else 0
                start_idx = i*self.seq_length+offset
                end_idx = (i+1)*self.seq_length+offset
            else:
                start_idx = i*self.seq_length+self.offset
                end_idx = (i+1)*self.seq_length+self.offset
            
            # Ensure indices are within bounds
            start_idx = max(0, min(start_idx, len(self.data)-1))
            end_idx = max(start_idx+1, min(end_idx, len(self.data)))
            
            tokens = self.data[start_idx:end_idx]
            
            # Ensure token length is exactly seq_length
            if len(tokens) < self.seq_length:
                # Pad with the pad token ID
                padding = torch.full((self.seq_length - len(tokens),), self.pad_token_id, dtype=torch.long)
                tokens = torch.cat([tokens, padding])
            elif len(tokens) > self.seq_length:
                # Truncate to exact length
                tokens = tokens[:self.seq_length]
            
            # Validate token IDs are within bounds
            vocab_size = self.vocab_size
            
            # Always clamp to ensure bounds
            tokens = torch.clamp(tokens, min=0, max=vocab_size-1)
            
            # For language modeling
            input_ids = tokens
            labels = tokens.clone()
            
            # Final validation before returning
            if input_ids.max().item() >= vocab_size or input_ids.min().item() < 0:
                print(f"Critical: Token IDs still out of bounds after clamping!")
                input_ids = torch.clamp(input_ids, min=0, max=vocab_size-1)
                labels = torch.clamp(labels, min=0, max=vocab_size-1)
            
            return {
                "input_ids": input_ids,
                "labels": labels
            }
        
        except Exception as e:
            print(f"Error in __getitem__({i}): {str(e)}")
            # Return a safe fallback item
            safe_tokens = torch.full((self.seq_length,), self.pad_token_id, dtype=torch.long)
            return {
                "input_ids": safe_tokens,
                "labels": safe_tokens.clone()
            }
