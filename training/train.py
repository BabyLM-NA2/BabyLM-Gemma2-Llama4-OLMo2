# Training and deployment utilities
import os
import numpy as np
# from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from huggingface_hub import upload_file, create_repo, HfApi
from model.rwkv import RWKVForCausalLM, RWKVConfig
import random
import torch
import torch.nn.functional as F
import torch.quantization

# Dataset class for pre-tokenized data
class PreTokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, context_length=1024):
        # Load the pre-tokenized data
        self.tokenized_data = torch.load(file_path)
        self.context_length = context_length
        
    def __len__(self):
        return len(self.tokenized_data)
        
    def __getitem__(self, idx):
        # Get tokenized sample
        tokens = self.tokenized_data[idx]
        
        # Check if tokens is a scalar tensor (0-dimensional)
        if isinstance(tokens, torch.Tensor) and tokens.dim() == 0:
            tokens = tokens.unsqueeze(0)  # Convert to 1D tensor with single element
        
        # Handle length - truncate or pad as needed
        if len(tokens) > self.context_length:
            # Random starting point for longer sequences
            start_idx = random.randint(0, len(tokens) - self.context_length - 1)
            tokens = tokens[start_idx:start_idx + self.context_length]
        else:
            # Pad shorter sequences - use tensor operations instead of list
            padding = torch.zeros(self.context_length - len(tokens), 
                                 dtype=tokens.dtype, 
                                 device=tokens.device)
            tokens = torch.cat([tokens, padding])
        
        # Create input and target tensors for causal LM
        inputs = tokens
        labels = tokens.clone()
        
        return {"input_ids": inputs, "labels": labels}



def quantize_model_to_int8(model):
    """
    Apply dynamic quantization to the model for faster inference.
    """
    # Configure model for dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
    return quantized_model


def prepare_model_for_qat(model):
    """
    Prepare model for quantization-aware training.
    """
    # Set QConfig for QAT (needs to be done before any forward pass)
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    
    # Convert model to QAT mode
    torch.quantization.prepare_qat(model, inplace=True)
    return model


def convert_to_quantized(model):
    """
    Convert a trained QAT model to fully quantized INT8.
    """
    # Make sure model is in eval mode
    model.eval()
    # Convert to quantized model
    model_int8 = torch.quantization.convert(model, inplace=False)
    return model_int8


def train_rwkv_with_pretokenized_data(
    model_config=None,
    model_path=None,
    train_file="./data/train_10M_cleaned/tokenized_GPT2TokenizerFast_16000.pt",
    val_file="./data/dev/tokenized_GPT2TokenizerFast_16000.pt",
    tokenizer_name="gpt2",
    context_length=1024,
    batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_epochs=3,
    output_dir="./rwkv-trained",
    hub_model_id=None,
    use_quantization=False,
    quantization_mode="dynamic"  # 'dynamic' or 'qat'
):
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load or create model
    if model_path:
        model = RWKVForCausalLM.from_pretrained(model_path)
    else:
        model = RWKVForCausalLM(model_config)
    
    # Apply quantization if requested
    if use_quantization:
        if quantization_mode == "qat":
            print("Preparing model for Quantization-Aware Training (QAT)...")
            model = prepare_model_for_qat(model)
        elif quantization_mode == "dynamic":
            print("Dynamic quantization will be applied after training")
    
    # Move model to appropriate device
    model = model.to(device)
    
    # Load tokenizer for configuration only (not for tokenization)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Prepare datasets from pre-tokenized files
    train_dataset = PreTokenizedDataset(train_file, context_length)
    val_dataset = PreTokenizedDataset(val_file, context_length) if val_file else None
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        fp16=False,  # Enable mixed precision
        bf16=True,
        logging_steps=10,
        dataloader_num_workers=4,  # Parallel data loading
        save_steps=1000,
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=1000 if val_dataset else None,
        save_total_limit=3,
        push_to_hub=bool(hub_model_id),
        hub_model_id=hub_model_id,
        # gradient_checkpointing=True,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Train the model
    trainer.train()
    
    # Apply dynamic quantization after training if requested
    if use_quantization and quantization_mode == "dynamic":
        print("Applying dynamic quantization to trained model...")
        model = quantize_model_to_int8(model)
    
    # Convert QAT model to quantized model
    if use_quantization and quantization_mode == "qat":
        print("Converting QAT model to quantized model...")
        model = convert_to_quantized(model)
    
    # Save the trained model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return trainer, model


def generate_text(
    model,
    tokenizer,
    prompt,
    max_length=100,
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    repetition_penalty=1.1,
    use_rnn_mode=True,
    device="cuda"
):
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    # Initialize states for RNN mode
    states = [None] * model.config.num_hidden_layers if use_rnn_mode else None
    
    # Generate initial token distribution
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            states=states,
            use_cache=use_rnn_mode,
        )
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[1]
        states = outputs.past_key_values if hasattr(outputs, "past_key_values") else outputs[2]
    
    # Prepare for token generation
    generated = input_ids.clone()
    
    # Auto-regressive generation
    for _ in range(max_length):
        # Get the last token's logits
        next_token_logits = logits[:, -1, :]
        
        # Apply temperature
        next_token_logits = next_token_logits / temperature
        
        # Apply repetition penalty
        if repetition_penalty > 1.0:
            for token_id in set(generated[0].tolist()):
                next_token_logits[0, token_id] /= repetition_penalty
                
        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = -float("inf")
            
        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[0, indices_to_remove] = -float("inf")
            
        # Sample from the filtered distribution
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to generated
        generated = torch.cat((generated, next_token), dim=1)
        
        # Get next logits in RNN mode
        if use_rnn_mode:
            with torch.no_grad():
                outputs = model(
                    input_ids=next_token,
                    states=states,
                    use_cache=True,
                )
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[1]
                states = outputs.past_key_values if hasattr(outputs, "past_key_values") else outputs[2]
        else:
            with torch.no_grad():
                outputs = model(
                    input_ids=generated,
                    use_cache=False,
                )
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[1]
                
        # End generation if EOS token is generated
        if next_token[0, 0].item() == tokenizer.eos_token_id:
            break
            
    # Decode generated tokens
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return text