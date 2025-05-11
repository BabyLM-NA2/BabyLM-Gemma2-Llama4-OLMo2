# Training and deployment utilities
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
import gc
import random
import torch
import torch.nn.functional as F
import torch.quantization
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers.optimization import Adafactor
from transformers.trainer_callback import TrainerCallback
from model.llama import LlamaForCausalLM, LlamaConfig

from model.rwkv import RWKVForCausalLM, RWKVConfig

class CacheFlushCallback(TrainerCallback):
    def __init__(self, flush_interval=10):
        self.flush_interval = flush_interval
        self.step_count = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        """More selective cache clearing"""
        self.step_count += 1
        
        # Clear cache periodically rather than every step
        if self.step_count % self.flush_interval == 0:
            get_accelerator().empty_cache()
        return control

# Dataset class for pre-tokenized data with memory-efficient loading
class PreTokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, context_length=1024, chunk_size=1000000):
        # Load the pre-tokenized data info
        self.file_path = file_path
        self.context_length = context_length
        self.chunk_size = chunk_size
        
        # Get total size without loading all data
        temp_data = torch.load(file_path)
        self.total_length = len(temp_data)
        del temp_data  # Free memory
        
        # Initialize chunk tracking
        self.current_chunk = None
        self.current_chunk_idx = -1
        
    def _load_chunk(self, chunk_idx):

        gc.collect()
        torch.cuda.empty_cache()  # Clear cache before loading new data

        # Load only the necessary chunk of data
        full_data = torch.load(self.file_path)
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.total_length)
        chunk = full_data[start_idx:end_idx]
        del full_data  # Free memory immediately

        # Force cache clearing after deletion
        gc.collect()
        torch.cuda.empty_cache()

        return chunk
        
    def __len__(self):
        return self.total_length
        
    def __getitem__(self, idx):
        # Determine which chunk this index belongs to
        chunk_idx = idx // self.chunk_size
        
        # Load chunk if needed
        if self.current_chunk_idx != chunk_idx:
            self.current_chunk = self._load_chunk(chunk_idx)
            self.current_chunk_idx = chunk_idx
        
        # Get item from current chunk
        local_idx = idx % self.chunk_size
        tokens = self.current_chunk[local_idx]
        
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
    train_file="./data/train_10M_cleaned/tokenized_OLMo2SuperBPE.pt",
    val_file="./data/dev/tokenized_OLMo2SuperBPE.pt",
    tokenizer_name="UW/OLMo2-8B-SuperBPE-t180k",
    context_length=1024,
    batch_size=8,
    gradient_accumulation_steps=32,
    learning_rate=5e-5,
    num_epochs=3,
    output_dir="./rwkv-trained",
    hub_model_id=None,
    use_quantization=False,
    quantization_mode="dynamic",
    use_deepspeed=True,  # Enable DeepSpeed
    deepspeed_config="ds_config.json"  # Path to DeepSpeed config
):
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of available memory
    
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
    
    # Prepare datasets from pre-tokenized files with memory-efficient loading
    train_dataset = PreTokenizedDataset(train_file, context_length)
    val_dataset = PreTokenizedDataset(val_file, context_length) if val_file else None
    
    # Create 8-bit optimizer to save memory
    optimizer = Adafactor(
        model.parameters(),
        lr=learning_rate,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False
        )
    
    # Training arguments with memory optimizations
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        fp16=True,
        bf16=False,
        logging_steps=10,
        dataloader_num_workers=4,
        save_steps=1000,
        save_total_limit=3,
        push_to_hub=bool(hub_model_id),
        hub_model_id=hub_model_id,
        gradient_checkpointing=True,
        deepspeed=deepspeed_config if use_deepspeed else None,
        # For better multi-GPU handling
        local_rank=-1,  # Managed by distributed launcher
        ddp_find_unused_parameters=False,  # More efficient DDP
        tf32=True,  # For A100 GPUs specifically
        )
    
    # Initialize trainer with custom optimizer and cache flushing callback
    cache_flush_callback = CacheFlushCallback()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=(optimizer, None)  # Use custom optimizer
    )
    trainer.add_callback(cache_flush_callback)
    
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


def train_llama_with_pretokenized_data(
    model_config=None,
    model_path=None,
    train_file="./data/train_10M_cleaned/tokenized_OLMo2SuperBPE.pt",
    val_file="./data/dev/tokenized_OLMo2SuperBPE.pt",
    tokenizer_name="UW/OLMo2-8B-SuperBPE-t180k",
    context_length=1024,
    batch_size=8,
    gradient_accumulation_steps=16,
    learning_rate=5e-5,
    num_epochs=3,
    output_dir="./llama-trained",
    hub_model_id=None,
    use_quantization=False,
      # Only use DeepSpeed if available
    deepspeed_config="ds_config.json"  # Path to DeepSpeed config
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of available memory
    
    # Load or create model
    if model_path:
        model = LlamaForCausalLM.from_pretrained(model_path)
    else:
        model = LlamaForCausalLM(model_config)
    
    # Move model to appropriate device
    model = model.to(device)
    
    # Load tokenizer for configuration only (not for tokenization)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Prepare datasets from pre-tokenized files with memory-efficient loading
    train_dataset = PreTokenizedDataset(train_file, context_length)
    val_dataset = PreTokenizedDataset(val_file, context_length) if val_file else None
    
    # Create 8-bit optimizer to save memory
    optimizer = Adafactor(
        model.parameters(),
        lr=learning_rate,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False
        )
    
    # Training arguments with memory optimizations
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        fp16=torch.cuda.is_available(),
        bf16=False,
        logging_steps=10,
        dataloader_num_workers=1,
        save_steps=1000,
        save_total_limit=3,
        push_to_hub=bool(hub_model_id),
        hub_model_id=hub_model_id,
        gradient_checkpointing=True,
        deepspeed=deepspeed_config if HAS_DEEPSPEED else None,
        # For better multi-GPU handling
        local_rank=-1,  # Managed by distributed launcher
        ddp_find_unused_parameters=False,  # More efficient DDP
        tf32=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,  # For A100 GPUs specifically
        )
    
    # Initialize trainer with custom optimizer and cache flushing callback
    cache_flush_callback = CacheFlushCallback()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=(optimizer, None)  # Use custom optimizer
    )
    trainer.add_callback(cache_flush_callback)
    
    # Train the model
    trainer.train()
    
    # Apply dynamic quantization after training if requested
    if use_quantization:
        print("Applying dynamic quantization to trained model...")
        model = quantize_model_to_int8(model)
    
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
