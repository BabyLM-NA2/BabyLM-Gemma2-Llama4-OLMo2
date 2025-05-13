import os
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

def train_model(config, model, tokenizer, train_dataset=None, eval_dataset=None, output_dir="./models", 
               num_train_epochs=3, per_device_train_batch_size=32, 
               per_device_eval_batch_size=32, gradient_accumulation_steps=4, 
               learning_rate=5e-4, weight_decay=0.01, warmup_steps=500, fp16=True, 
               resume_from_checkpoint=None):
    """
    Train a RWKV model from Hugging Face's transformers library for causal language modeling.
    """
    # Set CUDA environment variables for better error reporting
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Set deterministic settings for better reproducibility
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    print(f"Resetting model weights...")
    
    # Verify weights are reset by checking the norm before and after initialization
    initial_norm = sum(p.norm().item() for p in model.parameters())
    print(f"Initial model weights norm: {initial_norm:.4f}")
    
    # Explicitly reset the weights to ensure they're properly initialized
    def _init_weights(module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            # Standard initialization for transformers
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    model.apply(_init_weights)
    
    # Verify weights are properly reset
    reset_norm = sum(p.norm().item() for p in model.parameters())
    print(f"Reset model weights norm: {reset_norm:.4f}")
    
    # Validate batch size against GPU memory
    def batch_size_validator(batch_size):
        """Validate batch size against model configuration"""
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            model_memory = config.hidden_size * config.num_hidden_layers * 4 * 4  # Rough estimate
            if model_memory * batch_size > gpu_memory * 0.8:
                return False
        except:
            # If we can't calculate, be conservative
            return batch_size <= 2
        return True

    # Check and adjust batch size if needed
    if not batch_size_validator(per_device_train_batch_size):
        print("Warning: Batch size might be too large for GPU memory, reducing...")
        per_device_train_batch_size = 1
        per_device_eval_batch_size = 1
    
    # Initialize training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        fp16=fp16,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        eval_strategy="steps",  
        save_strategy="steps",       
        load_best_model_at_end=True,
        save_total_limit=3,
        ddp_find_unused_parameters=False
    )

    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train the model
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save the model
    trainer.save_model()
    
    return trainer, model
