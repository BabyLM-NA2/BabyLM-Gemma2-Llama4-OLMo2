import os
import torch
import gc
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)

class GradientCallback(TrainerCallback):
    """Callback to regularly clear CUDA cache and check memory usage."""
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()

class SafeDataCollator(DataCollatorForLanguageModeling):
    """Data collator that ensures no out-of-bounds indices."""
    def __call__(self, features, return_tensors=None):
        batch = super().__call__(features, return_tensors)
        # Double-check for out-of-bounds indices
        if torch.max(batch["input_ids"]) >= self.tokenizer.vocab_size:
            print(f"Warning: Input IDs exceed vocab size. Max ID: {torch.max(batch['input_ids']).item()}")
            batch["input_ids"] = torch.clamp(batch["input_ids"], min=0, max=self.tokenizer.vocab_size-1)
        if "labels" in batch and batch["labels"].max() >= self.tokenizer.vocab_size:
            print(f"Warning: Labels exceed vocab size. Max label: {batch['labels'].max().item()}")
            batch["labels"] = torch.clamp(batch["labels"], min=0, max=self.tokenizer.vocab_size-1)
        # Handle special -100 values (ignored in loss calculation)
        if "labels" in batch:
            mask = batch["labels"] != -100
            if mask.any():
                valid_labels = batch["labels"][mask]
                if valid_labels.numel() > 0 and valid_labels.max() >= self.tokenizer.vocab_size:
                    valid_labels_clamped = torch.clamp(valid_labels, min=0, max=self.tokenizer.vocab_size-1)
                    batch["labels"][mask] = valid_labels_clamped
        return batch

def train_model(model, tokenizer, train_dataset=None, eval_dataset=None, output_dir="./models",
               num_train_epochs=3, per_device_train_batch_size=1,
               per_device_eval_batch_size=1, gradient_accumulation_steps=8,
               learning_rate=5e-4, weight_decay=0.01, warmup_steps=500, fp16=True,
               resume_from_checkpoint=None):
    """
    Train a RWKV model with enhanced memory safety and optimizations.
    """
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Verify weights are reset
    print(f"Initial model weights norm: {sum(p.norm().item() for p in model.parameters()):.4f}")
    
    # Explicitly reset the weights
    def _init_weights(module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    model.apply(_init_weights)
    print(f"Reset model weights norm: {sum(p.norm().item() for p in model.parameters()):.4f}")

    # Verify model and tokenizer compatibility
    if hasattr(model, 'config') and hasattr(model.config, 'vocab_size'):
        if model.config.vocab_size != len(tokenizer.get_vocab()):
            print(f"Warning: Model vocab size ({model.config.vocab_size}) != Tokenizer vocab size ({len(tokenizer.get_vocab())})")
            print("Adjusting model config to match tokenizer...")
            model.resize_token_embeddings(len(tokenizer.get_vocab()))
    
    # Initialize training arguments with memory optimizations
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        fp16=fp16,  # Enable mixed precision
        logging_dir="./logs",
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        save_total_limit=3,
        dataloader_num_workers=0,  # Use main process for data loading
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,  # Enable gradient checkpointing
        max_grad_norm=1.0,  # Gradient clipping
        optim="adamw_torch",  # Use PyTorch's AdamW
    )
    
    # Create safe data collator
    data_collator = SafeDataCollator(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Initialize trainer with memory callbacks
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[GradientCallback()]
    )
    
    # Train the model with error handling
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save the model
        trainer.save_model()
        
        return trainer, model
    except RuntimeError as e:
        print(f"Error during training: {e}")
        # Save checkpoint even if error occurs
        try:
            checkpoint_dir = f"{output_dir}/checkpoint-error"
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"Saved error checkpoint to {checkpoint_dir}")
        except Exception as save_error:
            print(f"Could not save error checkpoint: {save_error}")
        raise e
