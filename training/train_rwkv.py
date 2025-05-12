from transformers import (
    RwkvForCausalLM,
    RwkvConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

def train_rwkv(tokenizer, train_data, eval_data, vocab_size):
    config = RwkvConfig(
        vocab_size=vocab_size,
        hidden_size=768,   # Smaller size for demonstration
        num_hidden_layers=12,
        attention_hidden_size=768
    )

    # Initialize a model from scratch with random weights
    model = RwkvForCausalLM(config)
    
    training_args = TrainingArguments(
    output_dir="./model",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_steps=500,
    logging_steps=500,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    weight_decay=0.1,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=1000,
    fp16=True
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False  # Not using masked language modeling
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator
    )

    # Train the model
    trainer.train()

    # Save the trained model
    model.save_pretrained("./rwkv-trained-from-scratch")
    tokenizer.save_pretrained("./rwkv-trained-from-scratch")
    
    return trainer, model

