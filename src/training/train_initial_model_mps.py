# mtg_search/src/training/train_initial_model_mps.py
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling, TrainerCallback
from datasets import Dataset
import os
from src import config  # Use absolute import


# Custom callback to debug training
class DebugCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.dataloader_iter = None

    def on_train_begin(self, args, state, control, **kwargs):
        # Initialize the dataloader iterator at the start of training
        self.dataloader_iter = iter(kwargs['train_dataloader'])

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % 10 == 0:  # Debug every 10 steps
            model.eval()
            with torch.no_grad():
                try:
                    # Get the next batch from the dataloader iterator
                    batch = next(self.dataloader_iter)
                except StopIteration:
                    # If the iterator is exhausted, restart it
                    self.dataloader_iter = iter(kwargs['train_dataloader'])
                    batch = next(self.dataloader_iter)
                inputs = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**inputs)
                logits = outputs.logits
                loss = outputs.loss
                print(f"\nStep {state.global_step}:")
                print(f"Loss: {loss.item()}")
                print(f"Sample logits (first token, first 5 values): {logits[0, 0, :5]}")
                has_nan = torch.isnan(logits).any().item()
                has_inf = torch.isinf(logits).any().item()
                print(f"Logits contain nan: {has_nan}")
                print(f"Logits contain inf: {has_inf}")
            model.train()

            (min(3, len(tokenized_dataset)))


def tokenize_function(examples):
    # Tokenize without padding to max_length, let the collator handle padding
    return tokenizer(examples['text'], truncation=True, max_length=config.MAX_LENGTH)


def train():
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(config.MODEL_NAME)  # Use float32
    device = torch.device("cpu")  # Switch to CPU to avoid MPS issues
    model.to(device)

    dataset = load_data()

    # Debug: Print the first few examples in the dataset before tokenization
    print("First few examples in dataset (before tokenization):")
    for i in range(min(3, len(dataset))):
        example = dataset[i]
        print(f"Example {i}: text={example['text'][:100]}...")

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

    # Debug: Print the first few examples in the tokenized dataset
    print("First few examples in tokenized dataset:")
    for i in range(min(3, len(tokenized_dataset))):
        example = tokenized_dataset[i]
        # Count non-padding tokens (where attention_mask == 1)
        non_padding_length = sum(example['attention_mask'])
        print(
            f"Example {i}: input_ids={example['input_ids'][:10]}... attention_mask={example['attention_mask'][:10]}... total_length={len(example['input_ids'])} non_padding_length={non_padding_length}")

    tokenized_dataset.set_format("torch")

    # Data collator for masked language modeling (automatically masks tokens and creates labels)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,  # Enable masked language modeling
        mlm_probability=0.3,  # Mask 30% of tokens
        pad_to_multiple_of=8,  # Ensure padding aligns with model requirements
    )

    # Debug: Inspect the first batch before and after data collation
    batch_before = [tokenized_dataset[i] for i in range(min(4, len(tokenized_dataset)))]
    print("First batch before data collation:")
    print(f"Sample input_ids (before): {batch_before[0]['input_ids'][:10]}")
    print(f"Sample attention_mask (before): {batch_before[0]['attention_mask'][:10]}")

    batch_after = data_collator(batch_before)
    print("First batch after data collation:")
    print(f"input_ids shape: {batch_after['input_ids'].shape}")
    print(f"labels shape: {batch_after['labels'].shape}")
    print(f"Sample input_ids (after): {batch_after['input_ids'][0][:10]}")
    print(f"Sample labels (after): {batch_after['labels'][0][:10]}")
    # Count the number of masked tokens (where labels != -100)
    num_masked = sum(1 for label in batch_after['labels'][0] if label != -100)
    print(f"Number of masked tokens in first sequence: {num_masked}")

    # Debug: Inspect the model's forward pass
    model.eval()
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in batch_after.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        print("Model outputs (logits) for first sequence:")
        print(f"Logits shape: {logits.shape}")
        print(f"Sample logits (first token, first 5 values): {logits[0, 0, :5]}")
        # Check for nan or inf in logits
        has_nan = torch.isnan(logits).any().item()
        has_inf = torch.isinf(logits).any().item()
        print(f"Logits contain nan: {has_nan}")
        print(f"Logits contain inf: {has_inf}")

    training_args = TrainingArguments(
        output_dir=config.CHECKPOINT_DIR,
        num_train_epochs=1,  # Reduced to 1 epoch for faster iteration
        per_device_train_batch_size=8,  # Increased batch size to reduce steps
        per_device_eval_batch_size=8,  # Increased eval batch size
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=2e-6,  # Further lower the learning rate
        logging_dir='./logs',
        logging_steps=50,  # Log every 50 steps
        logging_first_step=False,  # Avoid logging on the first step
        logging_strategy="steps",  # Ensure logging is controlled by steps
        save_steps=5000,  # Save every 5000 steps
        save_total_limit=2,
        max_grad_norm=0.1,  # Stricter gradient clipping
        dataloader_num_workers=config.DATALOADER_NUM_WORKERS,
        fp16=False,  # Disable FP16
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,  # Use data collator for MLM
        callbacks=[DebugCallback()],  # Add debug callback to inspect training
    )

    trainer.train()

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    model.save_pretrained(config.MODEL_DIR)
    tokenizer.save_pretrained(config.MODEL_DIR)
    print(f"Model and tokenizer saved to {config.MODEL_DIR}")


if __name__ == "__main__":
    train()