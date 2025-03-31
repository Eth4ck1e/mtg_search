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
        if state.global_step % 50 == 0:  # Debug every 50 steps
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
                loss = outputs.loss
                print(f"\nStep {state.global_step}: Loss: {loss.item()}")
            model.train()


def load_data():
    # Load the processed data from complete_set.csv
    data_path = os.path.join(config.PROCESSED_DATA_DIR, 'complete_set.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    df = pd.read_csv(data_path, encoding='utf-8')
    # Ensure the 'text' column exists and is not empty
    if 'text' not in df.columns:
        raise ValueError("The 'text' column is missing from complete_set.csv")
    df = df[df['text'].str.strip() != '']

    # Convert to a datasets.Dataset object
    dataset = Dataset.from_pandas(df)
    print(f"Loaded dataset with {len(dataset)} examples")
    return dataset


def tokenize_function(examples):
    # Tokenize without padding to max_length, let the collator handle padding
    return tokenizer(examples['text'], truncation=True, max_length=config.MAX_LENGTH)


def train():
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(config.MODEL_NAME)

    # Check for MPS availability and set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for training")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU for training")
    model.to(device)

    # Load the dataset
    dataset = load_data()

    # Tokenize the dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    print(f"Tokenized dataset with {len(tokenized_dataset)} examples")

    # Debug: Inspect a few tokenized examples
    print("First few tokenized examples:")
    for i in range(min(3, len(tokenized_dataset))):
        example = tokenized_dataset[i]
        non_padding_length = sum(example['attention_mask'])
        print(
            f"Example {i}: input_ids={example['input_ids'][:10]}... "
            f"attention_mask={example['attention_mask'][:10]}... "
            f"total_length={len(example['input_ids'])} "
            f"non_padding_length={non_padding_length}"
        )

    tokenized_dataset.set_format("torch")

    # Data collator for masked language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,  # Enable masked language modeling
        mlm_probability=0.15,  # Mask 15% of tokens (standard for BERT-style MLM)
        pad_to_multiple_of=8,  # Ensure padding aligns with model requirements
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.CHECKPOINT_DIR,
        num_train_epochs=3,  # Keep 3 epochs since training is fast
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=5e-5,
        logging_dir='./logs',
        logging_steps=20,  # Log every 20 steps for more frequent updates
        logging_first_step=False,
        logging_strategy="steps",
        save_steps=5000,
        save_total_limit=2,
        max_grad_norm=1.0,
        dataloader_num_workers=config.DATALOADER_NUM_WORKERS,
        fp16=False,  # Disable FP16 (MPS doesn't support it well)
    )

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[DebugCallback()],
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    model.save_pretrained(config.MODEL_DIR)
    tokenizer.save_pretrained(config.MODEL_DIR)
    print(f"Model and tokenizer saved to {config.MODEL_DIR}")


if __name__ == "__main__":
    train()