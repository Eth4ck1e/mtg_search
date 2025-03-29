import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch
import os
import sys
import logging
import warnings
from accelerate import Accelerator

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils.checkpoint")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.file_download")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Disable MPS memory cap (risky, may crash system)
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data_processing.download_bulk_data import download_all
from src.data_processing.compile_initial_training_data import compile_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("huggingface_hub")
logger.setLevel(logging.INFO)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
INITIAL_SUBSET = os.path.join(PROJECT_ROOT, 'data', 'processed', 'training', 'initial', 'initial_subset.csv')
INITIAL_TRAINING_DATA = os.path.join(PROJECT_ROOT, 'data', 'processed', 'training', 'initial', 'initial_training_data.csv')
RULINGS_DATA = os.path.join(PROJECT_ROOT, 'data', 'processed', 'training', 'rulings', 'rulings_data.csv')
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'models', 'checkpoints', 'model_output')
INITIAL_MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'initial_model')

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 64
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 1
SAVE_STEPS = 500
LOGGING_STEPS = 10
LEARNING_RATE = 2e-5
DATALOADER_NUM_WORKERS = 0
GRADIENT_CHECKPOINTING = True

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

def print_memory_usage():
    if device.type == "mps":
        allocated = torch.mps.current_allocated_memory() / 1e9
        print(f"Memory allocated on MPS: {allocated:.2f}GB")

def ensure_data_files():
    required_files = [INITIAL_SUBSET, INITIAL_TRAINING_DATA, RULINGS_DATA]
    missing_files = [f for f in required_files if not os.path.exists(f) or os.path.getsize(f) == 0]
    if missing_files:
        print(f"Missing or empty data files: {missing_files}")
        print("Downloading raw data...")
        download_all()
        print("Processing raw data into CSVs...")
        compile_data()
    else:
        print("All required data files are present.")

def create_input_text(row):
    parts = [
        f"Name: {row['name']}",
        f"Type: {row['type_line']}",
        f"Cost: {row['mana_cost']}" if pd.notna(row['mana_cost']) else "",
        f"Colors: {', '.join(row['colors']) if isinstance(row['colors'], list) else ''}",
        f"Effect: {row['oracle_text']}",
        f"Keywords: {', '.join(row['keywords']) if isinstance(row['keywords'], list) else ''}"
    ]
    return " | ".join(part for part in parts if part)

def main():
    ensure_data_files()

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device.type == "mps" else torch.float32
    ).to(device)
    print("Model loaded successfully.")
    print_memory_usage()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Tokenizer loaded successfully.")

    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled.")

    print("Loading training data...")
    df = pd.read_csv(INITIAL_SUBSET)
    print("Training data loaded successfully.")
    print("Combining fields into input text...")
    df['input_text'] = df.apply(create_input_text, axis=1)
    dataset = Dataset.from_pandas(df[['input_text']])
    print("Input text combined successfully.")

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples['input_text'],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['input_text'])
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    print("Dataset tokenized successfully.")
    print_memory_usage()

    # Clear MPS cache
    if device.type == "mps":
        torch.mps.empty_cache()
        print("MPS cache cleared.")

    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        logging_steps=LOGGING_STEPS,
        learning_rate=LEARNING_RATE,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=False,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        optim="adamw_torch_fused",
        remove_unused_columns=False,
    )
    print("Training arguments set up successfully.")

    print("Initializing accelerator...")
    accelerator = Accelerator(device_placement=True)
    model, tokenized_dataset = accelerator.prepare(model, tokenized_dataset)

    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    print("Trainer initialized successfully.")

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    model.save_pretrained(INITIAL_MODEL_DIR)
    tokenizer.save_pretrained(INITIAL_MODEL_DIR)
    print(f"Initial model saved to {INITIAL_MODEL_DIR}")

if __name__ == '__main__':
    main()