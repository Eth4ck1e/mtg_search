import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Paths
INITIAL_TRAINING_DATA = '/home/eth4ck1e/mtg_search/data/processed/training/initial/initial_training_data.csv'
RULINGS_DATA = '/home/eth4ck1e/mtg_search/data/processed/training/rulings/rulings_data.csv'
MODEL_OUTPUT_DIR = '/home/eth4ck1e/mtg_search/models/checkpoints/model_output_finetune'
FINE_TUNED_MODEL_DIR = '/home/eth4ck1e/mtg_search/models/fine_tuned_model'
INITIAL_MODEL_DIR = '/home/eth4ck1e/mtg_search/models/initial_model'

# Load the initial model and tokenizer
model = AutoModelForCausalLM.from_pretrained(INITIAL_MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(INITIAL_MODEL_DIR)

# Load initial data and rulings
initial_df = pd.read_csv(INITIAL_TRAINING_DATA)
rulings_df = pd.read_csv(RULINGS_DATA)

# Merge rulings with initial data
df = initial_df.merge(rulings_df, on='oracle_id', how='left')
df['rulings'] = df['rulings'].fillna('')

# Combine fields into a single input string, including rulings
def create_input_text(row):
    parts = [
        f"Name: {row['name']}",
        f"Type: {row['type_line']}",
        f"Cost: {row['mana_cost']}" if pd.notna(row['mana_cost']) else "",
        f"Colors: {', '.join(row['colors']) if isinstance(row['colors'], list) else ''}",
        f"Effect: {row['oracle_text']}",
        f"Keywords: {', '.join(row['keywords']) if isinstance(row['keywords'], list) else ''}",
        f"Rulings: {row['rulings']}" if row['rulings'] else ""
    ]
    return " | ".join(part for part in parts if part)

df['input_text'] = df.apply(create_input_text, axis=1)
dataset = Dataset.from_pandas(df[['input_text']])

# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples['input_text'], padding="max_length", truncation=True, max_length=256, return_tensors="pt")

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['input_text'])
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Training arguments for fine-tuning
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=1e-5,
    fp16=True if torch.cuda.is_available() else False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Fine-tune
trainer.train()

# Save the fine-tuned model
model.save_pretrained(FINE_TUNED_MODEL_DIR)
tokenizer.save_pretrained(FINE_TUNED_MODEL_DIR)
print(f"Fine-tuned model saved to {FINE_TUNED_MODEL_DIR}")