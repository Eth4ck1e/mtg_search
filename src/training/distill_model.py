import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Paths
DEFAULT_CARDS = '/home/eth4ck1e/mtg_search/data/raw/bulk_jsons/default-cards.json'
DISTILLATION_DATA = '/home/eth4ck1e/mtg_search/data/processed/training/distillation/card_embeddings.csv'
FINE_TUNED_MODEL_DIR = '/home/eth4ck1e/mtg_search/models/fine_tuned_model'
DISTILLED_MODEL_DIR = '/home/eth4ck1e/mtg_search/models/distilled_model'
DISTILLATION_OUTPUT_DIR = '/home/eth4ck1e/mtg_search/models/checkpoints/distillation_output'

# Step 1: Generate teacher embeddings
teacher_model = AutoModelForCausalLM.from_pretrained(FINE_TUNED_MODEL_DIR)
teacher_tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_DIR)
teacher_model.eval()

# Load default cards
df = pd.read_json(DEFAULT_CARDS)
df = df[['oracle_text']].dropna(subset=['oracle_text'])
df['oracle_text'] = df['oracle_text'].str.strip().str.replace(r'\s+', ' ', regex=True)

# Tokenize and generate embeddings
def get_teacher_embeddings(texts):
    inputs = teacher_tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        outputs = teacher_model(**inputs, output_hidden_states=True)
        # Use the last hidden state as the embedding
        embeddings = outputs.hidden_states[-1][:, 0, :]  # [CLS] token embedding
    return embeddings.cpu().numpy()

# Process in batches to avoid memory issues
batch_size = 32
embeddings = []
for i in range(0, len(df), batch_size):
    batch_texts = df['oracle_text'][i:i+batch_size].tolist()
    batch_embeddings = get_teacher_embeddings(batch_texts)
    embeddings.extend(batch_embeddings)
df['teacher_embedding'] = embeddings

# Save distillation data
df[['oracle_text', 'teacher_embedding']].to_csv(DISTILLATION_DATA, index=False)
print(f"Distillation data saved to {DISTILLATION_DATA}")

# Step 2: Train the student model
student_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
student_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# Load distillation dataset
dataset = Dataset.from_pandas(df[['oracle_text', 'teacher_embedding']])

# Tokenize the data
def tokenize_function(examples):
    tokenized = student_tokenizer(examples['oracle_text'], padding="max_length", truncation=True, max_length=256, return_tensors="pt")
    tokenized['teacher_embedding'] = examples['teacher_embedding']
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['oracle_text'])
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'teacher_embedding'])

# Custom training loop for distillation
class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        student_hidden = outputs.hidden_states[-1][:, 0, :]  # [CLS] token embedding
        teacher_hidden = inputs['teacher_embedding']
        loss = nn.MSELoss()(student_hidden, teacher_hidden)
        return (loss, outputs) if return_outputs else loss

# Training arguments
training_args = TrainingArguments(
    output_dir=DISTILLATION_OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-5,
    fp16=False,  # CPU training
    dataloader_num_workers=16,
)

# Trainer
trainer = DistillationTrainer(
    model=student_model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train
trainer.train()

# Save the distilled model
student_model.save_pretrained(DISTILLED_MODEL_DIR)
student_tokenizer.save_pretrained(DISTILLED_MODEL_DIR)
print(f"Distilled model saved to {DISTILLED_MODEL_DIR}")