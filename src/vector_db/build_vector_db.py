# mtg_search/src/vector_db/build_vector_db.py
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss
import numpy as np
import os
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, 'data', 'vector_db')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'initial_model')

MODEL_NAME = "deepseek-ai/deepseek-llm-7b-base"
MAX_LENGTH = 64


def create_input_text(card):
    parts = [
        f"Name: {card.get('name', '')}",
        f"Type: {card.get('type_line', '')}",
        f"Cost: {card.get('mana_cost', '')}" if card.get('mana_cost') else "",
        f"Colors: {', '.join(card.get('colors', [])) if card.get('colors') else ''}",
        f"Effect: {card.get('oracle_text', '')}",
        f"Keywords: {', '.join(card.get('keywords', [])) if card.get('keywords') else ''}"
    ]
    return " | ".join(part for part in parts if part)


def build_vector_db():
    # Load trained model and tokenizer
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model.eval()

    # Load oracle cards
    oracle_file = os.path.join(RAW_DATA_DIR, 'oracle-cards.json')
    if not os.path.exists(oracle_file):
        print("oracle-cards.json not found. Downloading...")
        from src.data_processing.download_bulk_data import download_all
        download_all()
    with open(oracle_file, 'r') as f:
        oracle_cards = json.load(f)

    # Generate input texts
    texts = [create_input_text(card) for card in oracle_cards]
    print(f"Processing {len(texts)} unique cards...")

    # Generate embeddings
    embeddings = []
    batch_size = 16  # Small batches for 32GB
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(
            device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            embedding = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()  # Mean pooling
        embeddings.append(embedding)
        print(f"Processed batch {i // batch_size + 1}/{len(texts) // batch_size + 1}")
        torch.mps.empty_cache()  # Clear memory per batch

    embeddings = np.vstack(embeddings)
    print(f"Generated embeddings shape: {embeddings.shape}")

    # Save to FAISS index
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance
    index.add(embeddings)
    faiss.write_index(index, os.path.join(VECTOR_DB_DIR, 'card_vectors.faiss'))

    # Save metadata
    metadata = pd.DataFrame([
        {'name': card.get('name', ''), 'oracle_id': card.get('oracle_id', '')} for card in oracle_cards
    ])
    metadata.to_csv(os.path.join(VECTOR_DB_DIR, 'card_metadata.csv'), index=False)
    print("Vector database built in data/vector_db/ using oracle-cards.json.")


if __name__ == "__main__":
    build_vector_db()