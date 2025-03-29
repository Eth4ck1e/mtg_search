# mtg_search/src/vector_db/build_vector_db.py
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss
import numpy as np
import os
import json
import sys
import time

# Adjust sys.path to find src.data_processing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'bulk_jsons')
VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, 'data', 'vector_db')  # Both files here
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


def format_time(seconds):
    """Convert seconds to a human-readable time format (e.g., '5m 30s')."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}m {seconds}s"


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

    # Load existing metadata if it exists
    metadata_file = os.path.join(VECTOR_DB_DIR, 'card_metadata.csv')
    if os.path.exists(metadata_file):
        existing_metadata = pd.read_csv(metadata_file)
        existing_oracle_ids = set(existing_metadata['oracle_id'].values)
    else:
        existing_metadata = None
        existing_oracle_ids = set()

    # Identify new cards
    new_cards = [card for card in oracle_cards if card.get('oracle_id', '') not in existing_oracle_ids]
    print(f"Found {len(new_cards)} new cards to process out of {len(oracle_cards)} total cards.")

    if not new_cards:
        print("No new cards to process. Vector database is up to date.")
        return

    # Generate input texts for new cards
    texts = [create_input_text(card) for card in new_cards]

    # Generate embeddings with progress tracking
    embeddings = []
    batch_size = 16  # Small batches for 32GB
    total_batches = (len(texts) + batch_size - 1) // batch_size
    batch_times = []  # Track time per batch

    for i in range(0, len(texts), batch_size):
        start_time = time.time()
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(
            device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            embedding = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()  # Mean pooling
        embeddings.append(embedding)
        torch.mps.empty_cache()  # Clear memory per batch

        # Calculate batch time and update average
        batch_time = time.time() - start_time
        batch_times.append(batch_time)
        avg_batch_time = sum(batch_times) / len(batch_times)
        batches_remaining = total_batches - (i // batch_size + 1)
        eta_seconds = avg_batch_time * batches_remaining

        # Print progress on the same line
        progress_msg = (f"\rProcessed batch {i // batch_size + 1}/{total_batches} | "
                        f"Batch time: {format_time(batch_time)} | "
                        f"Avg batch time: {format_time(avg_batch_time)} | "
                        f"ETA: {format_time(eta_seconds)}")
        sys.stdout.write(progress_msg)
        sys.stdout.flush()

    # Ensure a newline after the loop
    print()

    embeddings = np.vstack(embeddings)
    print(f"Generated embeddings shape: {embeddings.shape}")

    # Load or create FAISS index
    faiss_file = os.path.join(VECTOR_DB_DIR, 'card_vectors.faiss')
    if os.path.exists(faiss_file):
        index = faiss.read_index(faiss_file)
    else:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)  # L2 distance

    # Add new embeddings to the index
    index.add(embeddings)
    print(f"Total vectors in FAISS index: {index.ntotal}")

    # Save the updated FAISS index
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    faiss.write_index(index, faiss_file)

    # Update metadata
    new_metadata = pd.DataFrame([
        {'name': card.get('name', ''), 'oracle_id': card.get('oracle_id', '')} for card in new_cards
    ])
    if existing_metadata is not None:
        updated_metadata = pd.concat([existing_metadata, new_metadata], ignore_index=True)
    else:
        updated_metadata = new_metadata
    updated_metadata.to_csv(metadata_file, index=False)
    print(f"Updated metadata with {len(updated_metadata)} total cards in data/vector_db/card_metadata.csv.")


if __name__ == "__main__":
    build_vector_db()