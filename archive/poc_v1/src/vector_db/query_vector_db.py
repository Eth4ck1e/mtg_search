# mtg_search/src/vector_db/query_vector_db.py
import os

import pandas as pd
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
from src import config

def query_vector_db(query, top_k=5):
    # Load the fine-tuned DistilBERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_DIR)
    model = AutoModel.from_pretrained(config.MODEL_DIR)
    model.eval()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    print(f"Loaded DistilBERT model from {config.MODEL_DIR} on device: {device}")

    # Load the FAISS index
    faiss_index_path = os.path.join(config.VECTOR_DB_DIR, 'vector_db.faiss')
    index = faiss.read_index(faiss_index_path)
    print(f"Loaded FAISS index with {index.ntotal} vectors")

    # Load the card metadata
    metadata_path = os.path.join(config.VECTOR_DB_DIR, 'card_metadata.csv')
    metadata = pd.read_csv(metadata_path)
    print(f"Loaded metadata with {len(metadata)} cards")

    # Embed the query
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=config.MAX_LENGTH)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        query_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype('float32')  # [CLS] token embedding

    # Search the FAISS index
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        card = metadata.iloc[idx]
        results.append({
            'id': card['id'],
            'name': card['name'],
            'text': card['text'],
            'distance': distance
        })
    return results

if __name__ == "__main__":
    query = "Legendary Creature with Landfall"
    results = query_vector_db(query, top_k=5)
    for result in results:
        print(f"Name: {result['name']}, Distance: {result['distance']}")
        print(f"Text: {result['text']}\n")