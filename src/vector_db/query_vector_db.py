# mtg_search/src/vector_db/query_vector_db.py
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
import faiss
import numpy as np
import sys
from src import config  # Use absolute import

# Set environment variables before any imports to avoid OpenMP conflicts
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['PYTORCH_MPS_NUM_THREADS'] = '1'
# Temporary workaround to allow duplicate OpenMP runtime (unsafe, for testing)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def create_input_text(text):
    """
    Expand a query text by adding assumed context to align with card embeddings.
    """
    text = text.lower().strip()

    # Rule-based expansion for common keywords
    if "sliver" in text:
        # Assume a Sliver creature with a typical effect
        expanded = f"Name: {text} | Type: Creature — Sliver | Effect: All Sliver creatures"
    elif "flicker" in text:
        # Assume a flicker effect
        expanded = f"Name: {text} | Effect: Exile and return"
    else:
        # Default: use the query as-is, but add minimal context
        expanded = f"Name: {text} | Effect: {text}"

    return expanded


def query_vector_db(query, top_k=10, model=None, tokenizer=None, index=None, metadata=None):
    """
    Query the vector database with a plain-text query and return the top_k matching cards.

    Args:
        query (str): The plain-text query (e.g., "red cards with flicker-like effects").
        top_k (int): Number of top matches to return (default: 10).
        model: Pre-loaded model (optional, for API use).
        tokenizer: Pre-loaded tokenizer (optional, for API use).
        index: Pre-loaded FAISS index (optional, for API use).
        metadata: Pre-loaded metadata DataFrame (optional, for API use).

    Returns:
        List of dicts with card name, oracle_id, distance, and similarity score.
    """
    # Load model and tokenizer if not provided
    if model is None or tokenizer is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")
        model = AutoModel.from_pretrained(config.MODEL_DIR).to(device)
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_DIR)
        model.eval()
        # Debug: Print hidden size
        hidden_size = model.config.hidden_size
        print(f"Model hidden size: {hidden_size}")

    # Load FAISS index and metadata if not provided
    if index is None:
        index = faiss.read_index(os.path.join(config.VECTOR_DB_DIR, 'vector_db.faiss'))
    if metadata is None:
        metadata = pd.read_csv(os.path.join(config.VECTOR_DB_DIR, 'card_metadata.csv'))

    # Generate query embedding
    query_text = create_input_text(query)
    print(f"Expanded query text: {query_text}")  # Debug: print expanded query
    inputs = tokenizer(query_text, padding=True, truncation=True, max_length=config.MAX_LENGTH, return_tensors="pt").to(
        model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        query_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Mean pooling over sequence length
    print(f"Query embedding shape: {query_embedding.shape}")
    print(f"Query embedding (first 5 values): {query_embedding[0][:5]}")  # Debug: print first 5 values

    # Search FAISS index
    distances, indices = index.search(query_embedding, top_k)
    print(f"FAISS distances: {distances[0]}")  # Debug: print distances
    print(f"FAISS indices: {indices[0]}")  # Debug: print indices

    # Map indices to cards
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        card = metadata.iloc[idx]
        results.append({
            'name': card['name'],
            'oracle_id': card['oracle_id'],
            'distance': float(distance),
            'similarity': 1 / (1 + distance)  # Convert distance to similarity score (0-1)
        })
    return results