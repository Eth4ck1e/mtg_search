# mtg_search/src/vector_db/query_vector_db.py
import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss
import numpy as np
import sys

# Set environment variables before any imports to avoid OpenMP conflicts
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['PYTORCH_MPS_NUM_THREADS'] = '1'
# Temporary workaround to allow duplicate OpenMP runtime (unsafe, for testing)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Adjust sys.path to find src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, 'data', 'vector_db')  # Both files here
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'initial_model')

MODEL_NAME = "deepseek-ai/deepseek-llm-7b-base"
MAX_LENGTH = 64


def create_input_text(text):
    return text  # For queries, use the raw text directly


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
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float16).to(device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model.eval()

    # Load FAISS index and metadata if not provided
    if index is None:
        index = faiss.read_index(os.path.join(VECTOR_DB_DIR, 'card_vectors.faiss'))
    if metadata is None:
        metadata = pd.read_csv(os.path.join(VECTOR_DB_DIR, 'card_metadata.csv'))

    # Generate query embedding
    inputs = tokenizer(query, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(
        model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        query_embedding = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()  # Mean pooling
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