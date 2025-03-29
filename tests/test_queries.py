# mtg_search/tests/test_queries.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vector_db.query_vector_db import query_vector_db, create_input_text
from transformers import AutoModel, AutoTokenizer
import torch
from src import config  # Use absolute import


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model
