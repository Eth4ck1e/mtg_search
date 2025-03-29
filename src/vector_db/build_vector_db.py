import pandas as pd
import os
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

from src.config import TRAINING_DATA_DIR, MODEL_DIR, VECTOR_DB_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# File path configuration
ORACLE_CARDS_PATH = os.path.join(TRAINING_DATA_DIR, "complete_set.csv")
VECTOR_DB_PATH = os.path.join(VECTOR_DB_DIR, "vector_db.faiss")
INDEX_DIMENSION = 768  # This assumes the model outputs 768-dimensional embeddings


def load_oracle_cards(file_path):
    """
    Load oracle_cards.csv into a pandas DataFrame.
    """
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return None

    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded {len(df)} rows from {file_path}.")
        return df
    except Exception as e:
        logging.error(f"Failed to load {file_path}: {e}")
        return None


def load_model(model_path):
    """
    Load a SentenceTransformer model from the given path.
    """
    try:
        model = SentenceTransformer(model_path)
        logging.info(f"Successfully loaded model from {model_path}.")
        return model
    except Exception as e:
        logging.error(f"Failed to load model from {model_path}: {e}")
        return None


def create_vector_database(df, model, vector_db_path):
    """
    Create a FAISS-based vector database from the DataFrame's text column.
    """
    # Ensure the 'text' column exists
    if 'text' not in df.columns:
        logging.error("'text' column is missing from the dataframe. Cannot proceed.")
        return False

    # Compute vectors using the model
    try:
        logging.info("Computing embeddings for the text data...")
        text_data = df['text'].fillna("").tolist()  # Handle NaN values
        embeddings = model.encode(text_data, convert_to_numpy=True, show_progress_bar=True)
        logging.info(f"Computed embeddings with shape: {embeddings.shape}.")
    except Exception as e:
        logging.error(f"Failed to compute embeddings: {e}")
        return False

    # Initialize FAISS index
    try:
        logging.info("Initializing FAISS index...")
        index = faiss.IndexFlatL2(INDEX_DIMENSION)  # L2 distance
        assert index.is_trained

        # Add embeddings to the index
        logging.info("Adding embeddings to the FAISS index...")
        index.add(embeddings)
        logging.info(f"Added {index.ntotal} embeddings to the FAISS index.")

        # Save the FAISS index to file
        logging.info(f"Saving FAISS index to {vector_db_path}...")
        faiss.write_index(index, vector_db_path)
        logging.info("FAISS index saved successfully.")
        return True
    except Exception as e:
        logging.error(f"Failed to create or save FAISS index: {e}")
        return False


def main():
    # Step 1: Load oracle_cards.csv
    df = load_oracle_cards(ORACLE_CARDS_PATH)
    if df is None:
        return

    # Step 2: Load the local model
    model = load_model(MODEL_DIR)
    if model is None:
        return

    # Step 3: Create vector database
    success = create_vector_database(df, model, VECTOR_DB_PATH)
    if success:
        logging.info("Vector database created successfully.")
    else:
        logging.error("Failed to create the vector database.")


if __name__ == "__main__":
    main()
