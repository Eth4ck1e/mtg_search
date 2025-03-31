# mtg_search/src/vector_db/build_vector_db.py
import pandas as pd
import numpy as np
import faiss
import os
import logging
from transformers import AutoTokenizer, AutoModel
import torch
from src import config
from src.config import MODEL_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# File path configuration
ORACLE_CARDS_PATH = os.path.join(config.PROCESSED_DATA_DIR, "complete_set.csv")
VECTOR_DB_PATH = os.path.join(config.VECTOR_DB_DIR, "vector_db.faiss")
INDEX_DIMENSION = 768  # DistilBERT outputs 768-dimensional embeddings

def load_oracle_cards(file_path):
    """
    Load complete_set.csv into a pandas DataFrame.
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
    Load the fine-tuned DistilBERT model and tokenizer from the given path.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        model.eval()
        logging.info(f"Successfully loaded model and tokenizer from {model_path}.")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Failed to load model or tokenizer from {model_path}: {e}")
        return None, None

def create_vector_database(df, model, tokenizer, vector_db_path, batch_size=10):
    """
    Create a FAISS-based vector database from the DataFrame's text column using the fine-tuned DistilBERT model.
    """
    # Ensure the 'text' column exists
    if 'text' not in df.columns:
        logging.error("'text' column is missing from the dataframe. Cannot proceed.")
        return False

    # Validate and clean the data
    df['text'] = df['text'].astype(str).fillna('')  # Ensure text is string and replace NaN with empty string
    df = df[df['text'].str.strip() != '']  # Remove empty strings
    logging.info(f"After cleaning, {len(df)} cards remain")

    # Set device (try MPS since it worked with SentenceTransformer)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    logging.info(f"Using device: {device}")

    # Test the model and tokenizer with a simple input
    try:
        test_input = "Test input for tokenization"
        test_inputs = tokenizer(test_input, return_tensors="pt", truncation=True, padding=True, max_length=config.MAX_LENGTH)
        test_inputs = {k: v.to(device) for k, v in test_inputs.items()}
        with torch.no_grad():
            test_outputs = model(**test_inputs)
            test_embedding = test_outputs.last_hidden_state[:, 0, :].cpu().numpy()
        logging.info(f"Successfully tested model with simple input. Test embedding shape: {test_embedding.shape}")
    except Exception as e:
        logging.error(f"Error testing model with simple input: {e}")
        return False

    # Log the first batch for debugging
    first_batch_texts = df['text'][0:batch_size].tolist()
    logging.info(f"First batch texts (rows 0 to {batch_size}): {first_batch_texts}")

    # Generate embeddings in batches
    embeddings = []
    for start_idx in range(0, len(df), batch_size):
        end_idx = min(start_idx + batch_size, len(df))
        batch_texts = df['text'][start_idx:end_idx].tolist()
        logging.info(f"Processing batch {start_idx // batch_size + 1}: {start_idx} to {end_idx}")

        # Tokenize the batch
        try:
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=config.MAX_LENGTH
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logging.info(f"Tokenized batch {start_idx // batch_size + 1} with input shape: {inputs['input_ids'].shape}")
        except Exception as e:
            logging.error(f"Error tokenizing batch {start_idx // batch_size + 1}: {e}")
            logging.error(f"Problematic texts: {batch_texts}")
            continue

        # Generate embeddings
        try:
            with torch.no_grad():
                outputs = model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] token embedding
            embeddings.append(batch_embeddings)
            logging.info(f"Generated embeddings for batch {start_idx // batch_size + 1} with shape: {batch_embeddings.shape}")
        except Exception as e:
            logging.error(f"Error generating embeddings for batch {start_idx // batch_size + 1}: {e}")
            continue

    # Combine embeddings
    if not embeddings:
        logging.error("No embeddings generated. Check input data or model inference.")
        return False
    embeddings = np.vstack(embeddings).astype('float32')
    logging.info(f"Generated embeddings with shape: {embeddings.shape}")

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
    except Exception as e:
        logging.error(f"Failed to create or save FAISS index: {e}")
        return False

    # Create card_metadata.csv
    try:
        logging.info("Creating card metadata...")
        metadata = df[['text']].reset_index().rename(columns={'index': 'id'})
        metadata['name'] = metadata['text'].str.extract(r'Name: (.*?)(?: \|)')
        metadata = metadata[['id', 'name', 'text']]  # Reorder columns
        metadata_path = os.path.join(config.VECTOR_DB_DIR, 'card_metadata.csv')
        metadata.to_csv(metadata_path, index=False)
        logging.info(f"Saved card metadata to {metadata_path}.")
        return True
    except Exception as e:
        logging.error(f"Failed to create card metadata: {e}")
        return False

def main():
    # Step 1: Load complete_set.csv
    df = load_oracle_cards(ORACLE_CARDS_PATH)
    if df is None:
        return

    # Step 2: Load the fine-tuned DistilBERT model and tokenizer
    model, tokenizer = load_model(MODEL_DIR)
    if model is None or tokenizer is None:
        return

    # Step 3: Create vector database and metadata
    success = create_vector_database(df, model, tokenizer, VECTOR_DB_PATH, batch_size=10)
    if success:
        logging.info("Vector database and metadata created successfully.")
    else:
        logging.error("Failed to create the vector database or metadata.")

if __name__ == "__main__":
    main()