# mtg_search/src/config.py
import os

# Model configuration
MODEL_NAME = "distilbert-base-uncased"  # Base model to use for training and inference
MODEL_TYPE = "bert"  # Model type ("bert" for DistilBERT, "causal_lm" for DeepSeek/Gemma)

# Data and processing parameters
MAX_LENGTH = 64  # Maximum sequence length for tokenization

# Directory paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw', 'bulk_jsons')  # For raw data (e.g., oracle-cards.json, complete_set.csv)
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed', 'training', 'initial')  # For processed data (e.g., initial_subset.csv)
VECTOR_DB_DIR = os.path.join(DATA_DIR, 'vector_db')  # For vector database (e.g., card_vectors.faiss, card_metadata.csv)
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'initial_model')  # For trained model
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'models', 'checkpoints', 'model_output')  # For training checkpoints

# Training parameters
DATALOADER_NUM_WORKERS = 0  # Number of workers for DataLoader (0 to avoid multiprocessing issues on macOS)