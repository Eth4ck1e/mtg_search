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
BULK_JSONS_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'bulk_jsons')
TRAINING_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'training', 'initial')
VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, 'data', 'vector_db')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'initial_model')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'models', 'checkpoints', 'model_output')
INITIAL_MODEL_DIR = MODEL_DIR  # Define INITIAL_MODEL_DIR for compatibility

# Training parameters
DATALOADER_NUM_WORKERS = 0  # Number of workers for DataLoader (0 to avoid multiprocessing issues on macOS)