# mtg_search/check_model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import numpy as np

# Define paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Go up one level from tests/
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'initial_model')
BASE_MODEL_NAME = "deepseek-ai/deepseek-llm-7b-base"

def check_model(model_path, model_name="Model"):
    # Load model and tokenizer
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nChecking {model_name}...")
    print(f"Using device: {device}")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None, None

    # Print model details
    print(f"Model type: {model.config.model_type}")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Number of layers: {model.config.num_hidden_layers}")
    print(f"Number of attention heads: {model.config.num_attention_heads}")
    print(f"Vocabulary size: {model.config.vocab_size}")

    # Estimate number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test a small input to check for NaN outputs
    test_input = "Test input for embedding generation"
    inputs = tokenizer(test_input, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        embedding = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
    print(f"Test embedding shape: {embedding.shape}")
    print(f"Test embedding (first 5 values): {embedding[0][:5]}")
    if np.any(np.isnan(embedding)):
        print("Warning: NaN values detected in test embedding")

    return model, tokenizer

if __name__ == "__main__":
    # Check the base model (deepseek-llm-7b-base)
    base_model, base_tokenizer = check_model(BASE_MODEL_NAME, "Base Model (deepseek-llm-7b-base)")

    # Check the fine-tuned model (initial_model)
    fine_tuned_model, fine_tuned_tokenizer = check_model(MODEL_DIR, "Fine-Tuned Model (initial_model)")