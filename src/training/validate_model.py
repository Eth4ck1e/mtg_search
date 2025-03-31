# mtg_search/src/training/validate_model.py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from src import config


def validate_model(model_path):
    """
    Validate the fine-tuned DistilBERT model by performing inference and checking for issues.
    """
    print(f"Validating the model at {model_path}...")

    # Load the model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForMaskedLM.from_pretrained(model_path)
        model.eval()
        print("Successfully loaded model and tokenizer.")
    except Exception as e:
        print(f"Failed to load model or tokenizer: {e}")
        return False

    # Set device (use MPS if available, as it worked in build_vector_db.py)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Test 1: Simple inference with a known-good input
    try:
        test_input = "Name: Test Card | Type: Creature | Cost: {3}{G} | Colors: G | Effect: This is a test."
        inputs = tokenizer(test_input, return_tensors="pt", truncation=True, padding=True, max_length=config.MAX_LENGTH)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            # Extract [CLS] token embedding for consistency with build_vector_db.py
            embedding = outputs.logits[:, 0, :].cpu().numpy()
        has_nan = torch.isnan(logits).any().item()
        has_inf = torch.isinf(logits).any().item()
        print("\nTest 1: Simple inference")
        print(f"Logits shape: {logits.shape}")
        print(f"Sample logits (first token, first 5 values): {logits[0, 0, :5]}")
        print(f"Logits contain nan: {has_nan}")
        print(f"Logits contain inf: {has_inf}")
        print(f"[CLS] embedding shape: {embedding.shape}")
    except Exception as e:
        print(f"Test 1 failed: {e}")
        return False

    # Test 2: Check model weights for nan or inf
    try:
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"Test 2 failed: Found nan or inf in weights of layer {name}")
                return False
        print("Test 2: Model weights are finite (no nan or inf)")
    except Exception as e:
        print(f"Test 2 failed: {e}")
        return False

    # Test 3: Predict a masked token
    try:
        test_input = "Name: Test Card | Type: Creature | Cost: {3}{G} | Colors: [MASK] | Effect: This is a test."
        inputs = tokenizer(test_input, return_tensors="pt", truncation=True, padding=True, max_length=config.MAX_LENGTH)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
        predicted_token_id = logits[0, mask_token_index, :].argmax(dim=-1)
        predicted_token = tokenizer.decode(predicted_token_id)
        print("\nTest 3: Masked token prediction")
        print(f"Input with mask: {test_input}")
        print(f"Predicted token for [MASK]: {predicted_token}")
    except Exception as e:
        print(f"Test 3 failed: {e}")
        return False

    print("\nAll validation tests passed. Model appears to be valid.")
    return True


if __name__ == "__main__":
    model_path = config.MODEL_DIR
    success = validate_model(model_path)
    if not success:
        print("Model validation failed. Please check the logs for details.")
    else:
        print("Model validation completed successfully.")