# mtg_search/tests/test_vector_db.py
import torch
import faiss
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
import pandas as pd
import os
import re
from src import config

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_DIR)
    model = AutoModelForMaskedLM.from_pretrained(config.MODEL_DIR)  # Use AutoModelForMaskedLM for MLM tasks
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device

def load_vector_db():
    index_path = os.path.join(config.VECTOR_DB_DIR, "vector_db.faiss")
    metadata_path = os.path.join(config.VECTOR_DB_DIR, "card_metadata.csv")
    index = faiss.read_index(index_path)
    metadata = pd.read_csv(metadata_path)
    return index, metadata

def get_embedding(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=config.MAX_LENGTH)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        # Use the [CLS] token embedding (first token) as the embedding for the text
        # Since we're using AutoModelForMaskedLM, we need to access the hidden states via the base model
        hidden_states = outputs.logits  # This is not correct for embedding; we need to use a different approach
        # Instead, we'll run the model in a way that gives us hidden states
        base_model = model.base_model  # Access the underlying DistilBERT model
        with torch.no_grad():
            outputs = base_model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] token embedding
    return embedding

def generate_structured_query(query, model, tokenizer, device):
    """
    Use the model to generate a structured query format from the raw query by filling in key fields.
    Example: "sliver" -> "Name: sliver | Type: Creature — Sliver | Cost: | CMC: | Colors: | Color Identity: | Effect: | Power: | Toughness: | Keywords: | Loyalty: "
    """
    query = query.lower().strip()

    # Initialize a fill-mask pipeline using the model and tokenizer
    fill_mask = pipeline(
        "fill-mask",
        model=model,
        tokenizer=tokenizer,
        device=0 if device.type == "mps" else -1  # Use MPS if available, otherwise CPU
    )

    # Initialize the structured fields
    structured_fields = {
        "Name": "",
        "Type": "",
        "Cost": "",
        "CMC": "",
        "Colors": "",
        "Color Identity": "",
        "Effect": "",
        "Power": "",
        "Toughness": "",
        "Keywords": "",
        "Loyalty": ""
    }

    # Define common patterns for fallback parsing
    creature_types = ["sliver", "dragon", "elf", "goblin", "human", "zombie", "vampire", "angel", "demon", "tyranid"]
    colors = {
        "white": "W", "blue": "U", "black": "B", "red": "R", "green": "G",
        "colorless": "", "multicolor": "W, U, B, R, G"
    }
    keywords = ["flying", "trample", "hexproof", "vigilance", "reach", "flash", "menace", "deathtouch"]
    effect_phrases = {
        "enter the battlefield": "enters the battlefield",
        "enters the battlefield": "enters the battlefield",
        "when this creature enters": "enters the battlefield",
        "flicker-like": "exile",
        "flicker": "exile",
        "blink": "exile"
    }

    # Split the query into words
    words = query.split()

    # Step 1: Use the model to predict key fields
    # Prompt for Type
    type_prompt = f"Name: {query} | Type: [MASK] | Colors: | Keywords: | Effect: "
    try:
        type_predictions = fill_mask(type_prompt, top_k=1)
        predicted_type = type_predictions[0]["token_str"].strip()
        if predicted_type.lower().startswith("creature"):
            structured_fields["Type"] = predicted_type
        else:
            structured_fields["Type"] = "Creature" if "creature" in query else ""
    except Exception as e:
        print(f"Error predicting Type: {e}")
        structured_fields["Type"] = "Creature" if "creature" in query else ""

    # Prompt for Colors
    colors_prompt = f"Name: {query} | Type: {structured_fields['Type']} | Colors: [MASK] | Keywords: | Effect: "
    try:
        colors_predictions = fill_mask(colors_prompt, top_k=1)
        predicted_colors = colors_predictions[0]["token_str"].strip().upper()
        if predicted_colors in ["W", "U", "B", "R", "G", ""]:
            structured_fields["Colors"] = predicted_colors
            structured_fields["Color Identity"] = predicted_colors
        else:
            structured_fields["Colors"] = ""
            structured_fields["Color Identity"] = ""
    except Exception as e:
        print(f"Error predicting Colors: {e}")
        structured_fields["Colors"] = ""
        structured_fields["Color Identity"] = ""

    # Prompt for Keywords
    keywords_prompt = f"Name: {query} | Type: {structured_fields['Type']} | Colors: {structured_fields['Colors']} | Keywords: [MASK] | Effect: "
    try:
        keywords_predictions = fill_mask(keywords_prompt, top_k=1)
        predicted_keywords = keywords_predictions[0]["token_str"].strip().lower()
        if predicted_keywords in keywords:
            structured_fields["Keywords"] = predicted_keywords.capitalize()
        else:
            structured_fields["Keywords"] = ""
    except Exception as e:
        print(f"Error predicting Keywords: {e}")
        structured_fields["Keywords"] = ""

    # Prompt for Effect
    effect_prompt = f"Name: {query} | Type: {structured_fields['Type']} | Colors: {structured_fields['Colors']} | Keywords: {structured_fields['Keywords']} | Effect: [MASK]"
    try:
        effect_predictions = fill_mask(effect_prompt, top_k=1)
        predicted_effect = effect_predictions[0]["token_str"].strip().lower()
        for phrase, mapped_effect in effect_phrases.items():
            if phrase in predicted_effect:
                structured_fields["Effect"] = mapped_effect
                break
        else:
            structured_fields["Effect"] = ""
    except Exception as e:
        print(f"Error predicting Effect: {e}")
        structured_fields["Effect"] = ""

    # Step 2: Fallback parsing for fields the model couldn't predict
    for word in words:
        # Check for creature types (e.g., "sliver", "dragon")
        if word in creature_types and not structured_fields["Name"]:
            structured_fields["Name"] = word
            if not structured_fields["Type"]:
                structured_fields["Type"] = f"Creature — {word.capitalize()}"

        # Check for colors (e.g., "red", "blue", "colorless")
        if word in colors and not structured_fields["Colors"]:
            structured_fields["Colors"] = colors[word]
            structured_fields["Color Identity"] = colors[word]

        # Check for keywords (e.g., "flying", "trample")
        if word in keywords and not structured_fields["Keywords"]:
            structured_fields["Keywords"] = word.capitalize()

    # Check for effect phrases (e.g., "enter the battlefield", "flicker-like")
    for phrase, mapped_effect in effect_phrases.items():
        if phrase in query and not structured_fields["Effect"]:
            structured_fields["Effect"] = mapped_effect
            break

    # If the query contains "creature", set the Type to Creature if not already set
    if "creature" in query and not structured_fields["Type"]:
        structured_fields["Type"] = "Creature"

    # If the query contains "cards" but no specific type, leave Type empty
    if "cards" in query and not structured_fields["Type"]:
        structured_fields["Type"] = ""

    # If no specific name is identified, leave Name empty for general queries
    if not structured_fields["Name"]:
        first_word = words[0] if words else ""
        if first_word in creature_types:
            structured_fields["Name"] = first_word
        else:
            structured_fields["Name"] = ""

    # Construct the final structured query
    structured_query = (
        f"Name: {structured_fields['Name']} | "
        f"Type: {structured_fields['Type']} | "
        f"Cost: {structured_fields['Cost']} | "
        f"CMC: {structured_fields['CMC']} | "
        f"Colors: {structured_fields['Colors']} | "
        f"Color Identity: {structured_fields['Color Identity']} | "
        f"Effect: {structured_fields['Effect']} | "
        f"Power: {structured_fields['Power']} | "
        f"Toughness: {structured_fields['Toughness']} | "
        f"Keywords: {structured_fields['Keywords']} | "
        f"Loyalty: {structured_fields['Loyalty']}"
    )
    return structured_query

def search(query, model, tokenizer, device, index, metadata, top_k=5):
    # Generate the structured query using the model
    formatted_query = generate_structured_query(query, model, tokenizer, device)
    print(f"Formatted query: {formatted_query}")

    # Generate embedding for the formatted query
    query_embedding = get_embedding(formatted_query, model, tokenizer, device)

    # Search the FAISS index
    distances, indices = index.search(query_embedding, top_k)

    # Retrieve the top results
    results = []
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        card_info = metadata.iloc[idx]
        result = {
            "name": card_info['name'],
            "text": card_info['text'],
            "distance": distance
        }
        results.append(result)
    return results

def main():
    # Load the model, tokenizer, and vector database
    model, tokenizer, device = load_model_and_tokenizer()
    print(f"Loaded DistilBERT model from {config.MODEL_DIR} on device: {device}")

    index, metadata = load_vector_db()
    print(f"Loaded FAISS index with {index.ntotal} vectors")
    print(f"Loaded metadata with {len(metadata)} cards")

    while True:
        query = input("\nQuery: ").strip()
        if query.lower() in ['exit', 'quit']:
            print("Exiting...")
            break

        # Search for the query
        results = search(query, model, tokenizer, device, index, metadata)

        # Print the results
        print(f"\nTop 5 results for query: '{query}'")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Name: {result['name']}, Distance: {result['distance']:.4f}")
            print(f"Text: {result['text']}")

if __name__ == "__main__":
    main()