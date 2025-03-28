# mtg_search/src/data_processing/compile_initial_training_data.py
import pandas as pd
import json
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
INITIAL_SUBSET = os.path.join(PROJECT_ROOT, 'data', 'processed', 'training', 'initial', 'initial_subset.csv')
INITIAL_TRAINING_DATA = os.path.join(PROJECT_ROOT, 'data', 'processed', 'training', 'initial', 'initial_training_data.csv')
RULINGS_DATA = os.path.join(PROJECT_ROOT, 'data', 'processed', 'training', 'rulings', 'rulings_data.csv')

def compile_data():
    # Load JSONs from data/raw/
    with open(os.path.join(RAW_DATA_DIR, 'oracle-cards.json'), 'r') as f:
        oracle_cards = json.load(f)
    with open(os.path.join(RAW_DATA_DIR, 'default-cards.json'), 'r') as f:
        default_cards = json.load(f)
    with open(os.path.join(RAW_DATA_DIR, 'rulings.json'), 'r') as f:
        rulings = json.load(f)

    # Process oracle cards
    oracle_data = []
    for card in oracle_cards:
        oracle_data.append({
            'name': card.get('name', ''),
            'type_line': card.get('type_line', ''),
            'mana_cost': card.get('mana_cost', ''),
            'colors': card.get('colors', []),
            'oracle_text': card.get('oracle_text', ''),
            'keywords': card.get('keywords', [])
        })
    df_oracle = pd.DataFrame(oracle_data)

    # Process default cards
    default_data = []
    for card in default_cards:
        default_data.append({
            'name': card.get('name', ''),
            'type_line': card.get('type_line', ''),
            'mana_cost': card.get('mana_cost', ''),
            'colors': card.get('colors', []),
            'oracle_text': card.get('oracle_text', ''),
            'keywords': card.get('keywords', [])
        })
    df_default = pd.DataFrame(default_data)

    # Process rulings
    rulings_data = []
    for ruling in rulings:
        rulings_data.append({
            'card_name': ruling.get('oracle_id', ''),
            'ruling': ruling.get('comment', '')
        })
    df_rulings = pd.DataFrame(rulings_data)

    # Save to processed/
    os.makedirs(os.path.dirname(INITIAL_SUBSET), exist_ok=True)
    os.makedirs(os.path.dirname(RULINGS_DATA), exist_ok=True)
    df_oracle.to_csv(INITIAL_SUBSET, index=False)
    df_default.to_csv(INITIAL_TRAINING_DATA, index=False)
    df_rulings.to_csv(RULINGS_DATA, index=False)
    print("Training data compiled successfully from data/raw/ to data/processed/.")

if __name__ == "__main__":
    compile_data()