# mtg_search/src/data_processing/compile_initial_training_data.py
import pandas as pd
import json
import os
import re
from src import config

# Dictionary of keyword abilities and their definitions
keyword_definitions = {
    "Flying": "(This creature can't be blocked except by creatures with flying or reach.)",
    "Trample": "(This creature can deal excess combat damage to the player or planeswalker it's attacking.)",
    "Haste": "(This creature can attack and {T} as soon as it comes under your control.)",
    "Vigilance": "(Attacking doesn't cause this creature to tap.)",
    "Reach": "(This creature can block creatures with flying.)",
    "Flash": "(You may cast this spell any time you could cast an instant.)",
    "Menace": "(This creature can't be blocked except by two or more creatures.)",
    "Deathtouch": "(Any amount of damage this deals to a creature is enough to destroy it.)",
    "Lifelink": "(Damage dealt by this creature also causes you to gain that much life.)",
    "Hexproof": "(This creature can't be the target of spells or abilities your opponents control.)",
    "Indestructible": "(This creature can't be destroyed by damage or effects that say 'destroy'.)",
    "Defender": "(This creature can't attack.)",
    "First Strike": "(This creature deals combat damage before creatures without first strike.)",
    "Double Strike": "(This creature deals both first-strike and regular combat damage.)",
    # Add more keywords as needed
}

def compile_data():
    # Load the raw JSON data
    raw_json_path = os.path.join(config.RAW_DATA_DIR, 'oracle-cards.json')
    with open(raw_json_path, 'r') as f:
        json_data = json.load(f)

    # Convert JSON to DataFrame
    df = pd.DataFrame(json_data)
    print(f"Starting number of entries: {len(df)}")

    # Convert all columns to strings to avoid type issues
    for col in df.columns:
        df[col] = df[col].apply(lambda x: ', '.join(str(i) for i in x) if isinstance(x, list) else str(x) if pd.notna(x) else '')

    # Filter out non-card entries (e.g., tokens, planes, dungeons, stickers)
    non_card_types = ['Token', 'Plane', 'Dungeon', 'Stickers']
    non_card_pattern = '|'.join([re.escape(t) for t in non_card_types])
    specific_non_card_types = ['Card // Card']
    df = df[~df['type_line'].str.contains(non_card_pattern, case=False, na=False)]
    df = df[~df['type_line'].isin(specific_non_card_types)]
    print(f"Filtered out non-card entries. Remaining rows: {len(df)}")

    # Keep only English-language cards
    df = df[df['lang'] == 'en']
    print(f"Filtered to English-language cards. Remaining rows: {len(df)}")

    # Filter out entries with no oracle_text (since we're focusing on abilities)
    df = df[df['oracle_text'].str.strip() != '']
    print(f"Filtered out entries with no oracle_text. Remaining rows: {len(df)}")

    # Keep only the relevant columns: oracle_id, keywords, oracle_text
    relevant_columns = ['oracle_id', 'keywords', 'oracle_text']
    df = df[relevant_columns]
    print(f"Kept relevant columns: {relevant_columns}")

    # Clean and format the data for training
    def format_text(row):
        oracle_id = row['oracle_id']
        keywords = row.get('keywords', '')
        effect = row.get('oracle_text', '')

        # Add definitions to keywords
        keywords_with_defs = [f"{kw} {keyword_definitions.get(kw, '')}" for kw in keywords.split(', ') if kw]
        keywords_str = ', '.join(keywords_with_defs) if keywords_with_defs else 'None'

        return f"Oracle ID: {oracle_id} | Keywords: {keywords_str} | Effect: {effect}"

    # Add the text column to the DataFrame
    df['text'] = df.apply(format_text, axis=1)

    # Save the processed data as complete_set.csv
    processed_dir = os.path.join(config.PROCESSED_DATA_DIR)
    os.makedirs(processed_dir, exist_ok=True)
    raw_csv_path = os.path.join(processed_dir, 'complete_set.csv')
    df.to_csv(raw_csv_path, index=False)
    print(f"Saved raw data as CSV to {raw_csv_path}")

if __name__ == "__main__":
    compile_data()