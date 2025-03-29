import pandas as pd
import json
import os
from src import config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def compile_data():
    # Define raw JSON file path
    raw_json_path = os.path.join(config.BULK_JSONS_DIR, 'oracle-cards.json')

    # Check if the raw JSON file exists
    if not os.path.exists(raw_json_path):
        logging.error(f"Raw JSON file not found at {raw_json_path}")
        return

    # Load JSON data with error handling
    try:
        with open(raw_json_path, 'r') as f:
            json_data = json.load(f)
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON data from {raw_json_path}")
        return
    except Exception as e:
        logging.error(f"An error occurred while reading the JSON file: {e}")
        return

    # Convert JSON to DataFrame
    df = pd.DataFrame(json_data)
    logging.info(f"Loaded JSON data into DataFrame with {len(df)} rows.")

    # Save the raw data as a CSV for future use
    raw_csv_path = os.path.join(config.TRAINING_DATA_DIR, 'oracle_cards.csv')
    try:
        os.makedirs(config.TRAINING_DATA_DIR, exist_ok=True)
        df.to_csv(raw_csv_path, index=False)
        logging.info(f"Saved raw data as CSV to {raw_csv_path}")
    except Exception as e:
        logging.error(f"Failed to save CSV to {raw_csv_path}: {e}")
        return

    # Helper function to format a row into text
    def format_text(row):
        # Clean the name by removing double '//'
        name = row.get('name', '').replace(' // ', ' / ') if pd.notna(row.get('name', '')) else ''

        # Handle missing or NaN values
        type_line = row.get('type_line', '') if pd.notna(row.get('type_line', '')) else ''
        mana_cost = row.get('mana_cost', '') if pd.notna(row.get('mana_cost', '')) else ''
        oracle_text = row.get('oracle_text', '') if pd.notna(row.get('oracle_text', '')) else ''

        # Handle colors and keywords, ensuring they are properly joined if lists
        colors = row.get('colors', [])
        colors = ', '.join(colors) if isinstance(colors, list) and len(colors) > 0 else ''

        keywords = row.get('keywords', [])
        keywords = ', '.join(keywords) if isinstance(keywords, list) and len(keywords) > 0 else ''

        # Format the text field
        text = (
            f"Name: {name} | "
            f"Type: {type_line} | "
            f"Cost: {mana_cost} | "
            f"Colors: {colors} | "
            f"Effect: {oracle_text} | "
            f"Keywords: {keywords}"
        )
        return text

    # Apply the format_text function to the entire dataset
    try:
        df['text'] = df.apply(format_text, axis=1)
    except Exception as e:
        logging.error(f"Failed to format text for the complete dataset: {e}")
        return

    # Save the processed complete data
    complete_set_path = os.path.join(config.TRAINING_DATA_DIR, 'complete_set.csv')
    try:
        df[['text']].to_csv(complete_set_path, index=False)
        logging.info(f"Complete dataset saved to {complete_set_path}")
    except Exception as e:
        logging.error(f"Failed to save complete dataset to {complete_set_path}: {e}")
        return

    # Select a subset for initial training (e.g., first 1000 rows)
    subset_df = df.head(1000)
    logging.info(f"Selected subset for training with {len(subset_df)} rows.")

    # Save the initial_subset CSV
    processed_path = os.path.join(config.TRAINING_DATA_DIR, 'initial_subset.csv')
    try:
        subset_df[['text']].to_csv(processed_path, index=False)
        logging.info(f"Processed initial subset saved to {processed_path}")
    except Exception as e:
        logging.error(f"Failed to save processed subset data to {processed_path}: {e}")
        return


if __name__ == "__main__":
    compile_data()
