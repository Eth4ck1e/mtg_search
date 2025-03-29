# mtg_search/src/data_processing/download_bulk_data.py
import requests
import time
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from src import config  # Use absolute import

def download_all():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=10, status_forcelist=[429, 503, 504], allowed_methods=['GET'])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    headers = {'User-Agent': 'MTGSearchApp/1.0 (eth4ck1e@example.com)', 'Accept': 'application/json;q=0.9,*/*;q=0.8'}

    bulk_data_url = 'https://api.scryfall.com/bulk-data'
    time.sleep(0.5)
    response = session.get(bulk_data_url, headers=headers, timeout=10)
    if response.status_code == 429:
        retry_after = int(response.headers.get('Retry-After', 10))
        print(f"Rate limit hit. Waiting {retry_after} seconds...")
        time.sleep(retry_after)
        response = session.get(bulk_data_url, headers=headers, timeout=10)
    response.raise_for_status()

    bulk_data = response.json()
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)

    for entry in bulk_data['data']:
        if entry['type'] in ['oracle_cards', 'default_cards', 'rulings']:
            download_url = entry['download_uri']
            file_name = f"{entry['type'].replace('_', '-')}.json"
            file_path = os.path.join(config.RAW_DATA_DIR, file_name)
            print(f"Downloading {file_name} to {file_path}...")
            time.sleep(0.5)
            response = session.get(download_url, headers=headers, timeout=30)
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 10))
                print(f"Rate limit hit. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                response = session.get(download_url, headers=headers, timeout=30)
            response.raise_for_status()
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {file_name}")

if __name__ == "__main__":
    download_all()