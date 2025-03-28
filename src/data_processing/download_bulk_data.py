# mtg_search/src/data_processing/download_bulk_data.py
import requests
import os
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')

session = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=10,
    status_forcelist=[429, 503, 504],
    allowed_methods=['GET']
)
session.mount('https://', HTTPAdapter(max_retries=retries))

headers = {
    'User-Agent': 'MTGSearchApp/1.0 (eth4ck1e@example.com)',
    'Accept': 'application/json;q=0.9,*/*;q=0.8'
}

def get_bulk_download_urls(target_types):
    bulk_data_url = 'https://api.scryfall.com/bulk-data'
    time.sleep(0.5)
    try:
        response = session.get(bulk_data_url, headers=headers, timeout=10)
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 10))
            print(f"Rate limit hit. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
            response = session.get(bulk_data_url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to fetch bulk data: {response.status_code} - {response.text}") from e

    bulk_data = response.json()
    download_urls = {}
    for entry in bulk_data['data']:
        if entry['type'] in target_types:
            download_urls[entry['type']] = {
                'url': entry['download_uri'],
                'filename': f"{entry['type'].replace('_', '-')}.json"
            }
    missing_types = set(target_types) - set(download_urls.keys())
    if missing_types:
        raise Exception(f"Bulk types not found: {missing_types}")
    return download_urls

def download_bulk_file(download_url, filepath):
    print(f"Downloading {filepath} from {download_url}...")
    try:
        response = session.get(download_url, headers=headers, timeout=30)
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 10))
            print(f"Rate limit hit. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
            response = session.get(download_url, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Download failed: {response.status_code} - {response.text}") from e

    with open(filepath, 'wb') as f:
        f.write(response.content)
    print(f"Saved to {filepath}")

def download_all():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    target_types = ['oracle_cards']  # Only oracle_cards for unique cards
    download_urls = get_bulk_download_urls(target_types)
    for bulk_type, info in download_urls.items():
        filepath = os.path.join(RAW_DATA_DIR, info['filename'])
        download_bulk_file(info['url'], filepath)
        time.sleep(0.5)
    print("Oracle cards downloaded successfully to data/raw/.")

if __name__ == "__main__":
    try:
        download_all()
    except Exception as e:
        print(f"Error: {e}")