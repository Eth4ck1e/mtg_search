import requests
import os
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Define paths
HOME_DIR = '/home/eth4ck1e'
PROJECT_DIR = os.path.join(HOME_DIR, 'mtg_search')
BULK_JSONS_DIR = os.path.join(PROJECT_DIR, 'resources', 'bulk_jsons')

# Set up a session with retries for transient errors
session = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=10,  # Increased to 10 for delays of 10s, 20s, 40s
    status_forcelist=[429, 503, 504],
    allowed_methods=['GET']
)
session.mount('https://', HTTPAdapter(max_retries=retries))

# Headers required by Scryfall
headers = {
    'User-Agent': 'MTGSearchApp/1.0 (eth4ck1e@example.com)',
    'Accept': 'application/json;q=0.9,*/*;q=0.8'
}


# Fetch the download URLs for specified bulk items
def get_bulk_download_urls(target_types):
    bulk_data_url = 'https://api.scryfall.com/bulk-data'
    time.sleep(0.5)  # 500ms delay before initial request
    try:
        response = session.get(bulk_data_url, headers=headers, timeout=10)
        if response.status_code == 429:  # Too Many Requests
            retry_after = int(response.headers.get('Retry-After', 10))  # Default to 10s if not specified
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


# Download a JSON file
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


# Main execution
if __name__ == "__main__":
    try:
        # Create the bulk_jsons directory
        os.makedirs(BULK_JSONS_DIR, exist_ok=True)

        # Define the bulk items we want
        target_types = ['oracle_cards', 'default_cards', 'rulings']

        # Fetch download URLs
        download_urls = get_bulk_download_urls(target_types)

        # Download each file
        for bulk_type, info in download_urls.items():
            filepath = os.path.join(BULK_JSONS_DIR, info['filename'])
            download_bulk_file(info['url'], filepath)
            time.sleep(0.5)  # 500ms delay between downloads (2 requests/sec)

        print("All bulk files downloaded successfully.")
    except Exception as e:
        print(f"Error: {e}")