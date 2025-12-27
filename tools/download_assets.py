import os
import requests
from pathlib import Path
import hashlib
import sys

# Configuration
ASSETS_DIR = Path('examples/resources/textures/hdr')
# TODO: User needs to update this URL once the release is created
BASE_URL = "https://github.com/yoyonel/moderngl-window/releases/download/assets-v1/"

ASSETS = [
    {
        "name": "abandoned_garage_4k.exr",
        # SHA256 should be updated with actual values if verification is strictly needed
        # For now we just check existence
    },
    { "name": "hangar_interior_4k.exr" },
    { "name": "klippad_sunrise_2_4k.exr" },
    { "name": "metro_noord_4k.exr" },
    { "name": "neon_photostudio_4k.exr" },
]

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = int(50 * downloaded / total_size)
                        sys.stdout.write(f"\r[{'=' * percent}{' ' * (50-percent)}] {downloaded//1024//1024}MB / {total_size//1024//1024}MB")
                        sys.stdout.flush()
        print("\nDone.")
        return True
    except Exception as e:
        print(f"\nError downloading {url}: {e}")
        if dest_path.exists():
            dest_path.unlink() # Remove partial file
        return False

def main():
    if not ASSETS_DIR.exists():
        print(f"Creating directory: {ASSETS_DIR}")
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    print("Checking assets...")
    for asset in ASSETS:
        file_path = ASSETS_DIR / asset["name"]
        
        if file_path.exists():
            print(f"[OK] {asset['name']} exists.")
            continue
            
        url = BASE_URL + asset["name"]
        print(f"[MISSING] {asset['name']}")
        if not download_file(url, file_path):
            print("Failed to download asset. Please check your internet connection or the URL.")
            sys.exit(1)

    print("\nAll assets verified.")

if __name__ == '__main__':
    main()
