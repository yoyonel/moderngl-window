import os
import json
import argparse
from pathlib import Path
import sys
import fnmatch
import urllib.request
import urllib.error

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}")
    try:
        # Open the URL
        with urllib.request.urlopen(url) as response:
            total_size = int(response.info().get('Content-Length', 0))
            block_size = 8192
            downloaded = 0
            
            with open(dest_path, 'wb') as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    
                    f.write(buffer)
                    downloaded += len(buffer)
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

def get_json(url):
    try:
        # GitHub API requires a User-Agent
        req = urllib.request.Request(url, headers={'User-Agent': 'Moderngl-Window-Asset-Downloader'})
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        raise Exception(f"Failed to fetch JSON from {url}: {e}")

def process_asset_group(group):
    name = group.get("name", "Unknown Group")
    repo = group.get("github_repo")
    tag = group.get("release_tag")
    dest_dir = Path(group.get("dest_dir", "."))
    pattern = group.get("file_pattern", "*")

    print(f"\n--- Processing Asset Group: {name} ---")
    if not repo or not tag:
        print("Error: github_repo and release_tag are required.")
        return

    if not dest_dir.exists():
        print(f"Creating directory: {dest_dir}")
        dest_dir.mkdir(parents=True, exist_ok=True)

    # Fetch release info via GitHub API
    api_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    print(f"Fetching asset list from GitHub: {api_url}")
    
    try:
        release_data = get_json(api_url)
    except Exception as e:
        print(f"Error fetching release info for {repo} @ {tag}: {e}")
        print("Note: If this is a private repo or you hit rate limits, you might need an API token.")
        return

    assets = release_data.get("assets", [])
    if not assets:
        print(f"No assets found in release {tag} of {repo}.")
        return

    for asset_meta in assets:
        file_name = asset_meta["name"]
        
        # Filter by pattern
        if not fnmatch.fnmatch(file_name, pattern):
            continue

        download_url = asset_meta["browser_download_url"]
        dest_path = dest_dir / file_name

        if dest_path.exists():
            local_size = dest_path.stat().st_size
            remote_size = asset_meta["size"]
            if local_size == remote_size:
                print(f"[OK] {file_name} already exists and size matches.")
                continue
            else:
                print(f"[UPDATE] {file_name} size mismatch ({local_size} vs {remote_size}). Re-downloading...")

        print(f"[MISSING/STALE] {file_name}")
        if not download_file(download_url, dest_path):
            print(f"Failed to download {file_name}.")
            continue

def main():
    parser = argparse.ArgumentParser(description="Dynamic Asset Downloader for Moderngl-window")
    parser.add_argument("--config", type=str, default="examples/resources/assets.json",
                        help="Path to the assets configuration file (JSON)")
    
    args = parser.parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        # Try relative to script location if not found
        script_dir = Path(__file__).parent.parent
        config_path = script_dir / args.config

    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)

    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)
    except Exception as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)

    asset_groups = config_data.get("assets", [])
    if not asset_groups:
        print("No asset groups found in configuration.")
        return

    for group in asset_groups:
        process_asset_group(group)

    print("\nAll tasks completed.")

if __name__ == '__main__':
    main()
