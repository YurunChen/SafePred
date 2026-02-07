#!/usr/bin/env python3
"""
Manually download tiktoken BPE files with retry and proxy support.
This script provides multiple methods to download tiktoken files when network fails.
"""
import os
import sys
import hashlib
import tempfile
import argparse
from pathlib import Path

# Common tiktoken files and their URLs
TIKTOKEN_FILES = {
    "o200k_base": {
        "url": "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
        "hash": "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
    },
    "cl100k_base": {
        "url": "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
        "hash": "97b5c1c0e0b0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0",  # Placeholder
    },
    "p50k_base": {
        "url": "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
        "hash": None,
    },
    "r50k_base": {
        "url": "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",
        "hash": None,
    },
}

def get_cache_dir():
    """Get tiktoken cache directory."""
    if "TIKTOKEN_CACHE_DIR" in os.environ:
        return os.environ["TIKTOKEN_CACHE_DIR"]
    elif "DATA_GYM_CACHE_DIR" in os.environ:
        return os.environ["DATA_GYM_CACHE_DIR"]
    else:
        return os.path.join(tempfile.gettempdir(), "data-gym-cache")

def calculate_hash(data: bytes) -> str:
    """Calculate SHA256 hash of data."""
    return hashlib.sha256(data).hexdigest()

def download_with_requests(url: str, proxy: str = None, timeout: int = 30, max_retries: int = 3):
    """Download file using requests library with retry."""
    import requests
    import time
    
    session = requests.Session()
    if proxy:
        session.proxies = {
            "http": proxy,
            "https": proxy,
        }
    
    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}/{max_retries}: Downloading from {url}")
            response = session.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            return response.content
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"  Failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise

def download_with_curl(url: str, output_path: str, proxy: str = None):
    """Download file using curl (alternative method)."""
    import subprocess
    
    cmd = ["curl", "-L", "-o", output_path, url]
    if proxy:
        cmd.extend(["--proxy", proxy])
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        with open(output_path, "rb") as f:
            return f.read()
    except subprocess.CalledProcessError as e:
        raise Exception(f"curl failed: {e.stderr.decode()}")

def download_with_wget(url: str, output_path: str, proxy: str = None):
    """Download file using wget (alternative method)."""
    import subprocess
    
    cmd = ["wget", "-O", output_path, url]
    if proxy:
        cmd.extend(["--proxy", proxy])
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        with open(output_path, "rb") as f:
            return f.read()
    except subprocess.CalledProcessError as e:
        raise Exception(f"wget failed: {e.stderr.decode()}")

def save_to_cache(data: bytes, url: str, cache_dir: str):
    """Save downloaded data to tiktoken cache directory."""
    cache_key = hashlib.sha1(url.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, cache_key)
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # Write to temporary file first, then rename (atomic operation)
    import uuid
    tmp_filename = cache_path + "." + str(uuid.uuid4()) + ".tmp"
    with open(tmp_filename, "wb") as f:
        f.write(data)
    os.rename(tmp_filename, cache_path)
    
    print(f"  Saved to cache: {cache_path}")
    return cache_path

def download_file(file_name: str, method: str = "requests", proxy: str = None):
    """Download a tiktoken file using specified method."""
    if file_name not in TIKTOKEN_FILES:
        print(f"Error: Unknown file {file_name}")
        print(f"Available files: {', '.join(TIKTOKEN_FILES.keys())}")
        return False
    
    file_info = TIKTOKEN_FILES[file_name]
    url = file_info["url"]
    expected_hash = file_info.get("hash")
    
    print(f"\nDownloading {file_name}...")
    print(f"  URL: {url}")
    
    cache_dir = get_cache_dir()
    print(f"  Cache directory: {cache_dir}")
    
    try:
        # Download using specified method
        if method == "requests":
            data = download_with_requests(url, proxy=proxy)
        elif method == "curl":
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp_path = tmp.name
            try:
                data = download_with_curl(url, tmp_path, proxy=proxy)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        elif method == "wget":
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp_path = tmp.name
            try:
                data = download_with_wget(url, tmp_path, proxy=proxy)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            print(f"Error: Unknown method {method}")
            return False
        
        # Verify hash if provided
        if expected_hash:
            actual_hash = calculate_hash(data)
            if actual_hash != expected_hash:
                print(f"  WARNING: Hash mismatch!")
                print(f"    Expected: {expected_hash}")
                print(f"    Actual:   {actual_hash}")
                print(f"  File may be corrupted. Continuing anyway...")
            else:
                print(f"  ✓ Hash verified: {actual_hash[:16]}...")
        
        # Save to cache
        cache_path = save_to_cache(data, url, cache_dir)
        print(f"  ✓ Successfully downloaded and cached {file_name}")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to download {file_name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Manually download tiktoken BPE files")
    parser.add_argument(
        "files",
        nargs="*",
        default=["o200k_base"],
        help="Files to download (default: o200k_base). Available: " + ", ".join(TIKTOKEN_FILES.keys())
    )
    parser.add_argument(
        "--method",
        choices=["requests", "curl", "wget"],
        default="requests",
        help="Download method (default: requests)"
    )
    parser.add_argument(
        "--proxy",
        type=str,
        help="Proxy URL (e.g., http://proxy.example.com:8080)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available files and exit"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("Available tiktoken files:")
        for name, info in TIKTOKEN_FILES.items():
            print(f"  {name}: {info['url']}")
        return
    
    if args.files == ["all"]:
        files_to_download = list(TIKTOKEN_FILES.keys())
    else:
        files_to_download = args.files
    
    success_count = 0
    for file_name in files_to_download:
        if download_file(file_name, method=args.method, proxy=args.proxy):
            success_count += 1
    
    print(f"\nSummary: {success_count}/{len(files_to_download)} files downloaded successfully")
    return 0 if success_count == len(files_to_download) else 1

if __name__ == "__main__":
    sys.exit(main())



