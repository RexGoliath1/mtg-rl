#!/usr/bin/env python3
"""
17lands Data Downloader

Downloads draft data from 17lands.com S3 bucket for training.

Usage:
    python scripts/download_17lands.py --sets FDN DSK BLB MKM LCI
    python scripts/download_17lands.py --all-recent
    python scripts/download_17lands.py --list
"""

import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import requests

# S3 bucket base URL
S3_BASE = "https://17lands-public.s3.amazonaws.com/analysis_data"

# Data types available
DATA_TYPES = ["draft", "game", "replay"]

# Event types available
EVENT_TYPES = ["PremierDraft", "TradDraft", "Sealed", "TradSealed", "QuickDraft"]

# Recent sets with good draft data availability
RECENT_SETS = [
    {"code": "FDN", "name": "Foundations", "release": "2024-11"},
    {"code": "DSK", "name": "Duskmourn", "release": "2024-09"},
    {"code": "BLB", "name": "Bloomburrow", "release": "2024-08"},
    {"code": "MH3", "name": "Modern Horizons 3", "release": "2024-06"},
    {"code": "OTJ", "name": "Outlaws of Thunder Junction", "release": "2024-04"},
    {"code": "MKM", "name": "Murders at Karlov Manor", "release": "2024-02"},
    {"code": "LCI", "name": "Lost Caverns of Ixalan", "release": "2023-11"},
    {"code": "WOE", "name": "Wilds of Eldraine", "release": "2023-09"},
    {"code": "LTR", "name": "Lord of the Rings", "release": "2023-06"},
    {"code": "MOM", "name": "March of the Machine", "release": "2023-04"},
    {"code": "ONE", "name": "Phyrexia: All Will Be One", "release": "2023-02"},
    {"code": "BRO", "name": "The Brothers' War", "release": "2022-11"},
    {"code": "DMU", "name": "Dominaria United", "release": "2022-09"},
    {"code": "SNC", "name": "Streets of New Capenna", "release": "2022-04"},
    {"code": "NEO", "name": "Kamigawa: Neon Dynasty", "release": "2022-02"},
]

DATA_DIR = Path(__file__).parent.parent / "data" / "17lands"


def get_download_url(set_code: str, data_type: str = "draft", event_type: str = "PremierDraft") -> str:
    """Generate the S3 download URL for a dataset."""
    # URL pattern: {S3_BASE}/{data_type}_data/{data_type}_data_public.{SET}.{EventType}.csv.gz
    return f"{S3_BASE}/{data_type}_data/{data_type}_data_public.{set_code}.{event_type}.csv.gz"


def check_url_exists(url: str) -> Tuple[bool, Optional[int]]:
    """Check if a URL exists and return its size."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
        if response.status_code == 200:
            size = int(response.headers.get('content-length', 0))
            return True, size
        return False, None
    except Exception:
        return False, None


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress display."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, stream=True, timeout=300)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        start_time = time.time()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Progress display
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        elapsed = time.time() - start_time
                        speed = downloaded / elapsed if elapsed > 0 else 0
                        eta = (total_size - downloaded) / speed if speed > 0 else 0

                        print(f"\r  {pct:5.1f}% | {format_size(downloaded)}/{format_size(total_size)} | "
                              f"{format_size(speed)}/s | ETA: {eta:.0f}s", end='', flush=True)

        print()  # Newline after progress
        return True

    except Exception as e:
        print(f"\n  ERROR: {e}")
        return False


def list_available_sets():
    """List all available sets with their download status."""
    print("\n" + "=" * 80)
    print("AVAILABLE 17LANDS DATASETS")
    print("=" * 80)
    print(f"\n{'Code':<6} {'Name':<35} {'Release':<12} {'Draft Size':<15}")
    print("-" * 70)

    for s in RECENT_SETS:
        url = get_download_url(s['code'])
        exists, size = check_url_exists(url)
        size_str = format_size(size) if exists and size else "N/A"
        print(f"{s['code']:<6} {s['name']:<35} {s['release']:<12} {size_str:<15}")

    print("\n" + "=" * 80)


def check_existing_files(sets: List[str]) -> Dict[str, Dict]:
    """Check which files already exist locally."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for set_code in sets:
        results[set_code] = {"draft": None, "game": None}

        # Check for draft data
        for event in EVENT_TYPES:
            filename = f"draft_data_public.{set_code}.{event}.csv.gz"
            path = DATA_DIR / filename
            if path.exists():
                size = path.stat().st_size
                results[set_code]["draft"] = {
                    "path": str(path),
                    "size": size,
                    "event": event,
                }
                break

        # Check for game data
        for event in EVENT_TYPES:
            filename = f"game_data_public.{set_code}.{event}.csv.gz"
            path = DATA_DIR / filename
            if path.exists():
                size = path.stat().st_size
                results[set_code]["game"] = {
                    "path": str(path),
                    "size": size,
                    "event": event,
                }
                break

    return results


def download_datasets(
    sets: List[str],
    data_types: List[str] = ["draft"],
    event_type: str = "PremierDraft",
    force: bool = False
):
    """Download datasets for specified sets."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    existing = check_existing_files(sets)

    print("\n" + "=" * 80)
    print("DOWNLOADING 17LANDS DATA")
    print("=" * 80)

    total_downloaded = 0
    total_size = 0

    for set_code in sets:
        for data_type in data_types:
            # Check if already exists
            if not force and existing[set_code].get(data_type):
                info = existing[set_code][data_type]
                print(f"\n[SKIP] {set_code} {data_type}: Already exists ({format_size(info['size'])})")
                total_size += info['size']
                continue

            url = get_download_url(set_code, data_type, event_type)
            exists, size = check_url_exists(url)

            if not exists:
                print(f"\n[N/A] {set_code} {data_type}: Not available on 17lands")
                continue

            filename = f"{data_type}_data_public.{set_code}.{event_type}.csv.gz"
            output_path = DATA_DIR / filename

            print(f"\n[DOWNLOAD] {set_code} {data_type}: {format_size(size)}")
            print(f"  URL: {url}")
            print(f"  Output: {output_path}")

            if download_file(url, output_path):
                total_downloaded += 1
                total_size += size
                print("  [OK] Downloaded successfully")
            else:
                print("  [FAIL] Download failed")

    print("\n" + "=" * 80)
    print(f"SUMMARY: Downloaded {total_downloaded} files, Total size: {format_size(total_size)}")
    print("=" * 80)

    # Create manifest
    create_manifest(sets, check_existing_files(sets))


def create_manifest(sets: List[str], existing: Dict[str, Dict]):
    """Create a manifest file of downloaded data."""
    from datetime import datetime

    manifest = {
        "sets": {},
        "total_size": 0,
        "download_date": datetime.now().isoformat(),
        "s3_base": S3_BASE,
    }

    for set_code in sets:
        if existing[set_code]["draft"] or existing[set_code]["game"]:
            manifest["sets"][set_code] = existing[set_code]
            if existing[set_code]["draft"]:
                manifest["total_size"] += existing[set_code]["draft"]["size"]
            if existing[set_code]["game"]:
                manifest["total_size"] += existing[set_code]["game"]["size"]

    manifest_path = DATA_DIR / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest saved to: {manifest_path}")


def show_status(sets: List[str]):
    """Show current download status."""
    existing = check_existing_files(sets)

    print("\n" + "=" * 80)
    print("CURRENT DATA STATUS")
    print("=" * 80)
    print(f"\nData directory: {DATA_DIR}")
    print(f"\n{'Set':<6} {'Draft':<20} {'Game':<20}")
    print("-" * 50)

    total_size = 0
    for set_code in sets:
        draft = existing[set_code]["draft"]
        game = existing[set_code]["game"]

        draft_str = format_size(draft["size"]) if draft else "Not downloaded"
        game_str = format_size(game["size"]) if game else "Not downloaded"

        print(f"{set_code:<6} {draft_str:<20} {game_str:<20}")

        if draft:
            total_size += draft["size"]
        if game:
            total_size += game["size"]

    print("-" * 50)
    print(f"{'TOTAL':<6} {format_size(total_size):<20}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="17lands Data Downloader")
    parser.add_argument("--sets", nargs="+", help="Set codes to download (e.g., FDN DSK BLB)")
    parser.add_argument("--all-recent", action="store_true", help="Download all recent sets (top 10)")
    parser.add_argument("--list", action="store_true", help="List available sets with sizes")
    parser.add_argument("--status", action="store_true", help="Check download status")
    parser.add_argument("--data-types", nargs="+", default=["draft"],
                        choices=["draft", "game", "replay"], help="Data types to download")
    parser.add_argument("--event", default="PremierDraft",
                        choices=EVENT_TYPES, help="Event type")
    parser.add_argument("--force", action="store_true", help="Force re-download existing files")

    args = parser.parse_args()

    if args.list:
        list_available_sets()
        return

    # Determine which sets to process
    if args.all_recent:
        sets = [s["code"] for s in RECENT_SETS[:10]]
    elif args.sets:
        sets = [s.upper() for s in args.sets]
    else:
        # Default: last 5 sets
        sets = [s["code"] for s in RECENT_SETS[:5]]

    if args.status:
        show_status(sets)
        return

    # Download
    download_datasets(sets, args.data_types, args.event, args.force)

    print("\nNext steps:")
    print("1. Verify downloads with: python scripts/download_17lands.py --status")
    print("2. Start training with: python draft_training.py --mode bc")


if __name__ == "__main__":
    main()
