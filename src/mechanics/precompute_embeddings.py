"""
Pre-compute Card Mechanics Embeddings

Fetches all MTG cards from Scryfall and pre-computes mechanics encodings,
storing them in HDF5 format for fast runtime access.

Output: data/card_mechanics.h5
- mechanics: (num_cards, vocab_size) binary matrix
- parameters: (num_cards, max_params) float matrix
- card_index: JSON mapping card_name -> row index
- metadata: vocab_size, format, timestamp

Usage:
    python -m src.mechanics.precompute_embeddings --format commander
    python -m src.mechanics.precompute_embeddings --format standard
    python -m src.mechanics.precompute_embeddings --format all
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import requests
import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("Warning: h5py not installed. Install with: pip install h5py")

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.mechanics.vocabulary import Mechanic, VOCAB_SIZE, encode_card_to_vector
from src.mechanics.card_parser import parse_card


# =============================================================================
# SCRYFALL API
# =============================================================================

SCRYFALL_BULK_URL = "https://api.scryfall.com/bulk-data"
SCRYFALL_SEARCH_URL = "https://api.scryfall.com/cards/search"

# Rate limiting
REQUEST_DELAY = 0.1  # 100ms between requests (Scryfall asks for this)


def get_bulk_data_url(data_type: str = "default_cards") -> str:
    """Get URL for Scryfall bulk data download."""
    response = requests.get(SCRYFALL_BULK_URL)
    response.raise_for_status()

    bulk_data = response.json()
    for item in bulk_data["data"]:
        if item["type"] == data_type:
            return item["download_uri"]

    raise ValueError(f"Bulk data type '{data_type}' not found")


def download_bulk_cards(output_path: str) -> str:
    """Download all cards from Scryfall bulk data."""
    print("Fetching bulk data URL...")
    url = get_bulk_data_url("default_cards")

    print(f"Downloading bulk card data from {url[:50]}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Get total size for progress
    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = downloaded / total_size * 100
                print(f"\r  Downloaded: {downloaded / 1024 / 1024:.1f} MB ({pct:.1f}%)", end="")

    print(f"\n  Saved to {output_path}")
    return output_path


def search_cards_by_format(format_name: str, max_cards: Optional[int] = None) -> List[Dict]:
    """
    Search for cards legal in a specific format using Scryfall API.

    Args:
        format_name: 'standard', 'commander', 'modern', 'legacy', 'vintage', 'pauper'
        max_cards: Optional limit on number of cards to fetch

    Returns:
        List of card objects
    """
    cards = []
    query = f"legal:{format_name}"
    url = f"{SCRYFALL_SEARCH_URL}?q={query}&unique=cards"

    print(f"Searching for {format_name}-legal cards...")

    page = 1
    while url:
        time.sleep(REQUEST_DELAY)
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        cards.extend(data.get("data", []))
        print(f"\r  Fetched {len(cards)} cards (page {page})...", end="")

        if max_cards and len(cards) >= max_cards:
            cards = cards[:max_cards]
            break

        url = data.get("next_page")
        page += 1

    print(f"\n  Total: {len(cards)} cards")
    return cards


def load_bulk_cards(json_path: str, format_filter: Optional[str] = None) -> List[Dict]:
    """
    Load cards from bulk JSON file, optionally filtering by format.

    Args:
        json_path: Path to Scryfall bulk JSON file
        format_filter: Optional format to filter by ('standard', 'commander', etc.)

    Returns:
        List of card objects
    """
    print(f"Loading cards from {json_path}...")

    with open(json_path, 'r', encoding='utf-8') as f:
        all_cards = json.load(f)

    print(f"  Loaded {len(all_cards)} total cards")

    # Filter to paper cards only (no digital-only)
    cards = [c for c in all_cards if c.get("games") and "paper" in c["games"]]
    print(f"  Paper cards: {len(cards)}")

    # Filter by format if specified
    if format_filter:
        format_key = format_filter.lower()
        cards = [
            c for c in cards
            if c.get("legalities", {}).get(format_key) in ["legal", "restricted"]
        ]
        print(f"  {format_filter}-legal: {len(cards)}")

    # Deduplicate by name (keep first printing)
    seen_names = set()
    unique_cards = []
    for card in cards:
        name = card.get("name", "")
        if name not in seen_names:
            seen_names.add(name)
            unique_cards.append(card)

    print(f"  Unique cards: {len(unique_cards)}")
    return unique_cards


# =============================================================================
# ENCODING
# =============================================================================

# Maximum number of numeric parameters to store per card
MAX_PARAMS = 20

# Parameter keys we track
PARAM_KEYS = [
    "draw_count", "damage", "token_count", "scry_count", "mill_count",
    "life_gain", "life_loss", "mana_value", "power", "toughness",
    "loyalty", "tax_amount", "counter_count", "x_value", "life_threshold",
    "warp_cmc", "kicker_cost", "flashback_cost", "escape_cost", "foretell_cost"
]


def encode_cards(cards: List[Dict], verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Encode a list of cards into mechanics matrices.

    Args:
        cards: List of Scryfall card objects
        verbose: Print progress

    Returns:
        mechanics_matrix: (num_cards, VOCAB_SIZE) binary matrix
        params_matrix: (num_cards, MAX_PARAMS) float matrix
        card_index: Dict mapping card name to row index
    """
    num_cards = len(cards)
    mechanics_matrix = np.zeros((num_cards, VOCAB_SIZE), dtype=np.uint8)
    params_matrix = np.zeros((num_cards, MAX_PARAMS), dtype=np.float32)
    card_index = {}

    parse_failures = []

    for i, card in enumerate(cards):
        if verbose and i % 1000 == 0:
            print(f"\r  Encoding card {i}/{num_cards}...", end="")

        try:
            # Parse the card
            encoding = parse_card(card)

            # Get vector representation
            vec = encode_card_to_vector(encoding)

            # Fill mechanics matrix (multi-hot)
            for m in encoding.mechanics:
                if m.value < VOCAB_SIZE:
                    mechanics_matrix[i, m.value] = 1

            # Fill parameters
            params = encoding.parameters
            for j, key in enumerate(PARAM_KEYS):
                if key in params:
                    val = params[key]
                    if isinstance(val, (int, float)):
                        params_matrix[i, j] = float(val)

            # Add power/toughness if creature
            if encoding.power is not None:
                params_matrix[i, PARAM_KEYS.index("power")] = float(encoding.power) if encoding.power >= 0 else -1
            if encoding.toughness is not None:
                params_matrix[i, PARAM_KEYS.index("toughness")] = float(encoding.toughness) if encoding.toughness >= 0 else -1

            # Store index
            card_index[card.get("name", f"unknown_{i}")] = i

        except Exception as e:
            parse_failures.append((card.get("name", "unknown"), str(e)))
            card_index[card.get("name", f"unknown_{i}")] = i

    if verbose:
        print(f"\r  Encoded {num_cards} cards")
        if parse_failures:
            print(f"  Parse failures: {len(parse_failures)}")
            for name, err in parse_failures[:5]:
                print(f"    - {name}: {err}")
            if len(parse_failures) > 5:
                print(f"    ... and {len(parse_failures) - 5} more")

    return mechanics_matrix, params_matrix, card_index


# =============================================================================
# HDF5 STORAGE
# =============================================================================

def save_to_hdf5(
    output_path: str,
    mechanics_matrix: np.ndarray,
    params_matrix: np.ndarray,
    card_index: Dict[str, int],
    format_name: str,
    compression: str = "gzip"
) -> None:
    """
    Save encoded cards to HDF5 file.

    Args:
        output_path: Path to output HDF5 file
        mechanics_matrix: (num_cards, VOCAB_SIZE) binary matrix
        params_matrix: (num_cards, MAX_PARAMS) float matrix
        card_index: Dict mapping card name to row index
        format_name: Name of format ('standard', 'commander', 'all')
        compression: Compression algorithm ('gzip', 'lzf', None)
    """
    if not HAS_H5PY:
        raise ImportError("h5py required for HDF5 output. Install with: pip install h5py")

    print(f"Saving to {output_path}...")

    with h5py.File(output_path, 'w') as f:
        # Store matrices with compression
        f.create_dataset(
            'mechanics',
            data=mechanics_matrix,
            compression=compression,
            chunks=(min(1000, len(mechanics_matrix)), VOCAB_SIZE)
        )
        f.create_dataset(
            'parameters',
            data=params_matrix,
            compression=compression,
            chunks=(min(1000, len(params_matrix)), MAX_PARAMS)
        )

        # Store card index as JSON in attributes
        f.attrs['card_index'] = json.dumps(card_index)
        f.attrs['param_keys'] = json.dumps(PARAM_KEYS)

        # Metadata
        f.attrs['vocab_size'] = VOCAB_SIZE
        f.attrs['num_cards'] = len(card_index)
        f.attrs['max_params'] = MAX_PARAMS
        f.attrs['format'] = format_name
        f.attrs['created'] = datetime.now().isoformat()
        f.attrs['version'] = "1.0"

    # Report file size
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Saved {len(card_index)} cards ({size_mb:.2f} MB)")


def load_from_hdf5(input_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict]:
    """
    Load encoded cards from HDF5 file.

    Args:
        input_path: Path to HDF5 file

    Returns:
        mechanics_matrix: (num_cards, VOCAB_SIZE) binary matrix
        params_matrix: (num_cards, MAX_PARAMS) float matrix
        card_index: Dict mapping card name to row index
        metadata: Dict with vocab_size, format, etc.
    """
    if not HAS_H5PY:
        raise ImportError("h5py required. Install with: pip install h5py")

    with h5py.File(input_path, 'r') as f:
        mechanics_matrix = f['mechanics'][:]
        params_matrix = f['parameters'][:]
        card_index = json.loads(f.attrs['card_index'])

        metadata = {
            'vocab_size': f.attrs['vocab_size'],
            'num_cards': f.attrs['num_cards'],
            'max_params': f.attrs['max_params'],
            'format': f.attrs['format'],
            'created': f.attrs['created'],
            'version': f.attrs.get('version', '1.0'),
            'param_keys': json.loads(f.attrs.get('param_keys', '[]')),
        }

    return mechanics_matrix, params_matrix, card_index, metadata


def get_card_encoding(
    card_name: str,
    mechanics_matrix: np.ndarray,
    params_matrix: np.ndarray,
    card_index: Dict[str, int]
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Get encoding for a single card by name.

    Args:
        card_name: Name of the card
        mechanics_matrix: Full mechanics matrix
        params_matrix: Full parameters matrix
        card_index: Card name to index mapping

    Returns:
        (mechanics_vector, params_vector) or None if not found
    """
    idx = card_index.get(card_name)
    if idx is None:
        return None
    return mechanics_matrix[idx], params_matrix[idx]


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pre-compute card mechanics embeddings")
    parser.add_argument(
        "--format",
        choices=["standard", "commander", "modern", "legacy", "vintage", "pauper", "all"],
        default="commander",
        help="MTG format to include (default: commander)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output HDF5 file path (default: data/card_mechanics_{format}.h5)"
    )
    parser.add_argument(
        "--bulk-json",
        default=None,
        help="Path to existing Scryfall bulk JSON (will download if not provided)"
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Don't download bulk data, use API search instead (slower)"
    )
    parser.add_argument(
        "--max-cards",
        type=int,
        default=None,
        help="Maximum number of cards to process (for testing)"
    )
    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        os.makedirs("data", exist_ok=True)
        output_path = f"data/card_mechanics_{args.format}.h5"

    print("=" * 60)
    print("Pre-computing Card Mechanics Embeddings")
    print("=" * 60)
    print(f"Format: {args.format}")
    print(f"Output: {output_path}")
    print(f"Vocab size: {VOCAB_SIZE}")
    print()

    # Get cards
    if args.bulk_json:
        # Use provided bulk JSON
        format_filter = None if args.format == "all" else args.format
        cards = load_bulk_cards(args.bulk_json, format_filter)
    elif args.no_download:
        # Use API search (slower but no large download)
        if args.format == "all":
            print("Warning: 'all' format with API search will be very slow")
            print("Consider using --bulk-json instead")
            # Fetch multiple formats
            cards = []
            for fmt in ["vintage"]:  # Vintage includes most cards
                cards.extend(search_cards_by_format(fmt, args.max_cards))
            # Deduplicate
            seen = set()
            cards = [c for c in cards if c["name"] not in seen and not seen.add(c["name"])]
        else:
            cards = search_cards_by_format(args.format, args.max_cards)
    else:
        # Download bulk data
        bulk_path = "data/scryfall_bulk_cards.json"
        os.makedirs("data", exist_ok=True)

        if not os.path.exists(bulk_path):
            download_bulk_cards(bulk_path)
        else:
            print(f"Using existing bulk data: {bulk_path}")

        format_filter = None if args.format == "all" else args.format
        cards = load_bulk_cards(bulk_path, format_filter)

    if args.max_cards:
        cards = cards[:args.max_cards]
        print(f"Limited to {len(cards)} cards")

    print()

    # Encode cards
    print("Encoding cards...")
    mechanics_matrix, params_matrix, card_index = encode_cards(cards)

    print()

    # Save to HDF5
    save_to_hdf5(output_path, mechanics_matrix, params_matrix, card_index, args.format)

    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)

    # Quick verification
    print("\nVerification - loading and checking a few cards:")
    mech, params, idx, meta = load_from_hdf5(output_path)
    print(f"  Loaded {meta['num_cards']} cards, vocab size {meta['vocab_size']}")

    # Show a sample card
    sample_names = ["Lightning Bolt", "Counterspell", "Sol Ring", "Rhystic Study"]
    for name in sample_names:
        result = get_card_encoding(name, mech, params, idx)
        if result:
            m, p = result
            mechanics_count = int(np.sum(m))
            print(f"  {name}: {mechanics_count} mechanics")


if __name__ == "__main__":
    main()
