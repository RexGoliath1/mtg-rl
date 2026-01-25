#!/usr/bin/env python3
"""
Enrich Card Data with Scryfall Metadata

Fetches card metadata from Scryfall API and creates an enriched database
for the v2 hybrid encoder.

Usage:
    python scripts/enrich_card_data.py --sets FDN DSK BLB TLA
    python scripts/enrich_card_data.py --all  # All cards in 17lands data

Output:
    data/card_metadata.json - Card name -> metadata mapping
    data/card_embeddings.pt - Pre-computed text embeddings (optional)

Size estimate: ~50MB for metadata, ~200MB for embeddings (all Standard cards)
"""

import argparse
import gzip
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
import requests

# Scryfall API rate limit: 10 requests/second
SCRYFALL_RATE_LIMIT = 0.1  # seconds between requests


def get_card_names_from_17lands(data_dir: Path, sets: List[str]) -> Set[str]:
    """Extract unique card names from 17lands data."""
    card_names = set()

    for set_code in sets:
        csv_path = data_dir / f"draft_data_public.{set_code}.PremierDraft.csv.gz"
        if not csv_path.exists():
            print(f"  Skipping {set_code} (not found)")
            continue

        print(f"  Scanning {set_code}...")
        with gzip.open(csv_path, 'rt', encoding='utf-8') as f:
            # Just read header to get card names
            header = f.readline().strip().split(',')
            for col in header:
                if col.startswith('pack_card_'):
                    card_name = col.replace('pack_card_', '')
                    card_names.add(card_name)

    return card_names


def fetch_card_from_scryfall(card_name: str, set_code: Optional[str] = None) -> Optional[Dict]:
    """Fetch card data from Scryfall API."""
    try:
        # Try exact name search first
        params = {"exact": card_name}
        if set_code:
            params["set"] = set_code.lower()

        response = requests.get(
            "https://api.scryfall.com/cards/named",
            params=params,
            timeout=10
        )

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            # Try fuzzy search
            response = requests.get(
                "https://api.scryfall.com/cards/named",
                params={"fuzzy": card_name},
                timeout=10
            )
            if response.status_code == 200:
                return response.json()

        return None

    except requests.RequestException as e:
        print(f"    Error fetching {card_name}: {e}")
        return None


def extract_card_metadata(scryfall_data: Dict) -> Dict:
    """Extract relevant fields from Scryfall response."""
    # Handle double-faced cards
    if "card_faces" in scryfall_data:
        front = scryfall_data["card_faces"][0]
        oracle_text = front.get("oracle_text", "")
        mana_cost = front.get("mana_cost", "")
        type_line = front.get("type_line", scryfall_data.get("type_line", ""))
        power = front.get("power")
        toughness = front.get("toughness")
    else:
        oracle_text = scryfall_data.get("oracle_text", "")
        mana_cost = scryfall_data.get("mana_cost", "")
        type_line = scryfall_data.get("type_line", "")
        power = scryfall_data.get("power")
        toughness = scryfall_data.get("toughness")

    # Parse mana cost to CMC components
    cmc = scryfall_data.get("cmc", 0)
    colors = scryfall_data.get("colors", [])
    color_identity = scryfall_data.get("color_identity", [])

    # Extract keywords
    keywords = scryfall_data.get("keywords", [])

    return {
        "name": scryfall_data["name"],
        "oracle_text": oracle_text,
        "mana_cost": mana_cost,
        "cmc": cmc,
        "colors": colors,
        "color_identity": color_identity,
        "type_line": type_line,
        "power": power,
        "toughness": toughness,
        "keywords": keywords,
        "rarity": scryfall_data.get("rarity", "common"),
        "set": scryfall_data.get("set", "").upper(),
        "scryfall_id": scryfall_data.get("id"),
    }


def fetch_set_bulk(set_code: str) -> List[Dict]:
    """Fetch all cards from a set using Scryfall search."""
    cards = []
    url = f"https://api.scryfall.com/cards/search?q=set:{set_code.lower()}"

    while url:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            print(f"    Error fetching set {set_code}: {response.status_code}")
            break

        data = response.json()
        cards.extend(data.get("data", []))

        if data.get("has_more"):
            url = data.get("next_page")
            time.sleep(SCRYFALL_RATE_LIMIT)
        else:
            url = None

    return cards


def build_enriched_database(
    data_dir: Path,
    sets: List[str],
    output_path: Path,
    use_bulk: bool = True,
) -> Dict[str, Dict]:
    """Build enriched card database from Scryfall."""

    print("\n[1/3] Extracting card names from 17lands data...")
    card_names = get_card_names_from_17lands(data_dir, sets)
    print(f"  Found {len(card_names)} unique cards")

    print("\n[2/3] Fetching card metadata from Scryfall...")
    metadata = {}

    if use_bulk:
        # Fetch by set (faster)
        for set_code in sets:
            print(f"  Fetching set {set_code}...")
            cards = fetch_set_bulk(set_code)
            print(f"    Got {len(cards)} cards")

            for card in cards:
                name = card["name"]
                if name in card_names:
                    metadata[name] = extract_card_metadata(card)

            time.sleep(SCRYFALL_RATE_LIMIT * 5)  # Be nice to API
    else:
        # Fetch individual cards (slower but handles missing sets)
        for i, card_name in enumerate(sorted(card_names)):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{len(card_names)}")

            scryfall_data = fetch_card_from_scryfall(card_name)
            if scryfall_data:
                metadata[card_name] = extract_card_metadata(scryfall_data)
            else:
                print(f"    Not found: {card_name}")

            time.sleep(SCRYFALL_RATE_LIMIT)

    # Check for missing cards
    missing = card_names - set(metadata.keys())
    if missing:
        print(f"\n  Fetching {len(missing)} missing cards individually...")
        for card_name in sorted(missing):
            scryfall_data = fetch_card_from_scryfall(card_name)
            if scryfall_data:
                metadata[card_name] = extract_card_metadata(scryfall_data)
            else:
                print(f"    Still missing: {card_name}")
            time.sleep(SCRYFALL_RATE_LIMIT)

    print(f"\n  Total cards with metadata: {len(metadata)}")

    print("\n[3/3] Saving enriched database...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved to: {output_path} ({size_mb:.1f} MB)")

    return metadata


def precompute_embeddings(
    metadata: Dict[str, Dict],
    output_path: Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """Pre-compute text embeddings for all cards."""
    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except ImportError:
        print("  sentence-transformers not installed, skipping embeddings")
        return

    print("\n[Optional] Pre-computing text embeddings...")
    print(f"  Model: {model_name}")

    model = SentenceTransformer(model_name)

    # Build text for each card
    card_texts = {}
    for name, data in metadata.items():
        # Combine relevant text fields
        text_parts = [
            name,
            data.get("type_line", ""),
            data.get("oracle_text", ""),
        ]
        if data.get("keywords"):
            text_parts.append(" ".join(data["keywords"]))
        card_texts[name] = " ".join(text_parts)

    # Compute embeddings in batches
    names = list(card_texts.keys())
    texts = [card_texts[n] for n in names]

    print(f"  Encoding {len(texts)} cards...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_tensor=True)

    # Save as dict mapping name -> embedding
    embedding_dict = {name: embeddings[i] for i, name in enumerate(names)}

    torch.save({
        "embeddings": embedding_dict,
        "model_name": model_name,
        "embedding_dim": embeddings.shape[1],
    }, output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved to: {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Enrich card data with Scryfall metadata")
    parser.add_argument("--sets", nargs="+", default=["FDN", "DSK", "BLB", "TLA"],
                       help="Set codes to process")
    parser.add_argument("--data-dir", type=str, default="data/17lands",
                       help="Directory containing 17lands data")
    parser.add_argument("--output", type=str, default="data/card_metadata.json",
                       help="Output path for metadata JSON")
    parser.add_argument("--compute-embeddings", action="store_true",
                       help="Also pre-compute text embeddings")
    parser.add_argument("--embeddings-output", type=str, default="data/card_embeddings.pt",
                       help="Output path for embeddings")

    args = parser.parse_args()

    print("=" * 60)
    print("Card Data Enrichment")
    print("=" * 60)
    print(f"Sets: {args.sets}")
    print(f"Data dir: {args.data_dir}")
    print(f"Output: {args.output}")

    metadata = build_enriched_database(
        data_dir=Path(args.data_dir),
        sets=args.sets,
        output_path=Path(args.output),
    )

    if args.compute_embeddings:
        precompute_embeddings(
            metadata=metadata,
            output_path=Path(args.embeddings_output),
        )

    print("\n" + "=" * 60)
    print("ENRICHMENT COMPLETE")
    print("=" * 60)
    print(f"Cards enriched: {len(metadata)}")
    print(f"Metadata: {args.output}")
    if args.compute_embeddings:
        print(f"Embeddings: {args.embeddings_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
