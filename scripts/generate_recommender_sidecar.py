#!/usr/bin/env python3
"""
Generate recommender metadata sidecar.

Reads Scryfall bulk JSON and outputs data/card_recommender_metadata.json with
creature subtypes, tribal types (from oracle text), and land status for each card.

Usage:
    python3 scripts/generate_recommender_sidecar.py
    python3 scripts/generate_recommender_sidecar.py --bulk-json data/scryfall_bulk_cards.json
    python3 scripts/generate_recommender_sidecar.py --format standard
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.mechanics.card_parser import TRIBAL_CONTEXT_PATTERNS  # noqa: E402


def load_bulk_cards(json_path: str, format_filter: str | None = None) -> list[dict]:
    """Load cards from bulk JSON, matching precompute_embeddings.py dedup logic."""
    print(f"Loading cards from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        all_cards = json.load(f)
    print(f"  Loaded {len(all_cards)} total cards")

    cards = [c for c in all_cards if c.get("games") and "paper" in c["games"]]
    print(f"  Paper cards: {len(cards)}")

    if format_filter:
        format_key = format_filter.lower()
        cards = [
            c for c in cards
            if c.get("legalities", {}).get(format_key) in ["legal", "restricted"]
        ]
        print(f"  {format_filter}-legal: {len(cards)}")

    seen_names = set()
    unique_cards = []
    for card in cards:
        name = card.get("name", "")
        if name not in seen_names:
            seen_names.add(name)
            unique_cards.append(card)

    print(f"  Unique cards: {len(unique_cards)}")
    return unique_cards


def extract_creature_subtypes(type_line: str) -> list[str]:
    """Extract creature subtypes from type_line (text after em-dash)."""
    if "\u2014" not in type_line:
        return []
    after_dash = type_line.split("\u2014", 1)[1].strip()
    # Handle DFC type lines: "Creature — Elf // Creature — Warrior"
    # Only take subtypes from first face
    if " // " in after_dash:
        after_dash = after_dash.split(" // ")[0].strip()
    subtypes = [s.strip().lower() for s in after_dash.split() if s.strip()]
    return subtypes


def extract_tribal_types(oracle_text: str) -> list[str]:
    """Extract creature types referenced in tribal context from oracle text."""
    if not oracle_text:
        return []
    text_lower = oracle_text.lower()
    tribal_types = set()
    for pattern in TRIBAL_CONTEXT_PATTERNS:
        for match in pattern.finditer(text_lower):
            tribal_types.add(match.group(1).lower())
    return sorted(tribal_types)


def generate_metadata(cards: list[dict]) -> dict:
    """Generate metadata dict keyed by card name."""
    metadata = {}
    for card in cards:
        name = card.get("name", "")
        type_line = card.get("type_line", "")

        # For DFC cards, concatenate oracle text from both faces
        oracle_text = card.get("oracle_text", "")
        if "card_faces" in card and card["card_faces"]:
            parts = []
            for face in card["card_faces"]:
                face_text = face.get("oracle_text", "")
                if face_text:
                    parts.append(face_text)
            if parts:
                oracle_text = "\n".join(parts)
            # Also grab type_line from first face if needed
            if not type_line:
                type_line = card["card_faces"][0].get("type_line", "")

        creature_subtypes = extract_creature_subtypes(type_line)
        tribal_types = extract_tribal_types(oracle_text)
        is_land = "Land" in type_line

        metadata[name] = {
            "creature_subtypes": creature_subtypes,
            "tribal_types": tribal_types,
            "is_land": is_land,
        }

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Generate recommender metadata sidecar")
    parser.add_argument("--bulk-json", default=os.path.join(PROJECT_ROOT, "data", "scryfall_bulk_cards.json"),
                        help="Path to Scryfall bulk JSON")
    parser.add_argument("--format", default="commander", help="Format filter (default: commander)")
    parser.add_argument("--output", default=os.path.join(PROJECT_ROOT, "data", "card_recommender_metadata.json"),
                        help="Output path")
    args = parser.parse_args()

    if not os.path.exists(args.bulk_json):
        print(f"Error: {args.bulk_json} not found. Download with:")
        print("  python3 -m src.mechanics.precompute_embeddings --format commander --bulk-json data/scryfall_bulk_cards.json")
        sys.exit(1)

    cards = load_bulk_cards(args.bulk_json, args.format)
    metadata = generate_metadata(cards)

    # Stats
    n_with_subtypes = sum(1 for v in metadata.values() if v["creature_subtypes"])
    n_with_tribal = sum(1 for v in metadata.values() if v["tribal_types"])
    n_lands = sum(1 for v in metadata.values() if v["is_land"])

    print("\nMetadata generated:")
    print(f"  Total cards: {len(metadata)}")
    print(f"  With creature subtypes: {n_with_subtypes}")
    print(f"  With tribal references: {n_with_tribal}")
    print(f"  Lands: {n_lands}")

    with open(args.output, "w") as f:
        json.dump(metadata, f, separators=(",", ":"))

    size_kb = os.path.getsize(args.output) / 1024
    print(f"\n  Saved to {args.output} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
