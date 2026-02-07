#!/usr/bin/env python3
"""
Fetch diverse Modern decks from MTGGoldfish for training data coverage.

Goal: 100+ unique Modern decks to maximize card/mechanic coverage.
Single format keeps things consistent.
"""

import os
import re
from pathlib import Path

from src.data.mtggoldfish_decks import MTGGoldfishScraper, save_decks


def save_forge_decks_unique(decks, output_dir: str):
    """Save decks in Forge-compatible format with unique names."""
    os.makedirs(output_dir, exist_ok=True)

    saved_names = set()
    saved_count = 0

    for i, deck in enumerate(decks):
        # Create unique filename from deck name
        base_name = re.sub(r"[^\w\-]", "_", deck.name.lower())
        base_name = re.sub(r"_+", "_", base_name).strip("_")

        if not base_name:
            base_name = f"deck_{i}"

        # Ensure uniqueness
        filename = base_name
        counter = 1
        while filename in saved_names:
            filename = f"{base_name}_{counter}"
            counter += 1

        saved_names.add(filename)
        path = os.path.join(output_dir, f"{filename}.dck")

        with open(path, "w") as f:
            f.write(deck.to_forge_format())

        saved_count += 1

    print(f"  Saved {saved_count} decks to {output_dir}")
    return list(saved_names)


def fetch_modern_decks(num_decks: int = 100):
    """Fetch Modern decks for maximum coverage."""

    scraper = MTGGoldfishScraper(delay=1.5)  # Be nice to the server

    print("=" * 70)
    print("FETCHING MODERN DECKS FOR TRAINING")
    print("=" * 70)
    print(f"Target: {num_decks} Modern decks")
    print()

    print(f"Fetching {num_decks} Modern decks...")
    decks = scraper.fetch_top_decks("modern", limit=num_decks, include_lists=True)

    print(f"\nGot {len(decks)} decks from Modern")

    # Filter valid decks
    valid_decks = [d for d in decks if d.total_cards >= 40]
    print(f"Valid decks (40+ cards): {len(valid_decks)}")

    return valid_decks


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-decks", type=int, default=100,
                        help="Number of Modern decks to fetch")
    args = parser.parse_args()

    output_dir = Path("decks/modern")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch decks
    decks = fetch_modern_decks(args.num_decks)

    if not decks:
        print("No decks fetched!")
        return

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total valid decks: {len(decks)}")

    # Unique cards across all decks
    all_cards = set()
    for deck in decks:
        for card in deck.mainboard:
            all_cards.add(card.name)
        for card in deck.sideboard:
            all_cards.add(card.name)

    print(f"Unique cards across all decks: {len(all_cards)}")

    # Archetype distribution
    archetypes = {}
    for d in decks:
        archetypes[d.archetype] = archetypes.get(d.archetype, 0) + 1
    print("\nArchetype distribution:")
    for arch, count in sorted(archetypes.items(), key=lambda x: -x[1]):
        print(f"  {arch}: {count}")

    # Print deck names
    print("\nDecks fetched:")
    for i, d in enumerate(decks[:20]):
        print(f"  {i+1}. {d.name} ({d.total_cards} cards)")
    if len(decks) > 20:
        print(f"  ... and {len(decks)-20} more")

    # Save to Forge format with unique names
    print(f"\nSaving decks to {output_dir}...")
    deck_names = save_forge_decks_unique(decks, str(output_dir))

    # Save JSON for reference
    json_path = "data/decks/modern_meta.json"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    save_decks(decks, json_path)

    # Create a deck list file for the collection script
    deck_list_path = output_dir / "deck_list.txt"
    with open(deck_list_path, "w") as f:
        for name in deck_names:
            f.write(f"modern/{name}.dck\n")

    print(f"\nDeck list saved to {deck_list_path}")
    print(f"\nDone! {len(decks)} Modern decks ready for training.")
    print(f"Covering {len(all_cards)} unique cards.")

    return decks


if __name__ == "__main__":
    main()
