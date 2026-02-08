#!/usr/bin/env python3
"""
Generate a sample MTG collection in Manabox CSV format for testing Collection Mode.

Picks ~950 random non-legendary cards + ~50 random legendary creatures from the HDF5 database.

Usage:
    python3 scripts/generate_sample_collection.py
"""

import json
import os
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    # Load card index from HDF5
    import h5py
    db_path = os.path.join(PROJECT_ROOT, "data", "card_mechanics_commander.h5")
    with h5py.File(db_path, "r") as f:
        card_index = json.loads(f.attrs["card_index"])

    # Load metadata sidecar
    meta_path = os.path.join(PROJECT_ROOT, "data", "card_recommender_metadata.json")
    if not os.path.exists(meta_path):
        print(f"Error: {meta_path} not found. Run scripts/generate_recommender_sidecar.py first.")
        return

    with open(meta_path) as f:
        metadata = json.load(f)

    all_names = list(card_index.keys())

    # Split into legendary creatures and non-legendary
    legendary = [n for n in all_names if metadata.get(n, {}).get("is_legendary", False)]
    non_legendary = [n for n in all_names if not metadata.get(n, {}).get("is_legendary", False)]

    print(f"Total cards: {len(all_names)}")
    print(f"Legendary creatures: {len(legendary)}")
    print(f"Non-legendary: {len(non_legendary)}")

    random.seed(42)
    sample_legendary = random.sample(legendary, min(50, len(legendary)))
    sample_non_legendary = random.sample(non_legendary, min(950, len(non_legendary)))
    sample = sample_legendary + sample_non_legendary
    random.shuffle(sample)

    # Write Manabox CSV format
    output_path = os.path.join(PROJECT_ROOT, "data", "sample_collection.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("Name,Set code,Set name,Collector number,Foil,Rarity,Quantity,ManaBox ID\n")
        for name in sample:
            # Escape commas in card names (DFCs like "Card A // Card B")
            escaped = f'"{name}"' if "," in name else name
            qty = random.randint(1, 4)
            f.write(f"{escaped},,,,,,{qty},\n")

    print(f"\nSample collection written to {output_path}")
    print(f"  Total cards: {len(sample)} ({len(sample_legendary)} legendary + {len(sample_non_legendary)} non-legendary)")


if __name__ == "__main__":
    main()
