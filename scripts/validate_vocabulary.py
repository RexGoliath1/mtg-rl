#!/usr/bin/env python3
"""
Validate Mechanics Vocabulary Against Scryfall Card Database

Downloads all unique cards from Scryfall's bulk data API and runs
the card parser against every Standard/Modern/Commander-legal card.

Reports:
- Confidence distribution (how well we parse each card)
- Mechanics coverage (how many primitives get used)
- Worst-parsed cards (highest unparsed text)
- Format-specific breakdowns
- Specific tricky card tests
"""

import json
import sys
import urllib.request
import urllib.error
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mechanics.card_parser import parse_card, parse_oracle_text
from src.mechanics.vocabulary import Mechanic


def download_scryfall_bulk():
    """Download Scryfall oracle cards bulk data."""
    print("Fetching Scryfall bulk data catalog...")
    url = "https://api.scryfall.com/bulk-data"
    req = urllib.request.Request(url, headers={"User-Agent": "ForgeRL/1.0"})
    with urllib.request.urlopen(req) as resp:
        catalog = json.loads(resp.read())

    # Find the "oracle_cards" bulk data
    oracle_url = None
    for entry in catalog.get("data", []):
        if entry.get("type") == "oracle_cards":
            oracle_url = entry.get("download_uri")
            break

    if not oracle_url:
        print("ERROR: Could not find oracle_cards bulk data")
        sys.exit(1)

    print(f"Downloading oracle cards from {oracle_url}...")
    req = urllib.request.Request(oracle_url, headers={"User-Agent": "ForgeRL/1.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())

    print(f"Downloaded {len(data)} unique cards")
    return data


def get_format_legality(card, fmt):
    """Check if a card is legal in a given format."""
    legalities = card.get("legalities", {})
    return legalities.get(fmt, "not_legal") in ("legal", "restricted")


def analyze_card(card):
    """Parse a card and return analysis results."""
    oracle_text = card.get("oracle_text", "")

    # Skip cards with no oracle text (basic lands, vanilla creatures)
    if not oracle_text or not oracle_text.strip():
        return {
            "name": card.get("name", "Unknown"),
            "confidence": 1.0,
            "mechanics_count": 0,
            "unparsed": "",
            "mechanics": [],
            "parameters": {},
            "type_line": card.get("type_line", ""),
            "oracle_text": "",
            "is_vanilla": True,
        }

    try:
        _ = parse_card(card)  # Validate card is parseable
        result = parse_oracle_text(oracle_text, card.get("type_line", ""))
        return {
            "name": card.get("name", "Unknown"),
            "confidence": result.confidence,
            "mechanics_count": len(result.mechanics),
            "unparsed": result.unparsed_text,
            "mechanics": [m.name for m in result.mechanics],
            "parameters": result.parameters,
            "type_line": card.get("type_line", ""),
            "oracle_text": oracle_text,
            "is_vanilla": False,
        }
    except Exception as e:
        return {
            "name": card.get("name", "Unknown"),
            "confidence": 0.0,
            "mechanics_count": 0,
            "unparsed": oracle_text,
            "mechanics": [],
            "parameters": {},
            "type_line": card.get("type_line", ""),
            "oracle_text": oracle_text,
            "is_vanilla": False,
            "error": str(e),
        }


def test_specific_cards(all_cards):
    """Test specific tricky cards and show detailed output."""
    tricky_names = [
        "Dragon's Rage Channeler",
        "Cryptic Command",
        "Omnath, Locus of Creation",
        "Korvold, Fae-Cursed King",
        "Atraxa, Grand Unifier",
        "Ragavan, Nimble Pilferer",
        "The One Ring",
        "Orcish Bowmasters",
        "Grief",
        "Fury",
        "Solitude",
        "Sheoldred, the Apocalypse",
        "Phyrexian Obliterator",
        "Niv-Mizzet, Parun",
        "Yawgmoth, Thran Physician",
        "Thassa's Oracle",
        "Dockside Extortionist",
        "Smothering Tithe",
        "Rhystic Study",
        "Cyclonic Rift",
        "Demonic Tutor",
        "Mana Drain",
        "Force of Will",
        "Counterspell",
        "Lightning Bolt",
        "Swords to Plowshares",
        "Path to Exile",
        "Thoughtseize",
        "Tarmogoyf",
        "Urza's Saga",
    ]

    card_map = {c.get("name", ""): c for c in all_cards}

    print("\n" + "=" * 80)
    print("SPECIFIC CARD TESTS")
    print("=" * 80)

    found = 0
    for name in tricky_names:
        card = card_map.get(name)
        if not card:
            continue
        found += 1

        result = analyze_card(card)
        conf_bar = "#" * int(result["confidence"] * 20)
        conf_pad = "." * (20 - int(result["confidence"] * 20))

        print(f"\n{'─' * 80}")
        print(f"  {name}")
        print(f"  Type: {result['type_line']}")
        print(f"  Confidence: [{conf_bar}{conf_pad}] {result['confidence']:.1%}")
        print(f"  Mechanics ({result['mechanics_count']}): {', '.join(result['mechanics'])}")
        if result["parameters"]:
            print(f"  Parameters: {result['parameters']}")
        if result["unparsed"]:
            unparsed_preview = result["unparsed"][:120]
            print(f"  UNPARSED: {unparsed_preview}")

    print(f"\n  Found {found}/{len(tricky_names)} test cards in Scryfall data")


def main():
    all_cards = download_scryfall_bulk()

    # Filter out tokens, emblems, art cards, etc.
    real_cards = [
        c for c in all_cards
        if c.get("layout") not in ("token", "emblem", "art_series", "double_faced_token")
        and c.get("set_type") not in ("token", "memorabilia", "funny")
        and "Card" not in c.get("type_line", "")  # Skip "Card" type (test cards)
    ]

    # Build format buckets
    formats = {
        "standard": [],
        "modern": [],
        "commander": [],
    }

    for card in real_cards:
        for fmt in formats:
            if get_format_legality(card, fmt):
                formats[fmt].append(card)

    print("\nCard counts by format:")
    print(f"  Total unique cards: {len(real_cards)}")
    for fmt, cards in formats.items():
        print(f"  {fmt.title()}: {len(cards)}")

    # Analyze ALL cards (not just format-legal)
    print(f"\nParsing {len(real_cards)} cards...")

    results = []
    errors = 0
    for card in real_cards:
        r = analyze_card(card)
        results.append(r)
        if "error" in r:
            errors += 1

    non_vanilla = [r for r in results if not r.get("is_vanilla", False)]
    vanilla_count = len(results) - len(non_vanilla)

    print(f"\nParsed {len(results)} cards ({errors} errors)")
    print(f"  Vanilla (no text): {vanilla_count}")
    print(f"  With oracle text: {len(non_vanilla)}")

    # Confidence distribution
    print("\n" + "=" * 80)
    print("CONFIDENCE DISTRIBUTION (cards with oracle text)")
    print("=" * 80)

    conf_buckets = Counter()
    for r in non_vanilla:
        bucket = int(r["confidence"] * 10) / 10
        bucket = min(bucket, 1.0)
        conf_buckets[bucket] += 1

    total_nv = len(non_vanilla)
    cumulative = 0
    for bucket in sorted(conf_buckets.keys()):
        count = conf_buckets[bucket]
        cumulative += count
        pct = count / total_nv * 100
        cum_pct = cumulative / total_nv * 100
        bar = "#" * int(pct)
        print(f"  {bucket:.1f}-{bucket+0.1:.1f}: {count:6d} ({pct:5.1f}%) cum: {cum_pct:5.1f}% {bar}")

    avg_conf = sum(r["confidence"] for r in non_vanilla) / max(1, len(non_vanilla))
    high_conf = sum(1 for r in non_vanilla if r["confidence"] >= 0.7)
    low_conf = sum(1 for r in non_vanilla if r["confidence"] < 0.4)
    print(f"\n  Average confidence: {avg_conf:.3f}")
    print(f"  High confidence (>=0.7): {high_conf} ({high_conf/total_nv*100:.1f}%)")
    print(f"  Low confidence (<0.4): {low_conf} ({low_conf/total_nv*100:.1f}%)")

    # Format-specific confidence
    print("\n" + "=" * 80)
    print("CONFIDENCE BY FORMAT")
    print("=" * 80)

    for fmt, cards in formats.items():
        fmt_results = []
        for card in cards:
            r = analyze_card(card)
            if not r.get("is_vanilla", False):
                fmt_results.append(r)

        if not fmt_results:
            continue

        avg = sum(r["confidence"] for r in fmt_results) / len(fmt_results)
        high = sum(1 for r in fmt_results if r["confidence"] >= 0.7)
        low = sum(1 for r in fmt_results if r["confidence"] < 0.4)
        print(f"\n  {fmt.title()} ({len(fmt_results)} cards with text):")
        print(f"    Average confidence: {avg:.3f}")
        print(f"    High confidence (>=0.7): {high} ({high/len(fmt_results)*100:.1f}%)")
        print(f"    Low confidence (<0.4): {low} ({low/len(fmt_results)*100:.1f}%)")

    # Mechanics usage
    print("\n" + "=" * 80)
    print("MECHANICS USAGE")
    print("=" * 80)

    mechanic_counts = Counter()
    for r in non_vanilla:
        for m in r["mechanics"]:
            mechanic_counts[m] += 1

    all_mechanics = set(m.name for m in Mechanic)
    used_mechanics = set(mechanic_counts.keys())
    unused_mechanics = all_mechanics - used_mechanics

    print(f"\n  Total primitives defined: {len(all_mechanics)}")
    print(f"  Primitives used: {len(used_mechanics)} ({len(used_mechanics)/len(all_mechanics)*100:.1f}%)")
    print(f"  Primitives unused: {len(unused_mechanics)}")

    print("\n  Top 30 most used mechanics:")
    for mech, count in mechanic_counts.most_common(30):
        pct = count / total_nv * 100
        bar = "#" * max(1, int(pct / 2))
        print(f"    {mech:30s} {count:6d} ({pct:5.1f}%) {bar}")

    if unused_mechanics:
        print(f"\n  Unused mechanics ({len(unused_mechanics)}):")
        for m in sorted(unused_mechanics):
            print(f"    - {m}")

    # Mechanics per card distribution
    print("\n" + "=" * 80)
    print("MECHANICS PER CARD")
    print("=" * 80)

    mech_per_card = Counter()
    for r in non_vanilla:
        mech_per_card[r["mechanics_count"]] += 1

    for count in sorted(mech_per_card.keys()):
        n = mech_per_card[count]
        pct = n / total_nv * 100
        bar = "#" * max(1, int(pct / 2))
        print(f"  {count:2d} mechanics: {n:6d} ({pct:5.1f}%) {bar}")

    avg_mechs = sum(r["mechanics_count"] for r in non_vanilla) / max(1, len(non_vanilla))
    print(f"\n  Average mechanics per card: {avg_mechs:.1f}")

    # Worst parsed cards (lowest confidence, with text)
    print("\n" + "=" * 80)
    print("WORST PARSED CARDS (lowest confidence, with oracle text)")
    print("=" * 80)

    worst = sorted(non_vanilla, key=lambda r: r["confidence"])[:30]
    for r in worst:
        print(f"\n  {r['name']} (confidence: {r['confidence']:.2f})")
        print(f"    Type: {r['type_line']}")
        oracle_preview = r["oracle_text"][:150].replace("\n", " | ")
        print(f"    Text: {oracle_preview}")
        if r["mechanics"]:
            print(f"    Found: {', '.join(r['mechanics'])}")
        if r["unparsed"]:
            print(f"    Unparsed: {r['unparsed'][:100]}")

    # Test specific tricky cards
    test_specific_cards(real_cards)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Total cards analyzed: {len(results)}")
    print(f"  Cards with oracle text: {len(non_vanilla)}")
    print(f"  Average confidence: {avg_conf:.3f}")
    print(f"  Cards >= 0.7 confidence: {high_conf} ({high_conf/total_nv*100:.1f}%)")
    print(f"  Cards < 0.4 confidence: {low_conf} ({low_conf/total_nv*100:.1f}%)")
    print(f"  Vocabulary utilization: {len(used_mechanics)}/{len(all_mechanics)} ({len(used_mechanics)/len(all_mechanics)*100:.1f}%)")
    print(f"  Average mechanics/card: {avg_mechs:.1f}")
    print(f"  Parser errors: {errors}")


if __name__ == "__main__":
    main()
