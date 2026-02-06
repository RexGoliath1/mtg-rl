#!/usr/bin/env python3
"""
Parser Coverage Report

Fetches all legal cards from Scryfall for a given format or set,
parses them through the card parser, and produces a gap analysis.

Usage:
    python3 -m scripts.parser_coverage_report --format standard
    python3 -m scripts.parser_coverage_report --format modern
    python3 -m scripts.parser_coverage_report --format commander
    python3 -m scripts.parser_coverage_report --set FDN
    python3 -m scripts.parser_coverage_report --set DSK --set BLB
    python3 -m scripts.parser_coverage_report --format standard --json report.json
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mechanics.card_parser import parse_card, parse_oracle_text
from src.mechanics.vocabulary import Mechanic

SCRYFALL_SEARCH_URL = "https://api.scryfall.com/cards/search"
REQUEST_DELAY = 0.1  # 100ms between requests


# =============================================================================
# SCRYFALL FETCH
# =============================================================================

def build_query(formats: Optional[List[str]] = None,
                sets: Optional[List[str]] = None) -> str:
    """Build Scryfall search query for legal cards only."""
    parts = ["has:oracle_text", "-is:digital", "-is:funny", "-is:token"]

    if formats:
        fmt_clauses = [f"f:{f}" for f in formats]
        if len(fmt_clauses) == 1:
            parts.append(fmt_clauses[0])
        else:
            parts.append(f"({' or '.join(fmt_clauses)})")

    if sets:
        set_clauses = [f"set:{s}" for s in sets]
        if len(set_clauses) == 1:
            parts.append(set_clauses[0])
        else:
            parts.append(f"({' or '.join(set_clauses)})")

    return " ".join(parts)


def fetch_all_cards(query: str, verbose: bool = True) -> List[Dict]:
    """Fetch all cards matching query via Scryfall paginated search."""
    cards = []
    url = SCRYFALL_SEARCH_URL
    params = {"q": query, "unique": "cards", "order": "name"}
    page = 1

    while True:
        resp = requests.get(url, params=params)
        if resp.status_code == 404:
            # No results
            break
        resp.raise_for_status()
        data = resp.json()

        batch = data.get("data", [])
        cards.extend(batch)
        total = data.get("total_cards", "?")

        if verbose:
            print(f"  Page {page}: {len(cards)}/{total} cards fetched",
                  file=sys.stderr)

        if not data.get("has_more", False):
            break

        # Follow next page URL directly (includes cursor)
        url = data["next_page"]
        params = {}  # next_page URL includes all params
        page += 1
        time.sleep(REQUEST_DELAY)

    return cards


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_card(card: Dict) -> Dict:
    """Parse a single card and return analysis dict."""
    name = card.get("name", "Unknown")
    oracle = card.get("oracle_text", "")
    type_line = card.get("type_line", "")
    set_code = card.get("set", "???").upper()

    # Handle DFCs
    if "card_faces" in card and card["card_faces"]:
        front = card["card_faces"][0]
        oracle = front.get("oracle_text", "")
        type_line = front.get("type_line", type_line)

    if not oracle.strip():
        return {
            "name": name,
            "set": set_code,
            "type_line": type_line,
            "confidence": 1.0,  # Vanilla — nothing to parse
            "mechanic_count": 0,
            "mechanics": [],
            "unparsed_text": "",
            "oracle_text": "",
            "is_vanilla": True,
        }

    try:
        result = parse_oracle_text(oracle, type_line)
        return {
            "name": name,
            "set": set_code,
            "type_line": type_line,
            "confidence": result.confidence,
            "mechanic_count": len(result.mechanics),
            "mechanics": [m.name for m in result.mechanics],
            "unparsed_text": result.unparsed_text,
            "oracle_text": oracle,
            "is_vanilla": False,
        }
    except Exception as e:
        return {
            "name": name,
            "set": set_code,
            "type_line": type_line,
            "confidence": 0.0,
            "mechanic_count": 0,
            "mechanics": [],
            "unparsed_text": oracle,
            "oracle_text": oracle,
            "error": str(e),
            "is_vanilla": False,
        }


def analyze_all(cards: List[Dict], verbose: bool = True) -> List[Dict]:
    """Parse all cards and return analysis results."""
    results = []
    t0 = time.time()
    for i, card in enumerate(cards):
        results.append(analyze_card(card))
        if verbose and (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  Parsed {i + 1}/{len(cards)} ({rate:.0f} cards/sec)",
                  file=sys.stderr)
    elapsed = time.time() - t0
    if verbose:
        print(f"  Parsed {len(cards)} cards in {elapsed:.1f}s "
              f"({len(cards)/elapsed:.0f} cards/sec)", file=sys.stderr)
    return results


# =============================================================================
# REPORT
# =============================================================================

def generate_report(results: List[Dict], label: str) -> Dict:
    """Generate a structured report from analysis results."""
    # Filter out vanilla cards for confidence stats
    non_vanilla = [r for r in results if not r.get("is_vanilla", False)]
    confidences = [r["confidence"] for r in non_vanilla]

    if not confidences:
        return {"error": "No cards with oracle text found"}

    avg_conf = sum(confidences) / len(confidences)

    # Confidence buckets
    buckets = {
        "high_90plus": len([c for c in confidences if c >= 0.9]),
        "good_70_90": len([c for c in confidences if 0.7 <= c < 0.9]),
        "medium_50_70": len([c for c in confidences if 0.5 <= c < 0.7]),
        "low_30_50": len([c for c in confidences if 0.3 <= c < 0.5]),
        "very_low_sub30": len([c for c in confidences if c < 0.3]),
    }

    # Per-set breakdown
    set_stats = {}
    for r in non_vanilla:
        s = r["set"]
        if s not in set_stats:
            set_stats[s] = {"confidences": [], "count": 0}
        set_stats[s]["confidences"].append(r["confidence"])
        set_stats[s]["count"] += 1

    per_set = {}
    for s, data in sorted(set_stats.items()):
        confs = data["confidences"]
        per_set[s] = {
            "count": data["count"],
            "avg_confidence": sum(confs) / len(confs),
            "above_50pct": len([c for c in confs if c >= 0.5]),
            "below_30pct": len([c for c in confs if c < 0.3]),
        }

    # Bottom N cards
    sorted_by_conf = sorted(non_vanilla, key=lambda r: r["confidence"])
    bottom_50 = sorted_by_conf[:50]

    # Mechanic frequency
    mechanic_counts = Counter()
    for r in non_vanilla:
        for m in r["mechanics"]:
            mechanic_counts[m] += 1

    # Unparsed phrase frequency
    phrase_counts = Counter()
    for r in non_vanilla:
        unparsed = r.get("unparsed_text", "")
        if not unparsed:
            continue
        # Split into meaningful fragments
        for part in unparsed.split("  "):
            part = part.strip()
            if len(part) > 5:
                # Normalize to first 60 chars
                phrase_counts[part[:60].lower()] += 1

    # Top unparsed phrases (appearing 3+ times)
    common_unparsed = [
        {"phrase": phrase, "count": count}
        for phrase, count in phrase_counts.most_common(50)
        if count >= 2
    ]

    return {
        "label": label,
        "total_cards": len(results),
        "cards_with_text": len(non_vanilla),
        "vanilla_cards": len(results) - len(non_vanilla),
        "avg_confidence": round(avg_conf, 4),
        "median_confidence": round(sorted(confidences)[len(confidences) // 2], 4),
        "confidence_buckets": buckets,
        "per_set": per_set,
        "bottom_50": [
            {
                "name": r["name"],
                "set": r["set"],
                "confidence": round(r["confidence"], 3),
                "type": r["type_line"],
                "unparsed": r.get("unparsed_text", "")[:100],
                "mechanics_found": r["mechanics"][:6],
            }
            for r in bottom_50
        ],
        "top_mechanics": dict(mechanic_counts.most_common(40)),
        "common_unparsed_phrases": common_unparsed,
    }


def print_report(report: Dict):
    """Print human-readable report."""
    print("=" * 80)
    print(f"PARSER COVERAGE REPORT: {report['label']}")
    print("=" * 80)

    print(f"\n  Total cards:       {report['total_cards']}")
    print(f"  Cards with text:   {report['cards_with_text']}")
    print(f"  Vanilla (no text): {report['vanilla_cards']}")
    print(f"  Avg confidence:    {report['avg_confidence']:.1%}")
    print(f"  Median confidence: {report['median_confidence']:.1%}")

    b = report["confidence_buckets"]
    total = report["cards_with_text"]
    print(f"\n  Confidence Distribution:")
    print(f"    90%+  : {b['high_90plus']:5d}  "
          f"({'#' * min(b['high_90plus'] * 50 // max(total, 1), 50)})")
    print(f"    70-90%: {b['good_70_90']:5d}  "
          f"({'#' * min(b['good_70_90'] * 50 // max(total, 1), 50)})")
    print(f"    50-70%: {b['medium_50_70']:5d}  "
          f"({'#' * min(b['medium_50_70'] * 50 // max(total, 1), 50)})")
    print(f"    30-50%: {b['low_30_50']:5d}  "
          f"({'#' * min(b['low_30_50'] * 50 // max(total, 1), 50)})")
    print(f"    <30%  : {b['very_low_sub30']:5d}  "
          f"({'#' * min(b['very_low_sub30'] * 50 // max(total, 1), 50)})")

    if report.get("per_set"):
        print(f"\n  Per-Set Breakdown:")
        print(f"    {'Set':6s} {'Cards':>6s} {'Avg Conf':>9s} {'≥50%':>6s} {'<30%':>6s}")
        print(f"    {'─' * 35}")
        for s, data in sorted(report["per_set"].items(),
                              key=lambda x: -x[1]["avg_confidence"]):
            print(f"    {s:6s} {data['count']:6d} "
                  f"{data['avg_confidence']:8.1%} "
                  f"{data['above_50pct']:6d} {data['below_30pct']:6d}")

    print(f"\n  Bottom 20 Cards (Lowest Confidence):")
    print(f"    {'Card':40s} {'Set':5s} {'Conf':>6s} Mechanics Found")
    print(f"    {'─' * 75}")
    for card in report["bottom_50"][:20]:
        mechs = ", ".join(card["mechanics_found"][:3])
        if len(card["mechanics_found"]) > 3:
            mechs += "..."
        print(f"    {card['name'][:40]:40s} {card['set']:5s} "
              f"{card['confidence']:5.0%}  {mechs}")

    if report.get("common_unparsed_phrases"):
        print(f"\n  Top Unparsed Phrases (appearing 2+ times):")
        for item in report["common_unparsed_phrases"][:25]:
            print(f"    [{item['count']:3d}x] {item['phrase']}")

    print(f"\n  Top 20 Mechanics by Frequency:")
    for mech, count in list(report["top_mechanics"].items())[:20]:
        bar = "#" * min(count * 40 // max(total, 1), 40)
        print(f"    {mech:30s} {count:5d}  {bar}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Parser coverage report for MTG cards")
    parser.add_argument("--format", "-f", action="append", dest="formats",
                        help="Format to check (standard, modern, commander). "
                             "Can be specified multiple times.")
    parser.add_argument("--set", "-s", action="append", dest="sets",
                        help="Set code to check (FDN, DSK, BLB). "
                             "Can be specified multiple times.")
    parser.add_argument("--json", "-j", type=str, default=None,
                        help="Output JSON report to file")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress progress output")
    args = parser.parse_args()

    if not args.formats and not args.sets:
        parser.error("Specify at least one --format or --set")

    # Build label
    parts = []
    if args.formats:
        parts.extend(args.formats)
    if args.sets:
        parts.extend(args.sets)
    label = " + ".join(parts).upper()

    # Build query
    query = build_query(formats=args.formats, sets=args.sets)
    verbose = not args.quiet

    if verbose:
        print(f"Query: {query}", file=sys.stderr)
        print(f"Fetching cards from Scryfall...", file=sys.stderr)

    cards = fetch_all_cards(query, verbose=verbose)
    if not cards:
        print("No cards found!", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"Parsing {len(cards)} cards...", file=sys.stderr)

    results = analyze_all(cards, verbose=verbose)
    report = generate_report(results, label)

    print_report(report)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nJSON report saved to: {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
