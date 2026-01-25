"""
MTGGoldfish Meta Deck Scraper

Fetches top meta decks from MTGGoldfish for training.
Using competitive decks ensures:
1. Reasonable deck construction
2. Diverse archetypes (aggro, control, combo, midrange)
3. Cards that synergize well together

Supported Formats:
- Modern
- Standard
- Pioneer
- Legacy
- Pauper

Usage:
    # Fetch top Modern decks
    python -m src.data.mtggoldfish_decks --format modern --top 10

    # Fetch and save to file
    python -m src.data.mtggoldfish_decks --format standard --output decks/standard_meta.json
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


# =============================================================================
# CONFIGURATION
# =============================================================================

MTGGOLDFISH_BASE = "https://www.mtggoldfish.com"

FORMAT_URLS = {
    "modern": "/metagame/modern/full",
    "standard": "/metagame/standard/full",
    "pioneer": "/metagame/pioneer/full",
    "legacy": "/metagame/legacy/full",
    "pauper": "/metagame/pauper/full",
    "vintage": "/metagame/vintage/full",
}

# Rate limiting
REQUEST_DELAY = 1.0  # Seconds between requests


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Card:
    """A card in a deck."""
    name: str
    quantity: int
    is_sideboard: bool = False


@dataclass
class Deck:
    """A complete deck list."""
    name: str
    archetype: str
    format: str
    meta_share: float  # Percentage of meta
    url: str
    mainboard: List[Card]
    sideboard: List[Card]

    @property
    def total_cards(self) -> int:
        return sum(c.quantity for c in self.mainboard)

    @property
    def unique_cards(self) -> int:
        return len(set(c.name for c in self.mainboard + self.sideboard))

    def to_forge_format(self) -> str:
        """Convert to Forge deck format."""
        lines = [f"[metadata]", f"Name={self.name}", f"[Main]"]
        for card in self.mainboard:
            lines.append(f"{card.quantity} {card.name}")
        lines.append("[Sideboard]")
        for card in self.sideboard:
            lines.append(f"{card.quantity} {card.name}")
        return "\n".join(lines)


# =============================================================================
# SCRAPING
# =============================================================================

class MTGGoldfishScraper:
    """Scrapes deck lists from MTGGoldfish."""

    def __init__(self, delay: float = REQUEST_DELAY):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; MTGBot/1.0; +research)",
        })

    def _get_page(self, url: str) -> BeautifulSoup:
        """Fetch and parse a page."""
        time.sleep(self.delay)
        response = self.session.get(url)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")

    def get_metagame(self, format_name: str, limit: int = 20) -> List[Dict]:
        """
        Get top meta decks for a format.

        Args:
            format_name: 'modern', 'standard', 'pioneer', 'legacy', 'pauper'
            limit: Maximum number of decks to fetch

        Returns:
            List of deck summaries with name, archetype, share, url
        """
        if format_name.lower() not in FORMAT_URLS:
            raise ValueError(f"Unknown format: {format_name}. Use: {list(FORMAT_URLS.keys())}")

        url = MTGGOLDFISH_BASE + FORMAT_URLS[format_name.lower()]
        print(f"Fetching metagame from {url}...")

        soup = self._get_page(url)
        decks = []

        # Find archetype tiles
        for tile in soup.select(".archetype-tile")[:limit]:
            try:
                name_elem = tile.select_one(".archetype-tile-title a")
                if not name_elem:
                    continue

                name = name_elem.text.strip()
                deck_url = urljoin(MTGGOLDFISH_BASE, name_elem.get("href", ""))

                # Get meta share
                share_elem = tile.select_one(".archetype-tile-statistic-value")
                share = 0.0
                if share_elem:
                    share_text = share_elem.text.strip().replace("%", "")
                    try:
                        share = float(share_text)
                    except ValueError:
                        pass

                decks.append({
                    "name": name,
                    "archetype": self._classify_archetype(name),
                    "format": format_name,
                    "meta_share": share,
                    "url": deck_url,
                })

            except Exception as e:
                print(f"  Warning: Failed to parse deck tile: {e}")
                continue

        print(f"  Found {len(decks)} decks")
        return decks

    def get_deck_list(self, deck_url: str) -> Optional[Deck]:
        """
        Fetch full deck list from a deck page.

        Args:
            deck_url: Full URL to the deck page

        Returns:
            Deck object with mainboard and sideboard
        """
        print(f"  Fetching deck: {deck_url}")

        try:
            soup = self._get_page(deck_url)

            # Get deck name
            title_elem = soup.select_one("h1.deck-title, h2.title")
            name = title_elem.text.strip() if title_elem else "Unknown Deck"

            # Parse format from URL
            format_match = re.search(r"/deck/(\w+)/", deck_url)
            format_name = format_match.group(1) if format_match else "unknown"

            mainboard = []
            sideboard = []

            # Find the deck table
            for row in soup.select(".deck-table tr"):
                qty_elem = row.select_one(".deck-col-qty")
                name_elem = row.select_one(".deck-col-card a")

                if not qty_elem or not name_elem:
                    continue

                try:
                    qty = int(qty_elem.text.strip())
                except ValueError:
                    continue

                card_name = name_elem.text.strip()

                # Check if sideboard
                is_sb = "sideboard" in row.get("class", []) or \
                        row.find_parent(class_="sideboard") is not None

                card = Card(name=card_name, quantity=qty, is_sideboard=is_sb)
                if is_sb:
                    sideboard.append(card)
                else:
                    mainboard.append(card)

            # Try alternative parsing if table method fails
            if not mainboard:
                mainboard, sideboard = self._parse_deck_text(soup)

            # Try JS parsing if still empty
            if not mainboard:
                mainboard, sideboard = self._parse_js_deck(soup)

            return Deck(
                name=name,
                archetype=self._classify_archetype(name),
                format=format_name,
                meta_share=0.0,
                url=deck_url,
                mainboard=mainboard,
                sideboard=sideboard,
            )

        except Exception as e:
            print(f"    Failed to fetch deck: {e}")
            return None

    def _parse_deck_text(self, soup: BeautifulSoup) -> tuple:
        """Alternative parsing using text-based deck list."""
        mainboard = []
        sideboard = []

        # Look for deck-view-decklist
        deck_text = soup.select_one(".deck-view-decklist, .deck-list-text")
        if deck_text:
            in_sideboard = False
            for line in deck_text.text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if "sideboard" in line.lower():
                    in_sideboard = True
                    continue

                # Parse "4 Lightning Bolt" format
                match = re.match(r"(\d+)\s+(.+)", line)
                if match:
                    qty = int(match.group(1))
                    name = match.group(2).strip()
                    card = Card(name=name, quantity=qty, is_sideboard=in_sideboard)
                    if in_sideboard:
                        sideboard.append(card)
                    else:
                        mainboard.append(card)

        return mainboard, sideboard

    def _parse_js_deck(self, soup: BeautifulSoup) -> tuple:
        """Parse deck from JavaScript initializeDeckComponents call."""
        from urllib.parse import unquote

        mainboard = []
        sideboard = []

        # Find script containing deck data
        for script in soup.find_all("script"):
            text = script.string or ""
            if "initializeDeckComponents" in text:
                # Extract the URL-encoded deck string
                match = re.search(r'initializeDeckComponents\([^,]+,\s*[^,]+,\s*"([^"]+)"', text)
                if match:
                    encoded_deck = match.group(1)
                    decoded = unquote(encoded_deck)

                    in_sideboard = False
                    for line in decoded.split("\n"):
                        line = line.strip()
                        if not line:
                            continue

                        # Check for sideboard marker
                        if line.lower().startswith("sideboard"):
                            in_sideboard = True
                            continue

                        # Parse "4 Lightning Bolt" format
                        card_match = re.match(r"(\d+)\s+(.+)", line)
                        if card_match:
                            qty = int(card_match.group(1))
                            name = card_match.group(2).strip()
                            card = Card(name=name, quantity=qty, is_sideboard=in_sideboard)
                            if in_sideboard:
                                sideboard.append(card)
                            else:
                                mainboard.append(card)
                    break

        return mainboard, sideboard

    def _classify_archetype(self, deck_name: str) -> str:
        """Classify deck into broad archetype."""
        name_lower = deck_name.lower()

        # Aggro indicators
        if any(x in name_lower for x in ["burn", "prowess", "aggro", "zoo", "blitz", "red deck"]):
            return "aggro"

        # Control indicators
        if any(x in name_lower for x in ["control", "miracles", "taxes", "prison"]):
            return "control"

        # Combo indicators
        if any(x in name_lower for x in ["combo", "storm", "scapeshift", "titan", "tron"]):
            return "combo"

        # Midrange indicators
        if any(x in name_lower for x in ["midrange", "jund", "rock", "abzan"]):
            return "midrange"

        # Tempo indicators
        if any(x in name_lower for x in ["delver", "tempo", "shadow"]):
            return "tempo"

        return "unknown"

    def fetch_top_decks(
        self,
        format_name: str,
        limit: int = 10,
        include_lists: bool = True
    ) -> List[Deck]:
        """
        Fetch top meta decks with full lists.

        Args:
            format_name: Format to fetch
            limit: Number of decks
            include_lists: Whether to fetch full deck lists (slower)

        Returns:
            List of Deck objects
        """
        metagame = self.get_metagame(format_name, limit)
        decks = []

        for deck_info in metagame:
            if include_lists:
                deck = self.get_deck_list(deck_info["url"])
                if deck:
                    deck.meta_share = deck_info["meta_share"]
                    deck.archetype = deck_info["archetype"]
                    decks.append(deck)
            else:
                # Return partial deck (no card list)
                decks.append(Deck(
                    name=deck_info["name"],
                    archetype=deck_info["archetype"],
                    format=format_name,
                    meta_share=deck_info["meta_share"],
                    url=deck_info["url"],
                    mainboard=[],
                    sideboard=[],
                ))

        return decks


# =============================================================================
# UTILITIES
# =============================================================================

def save_decks(decks: List[Deck], output_path: str):
    """Save decks to JSON file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    data = {
        "format": decks[0].format if decks else "unknown",
        "num_decks": len(decks),
        "decks": [
            {
                "name": d.name,
                "archetype": d.archetype,
                "meta_share": d.meta_share,
                "url": d.url,
                "mainboard": [asdict(c) for c in d.mainboard],
                "sideboard": [asdict(c) for c in d.sideboard],
                "total_cards": d.total_cards,
                "unique_cards": d.unique_cards,
            }
            for d in decks
        ]
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(decks)} decks to {output_path}")


def load_decks(input_path: str) -> List[Deck]:
    """Load decks from JSON file."""
    with open(input_path) as f:
        data = json.load(f)

    decks = []
    for d in data["decks"]:
        decks.append(Deck(
            name=d["name"],
            archetype=d["archetype"],
            format=data["format"],
            meta_share=d["meta_share"],
            url=d["url"],
            mainboard=[Card(**c) for c in d["mainboard"]],
            sideboard=[Card(**c) for c in d["sideboard"]],
        ))

    return decks


def save_forge_decks(decks: List[Deck], output_dir: str):
    """Save decks in Forge-compatible format."""
    os.makedirs(output_dir, exist_ok=True)

    for i, deck in enumerate(decks):
        filename = re.sub(r"[^\w\-]", "_", deck.name.lower())
        path = os.path.join(output_dir, f"{filename}.dck")

        with open(path, "w") as f:
            f.write(deck.to_forge_format())

        print(f"  Saved: {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fetch meta decks from MTGGoldfish")
    parser.add_argument("--format", type=str, default="modern",
                        choices=list(FORMAT_URLS.keys()),
                        help="Format to fetch (default: modern)")
    parser.add_argument("--top", type=int, default=10,
                        help="Number of top decks to fetch")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file (default: data/decks/{format}_meta.json)")
    parser.add_argument("--forge-dir", type=str, default=None,
                        help="Also save in Forge format to this directory")
    parser.add_argument("--no-lists", action="store_true",
                        help="Skip fetching full deck lists (faster)")
    args = parser.parse_args()

    print("=" * 70)
    print("MTGGoldfish Meta Deck Fetcher")
    print("=" * 70)
    print(f"Format: {args.format}")
    print(f"Top decks: {args.top}")
    print()

    # Fetch decks
    scraper = MTGGoldfishScraper()
    decks = scraper.fetch_top_decks(
        args.format,
        limit=args.top,
        include_lists=not args.no_lists
    )

    if not decks:
        print("No decks found!")
        return

    # Print summary
    print()
    print("=" * 70)
    print("DECK SUMMARY")
    print("=" * 70)
    for deck in decks:
        cards = f"({deck.total_cards} cards)" if deck.mainboard else "(no list)"
        print(f"  {deck.meta_share:5.1f}% | {deck.archetype:10} | {deck.name} {cards}")

    # Save JSON
    output_path = args.output or f"data/decks/{args.format}_meta.json"
    save_decks(decks, output_path)

    # Save Forge format
    if args.forge_dir:
        save_forge_decks(decks, args.forge_dir)

    print()
    print("Done!")


if __name__ == "__main__":
    main()
