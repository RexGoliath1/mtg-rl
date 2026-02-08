# Local Tools

ForgeRL ships with several interactive tools for exploring card encodings and building decks.

## Card Recommender

**URL**: [http://localhost:8000](http://localhost:8000)

An EDHREC-inspired web application that recommends cards for your Commander deck based on mechanical similarity.

**Features**:
- Paste a decklist (supports Moxfield, Archidekt, Arena, MTGGoldfish formats)
- Three recommendation modes: centroid matching, gap filling, and card alternatives
- Curve-aware scoring that prioritizes underrepresented mana costs
- Tribal filtering, graveyard theme detection, mana fixing boosts
- "Cards to Cut" analysis with replacement suggestions
- Dark theme UI with Scryfall card images

**Run locally**:
```bash
uv run python3 scripts/card_recommender.py
```

## Embedding Quiz

**URL**: [http://localhost:8787](http://localhost:8787)

A review tool for auditing how well the mechanics parser encodes individual cards.

**Features**:
- 3x3 grid of cards with their mechanic encodings
- Thumbs up/down approval for each card's encoding
- Smart suggestion chips for common encoding issues
- Back-face display for double-faced cards (DFCs)
- Reports save to `data/quiz_reports/`

**Run locally**:
```bash
uv run python3 scripts/embedding_quiz.py
```

## Collection Mode

The Card Recommender supports filtering recommendations to only cards you own. Export your collection from your preferred platform and upload it alongside your decklist to get recommendations constrained to your card pool.

## Parser Coverage Report

Generate a report showing how well the mechanics parser covers cards in a given format.

**Run**:
```bash
uv run python3 scripts/parser_coverage_report.py --format standard
```

The report shows per-card confidence scores, average/median coverage, and highlights cards with low encoding confidence.
