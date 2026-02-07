#!/usr/bin/env python3
"""
MTG Card Recommender — EDHREC-inspired card suggestions.

Features:
- KMeans clustering to identify deck themes and recommend per-theme
- Color identity filtering (respects deck's color pips)
- DFC/Adventure name resolution ("Beanstalk Giant" → "Beanstalk Giant // Fertile Footsteps")
- Deck analysis panel with mechanic profile and color breakdown
- Dark-themed 3x3 card grid with Scryfall images

Usage:
    python3 scripts/card_recommender.py
    python3 scripts/card_recommender.py --port 8000

Then open http://localhost:8000 in your browser.
"""

import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs

import numpy as np
import h5py

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from src.mechanics.vocabulary import Mechanic  # noqa: E402

# ---------------------------------------------------------------------------
# Database loading
# ---------------------------------------------------------------------------

DB_PATH = os.path.join(PROJECT_ROOT, "data", "card_mechanics_commander.h5")
COLOR_PATH = os.path.join(PROJECT_ROOT, "data", "card_color_identity.json")

print("Loading card database...")
with h5py.File(DB_PATH, "r") as f:
    MECHANICS = f["mechanics"][:]
    PARAMETERS = f["parameters"][:]
    CARD_INDEX = json.loads(f.attrs["card_index"])
    DB_VOCAB_SIZE = int(f.attrs["vocab_size"])
    PARAM_KEYS = json.loads(f.attrs.get("param_keys", "[]"))

IDX_TO_NAME = {v: k for k, v in CARD_INDEX.items()}

# Build reverse mechanic index → name
MECHANIC_NAMES = {}
for m in Mechanic:
    if m.value < DB_VOCAB_SIZE:
        MECHANIC_NAMES[m.value] = m.name

# Build front-face name index for DFCs/Adventures
# "Beanstalk Giant" → "Beanstalk Giant // Fertile Footsteps"
FRONT_FACE_INDEX: dict[str, str] = {}
for full_name in CARD_INDEX:
    if " // " in full_name:
        front = full_name.split(" // ")[0]
        FRONT_FACE_INDEX[front] = full_name

# Load color identity
COLOR_IDENTITY: dict[str, list[str]] = {}
if os.path.exists(COLOR_PATH):
    with open(COLOR_PATH) as f:
        COLOR_IDENTITY = json.load(f)
    print(f"Loaded color identity for {len(COLOR_IDENTITY):,} cards")
else:
    print(f"Warning: {COLOR_PATH} not found, color filtering disabled")

print(f"Loaded {len(CARD_INDEX):,} cards (vocab={DB_VOCAB_SIZE}, DFCs={len(FRONT_FACE_INDEX)})")

# Load recommender metadata sidecar (creature subtypes, tribal types, is_land)
METADATA_PATH = os.path.join(PROJECT_ROOT, "data", "card_recommender_metadata.json")
METADATA: dict[str, dict] = {}
IS_LAND = np.zeros(len(CARD_INDEX), dtype=bool)
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH) as f:
        METADATA = json.load(f)
    for name, idx in CARD_INDEX.items():
        if METADATA.get(name, {}).get("is_land", False):
            IS_LAND[idx] = True
    print(f"Loaded metadata sidecar ({len(METADATA):,} cards, {IS_LAND.sum()} lands)")
else:
    print(f"Warning: {METADATA_PATH} not found, run scripts/generate_recommender_sidecar.py")

# Pre-compute CREATURE_TYPE_MATTERS index for tribal filtering
CTM_IDX = Mechanic.CREATURE_TYPE_MATTERS.value if Mechanic.CREATURE_TYPE_MATTERS.value < len(MECHANICS[0]) else None

# Pre-compute mechanics counts per card (used for filtering)
MECH_COUNTS = MECHANICS.sum(axis=1)

# EDHREC rank quality signal (lower rank = more popular)
EDHREC_RANK = np.full(len(CARD_INDEX), 999999, dtype=np.float64)
if METADATA:
    for name, idx in CARD_INDEX.items():
        rank = METADATA.get(name, {}).get("edhrec_rank")
        if rank is not None:
            EDHREC_RANK[idx] = rank
    n_ranked = int((EDHREC_RANK < 999999).sum())
    print(f"EDHREC ranks loaded: {n_ranked:,} cards ranked")

# Quality boost: mild multiplier based on EDHREC rank (log scale)
# Sol Ring (rank ~1) gets ~1.05x, unranked cards get 1.0x
QUALITY_BOOST = 1.0 + 0.1 * np.clip(1.0 - np.log1p(EDHREC_RANK) / 12.0, 0, 0.5)

# Pre-compute CMC array for curve-aware recommendations
CARD_CMC = np.full(len(CARD_INDEX), -1.0, dtype=np.float64)
if METADATA:
    for name, idx in CARD_INDEX.items():
        cmc = METADATA.get(name, {}).get("cmc")
        if cmc is not None and not METADATA.get(name, {}).get("is_land", False):
            CARD_CMC[idx] = cmc
    n_with_cmc = int((CARD_CMC >= 0).sum())
    print(f"CMC data loaded: {n_with_cmc:,} nonland cards")

# CMC bucket for each card (0-7, where 7 = 7+). -1 for lands/unknown.
CARD_CMC_BUCKET = np.full(len(CARD_INDEX), -1, dtype=np.int32)
CARD_CMC_BUCKET[CARD_CMC >= 0] = np.clip(CARD_CMC[CARD_CMC >= 0].astype(np.int32), 0, 7)

# Ideal curve proportions (fraction of nonland cards per CMC bucket 0-7+)
# Commander: slightly higher curve, more 4-5 drops
IDEAL_CURVE_COMMANDER = {0: 0.03, 1: 0.12, 2: 0.22, 3: 0.20, 4: 0.15, 5: 0.11, 6: 0.08, 7: 0.09}
# 60-card: lower, faster curve
IDEAL_CURVE_60 = {0: 0.03, 1: 0.22, 2: 0.28, 3: 0.22, 4: 0.14, 5: 0.06, 6: 0.03, 7: 0.02}

# Graveyard-theme mechanic indices (for theme relevance filtering)
_GY_MECH_NAMES = {
    "MILL", "REANIMATE", "REGROWTH", "CAST_FROM_GRAVEYARD", "FROM_GRAVEYARD",
    "TO_GRAVEYARD", "FLASHBACK", "ESCAPE", "GRAVEYARD_SHUFFLE", "DREDGE",
    "THRESHOLD", "DELIRIUM",
}
GRAVEYARD_THEME_INDICES: set[int] = set()
for m in Mechanic:
    if m.name in _GY_MECH_NAMES and m.value < DB_VOCAB_SIZE:
        GRAVEYARD_THEME_INDICES.add(m.value)

# Color name mapping
COLOR_NAMES = {"W": "White", "U": "Blue", "B": "Black", "R": "Red", "G": "Green"}
COLOR_HEX = {"W": "#f9faf4", "U": "#0e68ab", "B": "#150b00", "R": "#d3202a", "G": "#00733e"}
COLOR_CSS = {"W": "#f0e6c0", "U": "#4a9bd9", "B": "#a070a0", "R": "#e05050", "G": "#40b060"}


# ---------------------------------------------------------------------------
# Card name resolution
# ---------------------------------------------------------------------------

def resolve_card_name(name: str) -> str | None:
    """Resolve a card name to its HDF5 index key, handling DFCs."""
    if name in CARD_INDEX:
        return name
    # Try front-face lookup
    if name in FRONT_FACE_INDEX:
        return FRONT_FACE_INDEX[name]
    # Try case-insensitive
    name_lower = name.lower()
    for key in CARD_INDEX:
        if key.lower() == name_lower:
            return key
    for front, full in FRONT_FACE_INDEX.items():
        if front.lower() == name_lower:
            return full
    return None


# ---------------------------------------------------------------------------
# Scryfall helpers
# ---------------------------------------------------------------------------

SCRYFALL_NAMED_URL = "https://api.scryfall.com/cards/named"
_scryfall_cache: dict[str, dict | None] = {}


def scryfall_card_info(name: str) -> dict | None:
    """Fetch card info from Scryfall (cached)."""
    if name in _scryfall_cache:
        return _scryfall_cache[name]

    # For DFC names, try the front face for Scryfall lookup
    lookup_name = name.split(" // ")[0] if " // " in name else name
    url = f"{SCRYFALL_NAMED_URL}?exact={urllib.parse.quote(lookup_name)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MTG-CardRecommender/1.0"})
        with urllib.request.urlopen(req) as resp:
            card = json.loads(resp.read().decode())
    except Exception:
        _scryfall_cache[name] = None
        return None

    image_uri = ""
    oracle_text = card.get("oracle_text", "")
    type_line = card.get("type_line", "")
    mana_cost = card.get("mana_cost", "")
    color_identity = card.get("color_identity", [])

    if "card_faces" in card and card["card_faces"]:
        front = card["card_faces"][0]
        image_uri = front.get("image_uris", {}).get("normal", "")
        oracle_text = front.get("oracle_text", oracle_text)
        type_line = front.get("type_line", type_line)
        mana_cost = front.get("mana_cost", mana_cost)

    if not image_uri:
        image_uri = card.get("image_uris", {}).get("normal", "")

    info = {
        "name": name,
        "image_uri": image_uri,
        "type_line": type_line,
        "oracle_text": oracle_text,
        "mana_cost": mana_cost,
        "color_identity": color_identity,
        "scryfall_uri": card.get("scryfall_uri", ""),
    }
    _scryfall_cache[name] = info
    return info


def batch_scryfall_lookup(names: list[str]) -> list[dict]:
    """Look up card info for a batch, with rate limiting."""
    results = []
    for name in names:
        info = scryfall_card_info(name)
        results.append(info or {
            "name": name, "image_uri": "", "type_line": "", "oracle_text": "",
            "mana_cost": "", "color_identity": [], "scryfall_uri": "",
        })
        if name not in _scryfall_cache or _scryfall_cache.get(name) is None:
            time.sleep(0.105)
    return results


# ---------------------------------------------------------------------------
# Decklist parser
# ---------------------------------------------------------------------------

def parse_decklist(text: str) -> dict:
    """Parse decklist from Moxfield, Archidekt, Arena, MTGGoldfish formats."""
    lines = text.strip().split("\n")
    result = {"mainboard": [], "sideboard": [], "commander": []}
    current_section = "mainboard"

    section_names = {
        "sideboard": "sideboard", "sideboard:": "sideboard",
        "commander": "commander", "commander:": "commander",
        "companion": "sideboard", "companion:": "sideboard",
        "deck": "mainboard", "deck:": "mainboard",
        "mainboard": "mainboard", "mainboard:": "mainboard",
    }

    for line in lines:
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("#"):
            continue

        lower = line.lower()
        if lower in section_names:
            current_section = section_names[lower]
            continue

        match = re.match(
            r"^(\d+)x?\s+([^(]+?)(?:\s+\([^)]+\))?(?:\s+\d+[a-z]?)?(?:\s+\*[FE]\*)?$",
            line,
        )
        if match:
            qty = int(match.group(1))
            card_name = match.group(2).strip()
            result[current_section].append((qty, card_name))

    return result


# ---------------------------------------------------------------------------
# Deck analysis
# ---------------------------------------------------------------------------

def analyze_deck(card_names: list[str]) -> dict:
    """Analyze a deck's mechanics profile, color identity, and themes."""
    resolved = []
    found = []
    missing = []

    for name in card_names:
        key = resolve_card_name(name)
        if key:
            resolved.append(key)
            found.append(name)
        else:
            missing.append(name)

    if not resolved:
        return {"found": found, "missing": missing, "colors": set(), "mechanics_profile": {},
                "themes": [], "vecs": np.array([]), "resolved": []}

    # Deck color identity
    deck_colors = set()
    for name in resolved:
        ci = COLOR_IDENTITY.get(name, [])
        deck_colors.update(ci)

    # Get vectors
    vecs = np.array([MECHANICS[CARD_INDEX[name]] for name in resolved])

    # Mechanics frequency across deck
    mech_freq = Counter()
    for vec in vecs:
        for i in range(len(vec)):
            if vec[i] and i in MECHANIC_NAMES:
                mech_freq[MECHANIC_NAMES[i]] += 1

    # KMeans clustering to find themes (k based on deck size)
    n_cards = len(resolved)
    if n_cards < 6:
        k = 1
    elif n_cards < 15:
        k = 2
    else:
        k = 3

    themes = _kmeans_themes(vecs, resolved, k)

    # Extract deck creature types from metadata
    deck_creature_types: set[str] = set()
    for name in resolved:
        meta = METADATA.get(name, {})
        deck_creature_types.update(meta.get("creature_subtypes", []))
        deck_creature_types.update(meta.get("tribal_types", []))

    # Deck statistics (mana curve, land count, avg CMC)
    deck_stats = compute_deck_stats(resolved)

    return {
        "found": found,
        "missing": missing,
        "resolved": resolved,
        "colors": deck_colors,
        "mechanics_profile": dict(mech_freq.most_common(15)),
        "themes": themes,
        "vecs": vecs,
        "creature_types": deck_creature_types,
        "deck_stats": deck_stats,
    }


def compute_deck_stats(resolved: list[str]) -> dict:
    """Compute mana curve, land count, avg CMC, and recommended land count."""
    if not resolved:
        return {"curve": {}, "avg_cmc": 0, "land_count": 0, "nonland_count": 0,
                "total": 0, "ramp_count": 0, "recommended_lands": 0, "curve_max": 0}

    # Classify cards and build mana curve
    lands = 0
    mdfc_lands = 0
    nonland_cmcs = []
    ramp_count = 0
    curve = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}  # 7 = "7+"

    # Ramp-related mechanic indices
    ramp_mechs = set()
    for name in ("ADD_MANA", "TUTOR_LAND", "MANA_FIXING", "EXTRA_LAND_PLAY"):
        m = getattr(Mechanic, name, None)
        if m and m.value < DB_VOCAB_SIZE:
            ramp_mechs.add(m.value)

    for card_name in resolved:
        meta = METADATA.get(card_name, {})
        is_land = meta.get("is_land", False)
        layout = meta.get("layout", "")
        cmc = meta.get("cmc", 0)

        if is_land:
            lands += 1
            continue

        # Count MDFCs with land backs (they function as partial mana sources)
        if layout == "modal_dfc" and " // " in card_name:
            if meta.get("is_mdfc_land", False):
                mdfc_lands += 1

        # Track nonland CMC for curve
        cmc_int = int(cmc)
        bucket = min(cmc_int, 7)
        curve[bucket] = curve.get(bucket, 0) + 1
        nonland_cmcs.append(cmc)

        # Check if card provides ramp
        if card_name in CARD_INDEX:
            vec = MECHANICS[CARD_INDEX[card_name]]
            if any(vec[m] for m in ramp_mechs if m < len(vec)):
                ramp_count += 1

    avg_cmc = sum(nonland_cmcs) / len(nonland_cmcs) if nonland_cmcs else 0
    total = len(resolved)
    nonland_count = len(nonland_cmcs)

    # Effective land count: full lands + MDFC lands count as ~0.5 each (can't play both sides)
    effective_lands = lands + mdfc_lands * 0.5

    # Recommended land count (Commander heuristic: 42 - floor(ramp/3), min 37)
    # For 60-card: Karsten formula: 19.59 + 1.90 * avg_cmc
    if total > 80:
        # Commander (99-card deck)
        rec_lands = max(37, 42 - ramp_count // 3)
    else:
        # 60-card format
        rec_lands = round(19.59 + 1.90 * avg_cmc)

    curve_max = max(curve.values()) if any(curve.values()) else 1

    # Compute per-bucket curve boost multipliers for UI display
    ideal = IDEAL_CURVE_COMMANDER if total > 80 else IDEAL_CURVE_60
    bucket_boosts = {}
    if nonland_count >= 5:
        for bucket in range(8):
            actual_prop = curve.get(bucket, 0) / nonland_count
            gap = ideal[bucket] - actual_prop
            bucket_boosts[bucket] = round(1.0 + max(-0.15, min(0.25, gap * 2.5)), 2)
    else:
        bucket_boosts = {b: 1.0 for b in range(8)}

    return {
        "curve": curve,
        "avg_cmc": round(avg_cmc, 2),
        "land_count": lands,
        "mdfc_land_count": mdfc_lands,
        "effective_lands": effective_lands,
        "nonland_count": nonland_count,
        "total": total,
        "ramp_count": ramp_count,
        "recommended_lands": rec_lands,
        "curve_max": curve_max,
        "bucket_boosts": bucket_boosts,
    }


def compute_curve_boost(deck_stats: dict) -> np.ndarray:
    """Compute per-card multiplier that boosts cards filling mana curve gaps.

    Cards in underrepresented CMC buckets get up to 1.25x boost.
    Cards in overrepresented buckets get up to 0.85x penalty.
    Lands and unknown-CMC cards get 1.0 (no change).
    """
    curve = deck_stats.get("curve", {})
    nonland_count = deck_stats.get("nonland_count", 0)
    total = deck_stats.get("total", 0)

    if nonland_count < 5:
        return np.ones(len(CARD_INDEX), dtype=np.float64)

    # Pick ideal curve based on deck size
    ideal = IDEAL_CURVE_COMMANDER if total > 80 else IDEAL_CURVE_60

    # Compute actual proportions
    actual_prop = {}
    for bucket in range(8):
        actual_prop[bucket] = curve.get(bucket, 0) / nonland_count

    # Gap = ideal - actual (positive = underrepresented = boost)
    # Scale: +0.10 gap → 1.25x boost, -0.10 gap → 0.85x penalty
    bucket_multiplier = {}
    for bucket in range(8):
        gap = ideal[bucket] - actual_prop[bucket]
        # Clamp multiplier to [0.85, 1.25]
        mult = 1.0 + np.clip(gap * 2.5, -0.15, 0.25)
        bucket_multiplier[bucket] = mult

    # Map to per-card array
    boost = np.ones(len(CARD_INDEX), dtype=np.float64)
    for bucket in range(8):
        mask = CARD_CMC_BUCKET == bucket
        boost[mask] = bucket_multiplier[bucket]

    return boost


def _kmeans_themes(vecs: np.ndarray, card_names: list[str], k: int, max_iter: int = 20) -> list[dict]:
    """Simple KMeans to identify deck themes. Returns list of {label, cards, centroid, top_mechanics}."""
    n = len(vecs)
    if n == 0 or k <= 0:
        return []

    k = min(k, n)
    vecs_float = vecs.astype(np.float64)

    # Initialize centroids (KMeans++)
    rng = np.random.default_rng(42)
    centroids = np.zeros((k, vecs_float.shape[1]))
    centroids[0] = vecs_float[rng.integers(n)]

    for i in range(1, k):
        dists = np.array([np.min([np.sum((v - c) ** 2) for c in centroids[:i]]) for v in vecs_float])
        probs = dists / (dists.sum() + 1e-8)
        centroids[i] = vecs_float[rng.choice(n, p=probs)]

    # Iterate
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # Assign
        dists = np.array([[np.sum((v - c) ** 2) for c in centroids] for v in vecs_float])
        new_labels = dists.argmin(axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        # Update centroids
        for i in range(k):
            mask = labels == i
            if mask.any():
                centroids[i] = vecs_float[mask].mean(axis=0)

    # Build theme info
    # Skip mechanics that are too generic for labeling
    SKIP_LABEL = {"TRIGGERED_ABILITY", "ACTIVATED_ABILITY", "SORCERY_SPEED", "INSTANT_SPEED",
                  "UNTIL_END_OF_TURN", "TARGET_CREATURE", "TARGET_PLAYER"}

    themes = []
    for i in range(k):
        mask = labels == i
        if not mask.any():
            continue
        cluster_cards = [card_names[j] for j in range(n) if labels[j] == i]
        centroid = centroids[i]

        # Top mechanics for this cluster
        top_indices = np.argsort(-centroid)[:20]
        top_mechs = []
        for idx in top_indices:
            if centroid[idx] < 0.1:
                break
            name = MECHANIC_NAMES.get(idx, "")
            if name and name not in SKIP_LABEL:
                top_mechs.append((name, float(centroid[idx])))
            if len(top_mechs) >= 5:
                break

        # Generate a readable label from top 2-3 mechanics
        label_parts = [m[0].replace("_", " ").title() for m in top_mechs[:3]]
        label = " / ".join(label_parts) if label_parts else f"Theme {i+1}"

        themes.append({
            "label": label,
            "cards": cluster_cards,
            "centroid": centroid,
            "top_mechanics": top_mechs,
            "cluster_id": i,
        })

    return themes


# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------

BASIC_LANDS = {"Plains", "Island", "Swamp", "Mountain", "Forest",
               "Snow-Covered Plains", "Snow-Covered Island",
               "Snow-Covered Swamp", "Snow-Covered Mountain",
               "Snow-Covered Forest", "Wastes"}


def get_mechanics_for_idx(idx: int) -> list[str]:
    """Get mechanic names for a card by HDF5 index."""
    vec = MECHANICS[idx]
    return [MECHANIC_NAMES[i] for i in range(len(vec)) if vec[i] and i in MECHANIC_NAMES]


def _build_exclusion_mask(resolved_cards: list[str]) -> np.ndarray:
    """Build a boolean mask of cards to exclude (deck cards, basics, zero-mechanics)."""
    exclude = np.zeros(len(MECHANICS), dtype=bool)
    for card in resolved_cards:
        if card in CARD_INDEX:
            exclude[CARD_INDEX[card]] = True
    for name in BASIC_LANDS:
        if name in CARD_INDEX:
            exclude[CARD_INDEX[name]] = True
    exclude[MECH_COUNTS == 0] = True
    return exclude


def _color_filter_mask(deck_colors: set[str]) -> np.ndarray:
    """Build mask: True for cards whose color identity fits within deck colors."""
    if not COLOR_IDENTITY or not deck_colors:
        return np.ones(len(MECHANICS), dtype=bool)

    mask = np.ones(len(MECHANICS), dtype=bool)
    deck_set = set(deck_colors)
    for name, idx in CARD_INDEX.items():
        card_ci = set(COLOR_IDENTITY.get(name, []))
        if not card_ci.issubset(deck_set):
            mask[idx] = False
    return mask


def _apply_tribal_and_land_penalties(scores: np.ndarray, deck_creature_types: set[str]) -> None:
    """Apply in-place penalties: wrong-tribe CREATURE_TYPE_MATTERS and lands."""
    # Land penalty — deprioritize lands (they rarely help with deck synergy)
    scores[IS_LAND] *= 0.5

    # Wrong-tribe penalty for CREATURE_TYPE_MATTERS cards
    if CTM_IDX is not None and deck_creature_types and METADATA:
        for idx in range(len(MECHANICS)):
            if MECHANICS[idx, CTM_IDX]:
                card_name = IDX_TO_NAME.get(idx, "")
                card_tribal = set(METADATA.get(card_name, {}).get("tribal_types", []))
                if card_tribal and not card_tribal & deck_creature_types:
                    scores[idx] *= 0.1


def recommend_by_themes(
    analysis: dict,
    cards_per_theme: int = 3,
) -> list[dict]:
    """
    Recommend cards grouped by deck theme clusters.

    Returns list of theme dicts with recommendations.
    """
    themes = analysis["themes"]
    resolved = analysis["resolved"]
    deck_colors = analysis["colors"]

    exclude = _build_exclusion_mask(resolved)
    color_ok = _color_filter_mask(deck_colors)
    eligible = ~exclude & color_ok

    # Pre-compute curve gap boost (same for all themes)
    deck_stats = analysis.get("deck_stats", {})
    curve_boost = compute_curve_boost(deck_stats)

    results = []
    already_recommended = set()

    for theme in themes:
        centroid = theme["centroid"].astype(np.float64)
        centroid_norm = np.sqrt((centroid ** 2).sum())

        if centroid_norm < 1e-8:
            results.append({**theme, "recommendations": []})
            continue

        # Cosine similarity
        mech_float = MECHANICS.astype(np.float64)
        card_norms = np.sqrt((mech_float ** 2).sum(axis=1))
        dots = mech_float @ centroid
        scores = dots / (card_norms * centroid_norm + 1e-8)

        # Penalize sparse cards
        scores[MECH_COUNTS < 3] *= 0.3

        # Tribal and land penalties
        _apply_tribal_and_land_penalties(scores, analysis.get("creature_types", set()))

        # Quality boost (EDHREC popularity)
        scores *= QUALITY_BOOST

        # Graveyard theme relevance: penalize graveyard-matters cards in non-graveyard themes
        theme_gy_density = sum(centroid[i] for i in GRAVEYARD_THEME_INDICES) / (centroid.sum() + 1e-8)
        if theme_gy_density < 0.1:
            for idx in range(len(MECHANICS)):
                card_gy_count = sum(1 for m in GRAVEYARD_THEME_INDICES if MECHANICS[idx, m])
                if card_gy_count >= 3:
                    scores[idx] *= 0.3

        # Mana curve gap boost: prefer cards that fill underrepresented CMC slots
        scores *= curve_boost

        # Apply masks
        scores[~eligible] = 0
        for name in already_recommended:
            if name in CARD_INDEX:
                scores[CARD_INDEX[name]] = 0

        top_idx = np.argsort(-scores)[:cards_per_theme]
        recs = []
        for idx in top_idx:
            if scores[idx] <= 0:
                break
            name = IDX_TO_NAME[idx]
            already_recommended.add(name)
            mechanics = get_mechanics_for_idx(idx)

            # Find shared mechanics with theme
            theme_mech_names = {m[0] for m in theme["top_mechanics"]}
            shared = [m for m in mechanics if m in theme_mech_names]

            recs.append({
                "name": name,
                "score": float(scores[idx]),
                "mechanics": mechanics,
                "shared_with_theme": shared,
            })

        results.append({**theme, "recommendations": recs})

    return results


def recommend_alternatives(
    card_name: str,
    resolved_cards: list[str],
    deck_colors: set[str],
    limit: int = 9,
) -> list[dict]:
    """Find cards similar to a specific card, respecting color identity."""
    key = resolve_card_name(card_name)
    if not key or key not in CARD_INDEX:
        return []

    query_vec = MECHANICS[CARD_INDEX[key]]
    exclude = _build_exclusion_mask(resolved_cards)
    color_ok = _color_filter_mask(deck_colors)

    intersection = (MECHANICS & query_vec).sum(axis=1)
    union = (MECHANICS | query_vec).sum(axis=1)
    scores = intersection.astype(np.float64) / (union.astype(np.float64) + 1e-8)

    scores[~(~exclude & color_ok)] = 0

    top_idx = np.argsort(-scores)[:limit]
    recs = []
    for idx in top_idx:
        if scores[idx] <= 0:
            break
        name = IDX_TO_NAME[idx]
        recs.append({
            "name": name,
            "score": float(scores[idx]),
            "mechanics": get_mechanics_for_idx(idx),
        })
    return recs


def find_cuts(analysis: dict, limit: int = 5) -> list[dict]:
    """Find deck cards that are least synergistic with the rest of the deck.

    For each card, compute cosine similarity to the centroid of all OTHER deck
    cards. Cards with lowest similarity are the weakest links.
    """
    resolved = analysis["resolved"]
    vecs = analysis["vecs"]
    if len(resolved) < 3:
        return []

    vecs_float = vecs.astype(np.float64)
    results = []

    for i, name in enumerate(resolved):
        # Skip basic lands
        if name in BASIC_LANDS:
            continue
        # Protect non-basic lands from cuts (mana base is structural)
        if name in CARD_INDEX and IS_LAND[CARD_INDEX[name]]:
            continue
        # Protect adventure/MDFC cards (dual utility makes them harder to evaluate)
        card_layout = METADATA.get(name, {}).get("layout", "normal")
        if card_layout in ("adventure", "modal_dfc"):
            continue
        # Centroid of all other cards
        other_vecs = np.delete(vecs_float, i, axis=0)
        centroid = other_vecs.mean(axis=0)
        centroid_norm = np.sqrt((centroid ** 2).sum())
        card_norm = np.sqrt((vecs_float[i] ** 2).sum())
        if centroid_norm < 1e-8 or card_norm < 1e-8:
            continue
        sim = float(np.dot(vecs_float[i], centroid) / (card_norm * centroid_norm))
        results.append({
            "name": name,
            "score": sim,
            "mechanics": get_mechanics_for_idx(CARD_INDEX[name]),
        })

    results.sort(key=lambda x: x["score"])
    return results[:limit]


def map_cuts_to_recs(cuts: list[dict], recs_flat: list[dict]) -> list[dict]:
    """For each recommendation, find the best cut candidate to replace.

    Match by highest shared non-zero mechanics count (the rec "replaces"
    the cut with the most mechanical overlap).
    """
    if not cuts or not recs_flat:
        return recs_flat

    cut_vecs = {}
    for c in cuts:
        if c["name"] in CARD_INDEX:
            cut_vecs[c["name"]] = MECHANICS[CARD_INDEX[c["name"]]]

    if not cut_vecs:
        return recs_flat

    for rec in recs_flat:
        if rec["name"] not in CARD_INDEX:
            continue
        rec_vec = MECHANICS[CARD_INDEX[rec["name"]]]
        best_cut = None
        best_overlap = -1
        for cut_name, cut_vec in cut_vecs.items():
            overlap = int((rec_vec & cut_vec).sum())
            if overlap > best_overlap:
                best_overlap = overlap
                best_cut = cut_name
        if best_cut and best_overlap > 0:
            rec["suggested_cut"] = best_cut

    return recs_flat


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MTG Card Recommender</title>
<style>
  :root {
    --bg: #1a1a2e;
    --card-bg: #16213e;
    --card-border: #0f3460;
    --accent: #e94560;
    --good: #00b894;
    --bad: #e17055;
    --text: #eee;
    --text-muted: #999;
    --tag-bg: #0f3460;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }
  header { text-align: center; padding: 24px 16px 12px; }
  header h1 { font-size: 1.6rem; font-weight: 700; }
  header p { color: var(--text-muted); font-size: 0.85rem; margin-top: 4px; }
  .container { max-width: 1400px; margin: 0 auto; padding: 0 16px 24px; }

  /* Form */
  .form-panel {
    background: var(--card-bg);
    border: 2px solid var(--card-border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
  }
  .form-panel h2 { font-size: 1.1rem; margin-bottom: 12px; }
  textarea {
    width: 100%; height: 200px; padding: 12px;
    border: 1px solid #2d3a5a; border-radius: 8px;
    background: #0d1b2a; color: var(--text);
    font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.85rem;
    resize: vertical;
  }
  textarea::placeholder { color: #555; }
  textarea:focus { outline: none; border-color: var(--accent); }

  .modes { display: flex; gap: 16px; margin: 16px 0; flex-wrap: wrap; }
  .mode-option {
    display: flex; align-items: flex-start; gap: 8px; cursor: pointer;
    padding: 10px 14px; background: #0d1b2a;
    border: 2px solid transparent; border-radius: 8px;
    transition: all 0.15s; flex: 1; min-width: 200px;
  }
  .mode-option:hover { border-color: #2d3a5a; }
  .mode-option.selected { border-color: var(--accent); background: #1a1030; }
  .mode-option input[type="radio"] { display: none; }
  .mode-title { font-weight: 700; font-size: 0.9rem; }
  .mode-desc { font-size: 0.75rem; color: var(--text-muted); margin-top: 2px; }

  .submit-btn {
    padding: 12px 36px; border-radius: 8px; border: none;
    background: var(--accent); color: #fff;
    font-size: 1rem; font-weight: 700; cursor: pointer;
    transition: all 0.15s;
  }
  .submit-btn:hover { transform: scale(1.03); opacity: 0.9; }

  /* Stats bar */
  .stats-bar {
    display: flex; justify-content: center; gap: 24px;
    padding: 8px 16px 16px; font-size: 0.85rem; color: var(--text-muted);
  }
  .stats-bar span { font-weight: 600; color: var(--text); }

  /* Deck analysis panel */
  .analysis-panel {
    background: var(--card-bg); border: 2px solid var(--card-border);
    border-radius: 12px; padding: 20px 24px; margin-bottom: 20px;
  }
  .analysis-panel h2 { font-size: 1rem; margin-bottom: 14px; }
  .analysis-grid {
    display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;
  }
  @media (max-width: 800px) { .analysis-grid { grid-template-columns: 1fr; } }

  .analysis-section h3 {
    font-size: 0.8rem; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;
  }

  /* Color pips */
  .color-pips { display: flex; gap: 6px; flex-wrap: wrap; }
  .color-pip {
    width: 28px; height: 28px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.75rem; border: 2px solid rgba(255,255,255,0.3);
  }

  /* Mechanic bars */
  .mech-bar-row {
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 4px; font-size: 0.72rem;
  }
  .mech-bar-label {
    width: 140px; text-align: right; color: var(--text-muted);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .mech-bar-track {
    flex: 1; height: 10px; background: #2d3a5a; border-radius: 5px; overflow: hidden;
  }
  .mech-bar-fill {
    height: 100%; border-radius: 5px; background: var(--accent);
    transition: width 0.3s;
  }
  .mech-bar-count { width: 20px; font-size: 0.7rem; color: var(--text-muted); }

  /* Theme section */
  .theme-section { margin-bottom: 24px; }
  .theme-header {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 12px; padding-bottom: 8px;
    border-bottom: 2px solid var(--card-border);
  }
  .theme-label {
    font-weight: 700; font-size: 1rem;
  }
  .theme-cards-count {
    font-size: 0.8rem; color: var(--text-muted);
    background: #0d1b2a; padding: 2px 10px; border-radius: 10px;
  }
  .theme-mechs {
    display: flex; gap: 4px; flex-wrap: wrap;
  }

  /* Card grid */
  .grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px;
  }
  @media (max-width: 1100px) { .grid { grid-template-columns: repeat(2, 1fr); } }
  @media (max-width: 700px) { .grid { grid-template-columns: 1fr; } }

  .card {
    background: var(--card-bg); border: 2px solid var(--card-border);
    border-radius: 12px; overflow: hidden;
    transition: border-color 0.2s, transform 0.15s;
    display: flex; flex-direction: column;
  }
  .card:hover { transform: translateY(-2px); border-color: var(--accent); }

  .card-top { display: flex; gap: 12px; padding: 12px; align-items: flex-start; }
  .card-img-wrap { position: relative; width: 130px; min-width: 130px; }
  .card-img { width: 130px; border-radius: 8px; transition: transform 0.2s; }
  .card-img:hover { transform: scale(1.05); }
  .card-img-placeholder {
    width: 130px; height: 181px; background: #0d1b2a; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    color: var(--text-muted); font-size: 0.7rem;
  }
  .card-info { flex: 1; min-width: 0; }
  .card-name { font-weight: 700; font-size: 0.95rem; margin-bottom: 2px; }
  .card-name a { color: var(--text); text-decoration: none; }
  .card-name a:hover { text-decoration: underline; }
  .card-type { font-size: 0.75rem; color: var(--text-muted); margin-bottom: 4px; }
  .card-mana { font-size: 0.75rem; color: #c4a7e7; margin-bottom: 6px; }
  .card-oracle {
    font-size: 0.72rem; color: #ccc; line-height: 1.4;
    max-height: 80px; overflow-y: auto; white-space: pre-wrap; padding-right: 4px;
  }

  .card-bottom { padding: 0 12px 12px; flex: 1; display: flex; flex-direction: column; }

  .score-row { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
  .score-bar-track {
    flex: 1; height: 6px; background: #2d3a5a; border-radius: 3px; overflow: hidden;
  }
  .score-bar-fill { height: 100%; border-radius: 3px; background: var(--good); }
  .score-label { font-size: 0.75rem; font-weight: 600; min-width: 36px; text-align: right; color: var(--good); }

  .mechanics-list { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 4px; }
  .mech-tag {
    display: inline-block; background: var(--tag-bg);
    color: #8ecae6; font-size: 0.6rem; font-weight: 600;
    padding: 2px 6px; border-radius: 4px; white-space: nowrap;
  }
  .mech-tag.shared {
    background: #1b3d2e; color: #7dcea0; border: 1px solid #2d5a3e;
  }

  .shared-label {
    font-size: 0.6rem; color: var(--text-muted); margin-bottom: 3px;
    font-style: italic;
  }
  .cut-suggestion {
    font-size: 0.65rem; color: var(--bad); margin-top: 4px;
    font-style: italic; opacity: 0.85;
  }

  /* Mana curve histogram */
  .curve-chart {
    display: flex; align-items: flex-end; gap: 3px;
    height: 120px; padding: 4px 0;
    border-bottom: 1px solid rgba(255,255,255,0.15);
  }
  .curve-col {
    flex: 1; display: flex; flex-direction: column;
    align-items: center; min-width: 0;
  }
  .curve-bar-wrap {
    flex: 1; width: 100%; display: flex; align-items: flex-end;
  }
  .curve-bar {
    width: 100%; border-radius: 3px 3px 0 0;
    background: var(--accent); transition: height 0.3s;
    min-height: 0;
  }
  .curve-bar.boosted { background: var(--good); }
  .curve-bar.penalized { background: var(--bad); opacity: 0.7; }
  .curve-count {
    font-size: 0.65rem; color: var(--text); font-weight: 600;
    height: 16px; display: flex; align-items: center; justify-content: center;
  }
  .curve-label {
    font-size: 0.6rem; color: var(--text-muted); height: 16px;
    display: flex; align-items: center; justify-content: center;
  }

  /* Deck stats */
  .stat-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 4px 0; font-size: 0.82rem;
  }
  .stat-label { color: var(--text-muted); }
  .stat-value { font-weight: 700; }
  .stat-value.good { color: var(--good); }
  .stat-value.bad { color: var(--bad); }
  .stat-value.neutral { color: var(--text); }
  .land-rec {
    margin-top: 8px; padding: 8px 10px; border-radius: 6px;
    font-size: 0.78rem; line-height: 1.5;
  }
  .land-rec.over { background: rgba(224,80,80,0.12); border: 1px solid rgba(224,80,80,0.25); color: var(--bad); }
  .land-rec.under { background: rgba(0,184,148,0.12); border: 1px solid rgba(0,184,148,0.25); color: var(--good); }
  .land-rec.ok { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); color: var(--text-muted); }

  .rank-badge {
    position: absolute; top: -4px; left: -4px;
    width: 24px; height: 24px; background: var(--accent); color: #fff;
    font-size: 0.7rem; font-weight: 700; border-radius: 50%;
    display: flex; align-items: center; justify-content: center; z-index: 1;
  }

  .warning {
    background: rgba(255, 193, 7, 0.1); border: 1px solid rgba(255, 193, 7, 0.3);
    padding: 10px 14px; border-radius: 8px; margin-bottom: 16px;
    font-size: 0.85rem; color: #ffc107;
  }

  .back-link {
    display: inline-block; margin-top: 16px;
    color: var(--accent); text-decoration: none; font-weight: 600; font-size: 0.9rem;
  }
  .back-link:hover { text-decoration: underline; }

  .info-panel {
    background: var(--card-bg); border: 2px solid var(--card-border);
    border-radius: 12px; padding: 20px 24px; margin-top: 20px;
    font-size: 0.82rem; color: var(--text-muted); line-height: 1.6;
  }
  .info-panel strong { color: var(--text); }
</style>
</head>
<body>
__BODY__
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------

def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _color_pips_html(colors: set[str]) -> str:
    """Render color identity as colored circles."""
    if not colors:
        return '<span class="color-pip" style="background:#888;color:#fff">C</span>'
    order = ["W", "U", "B", "R", "G"]
    pips = []
    for c in order:
        if c in colors:
            bg = COLOR_CSS[c]
            text_color = "#1a1a2e" if c in ("W", "G") else "#fff"
            pips.append(f'<span class="color-pip" style="background:{bg};color:{text_color}">{c}</span>')
    return "".join(pips)


def _mech_bars_html(profile: dict, max_count: int | None = None) -> str:
    """Render mechanic frequency as horizontal bars."""
    if not profile:
        return '<span style="color:var(--text-muted);font-size:0.8rem">No mechanics detected</span>'
    if max_count is None:
        max_count = max(profile.values()) if profile else 1

    skip = {"TRIGGERED_ABILITY", "ACTIVATED_ABILITY", "SORCERY_SPEED", "INSTANT_SPEED"}
    items = [(k, v) for k, v in profile.items() if k not in skip]
    items.sort(key=lambda x: -x[1])
    items = items[:10]

    html = ""
    for mech, count in items:
        pct = count / max_count * 100
        label = mech.replace("_", " ").title()
        html += f"""<div class="mech-bar-row">
          <div class="mech-bar-label">{label}</div>
          <div class="mech-bar-track"><div class="mech-bar-fill" style="width:{pct:.0f}%"></div></div>
          <div class="mech-bar-count">{count}</div>
        </div>"""
    return html


def _deck_stats_html(stats: dict) -> str:
    """Render mana curve histogram and deck statistics."""
    if not stats or stats["total"] == 0:
        return '<span style="color:var(--text-muted);font-size:0.8rem">No deck data</span>'

    curve = stats["curve"]
    curve_max = stats["curve_max"] or 1

    # Build curve histogram bars with boost/penalty indicators
    bars_html = ""
    labels = ["0", "1", "2", "3", "4", "5", "6", "7+"]
    bucket_boosts = stats.get("bucket_boosts", {})
    for i, label in enumerate(labels):
        count = curve.get(i, 0)
        height_pct = (count / curve_max) * 100 if curve_max > 0 else 0
        boost = bucket_boosts.get(i, 1.0)
        if boost >= 1.10:
            bar_class = "curve-bar boosted"
            boost_label = f"+{(boost - 1) * 100:.0f}%"
        elif boost <= 0.92:
            bar_class = "curve-bar penalized"
            boost_label = f"{(boost - 1) * 100:.0f}%"
        else:
            bar_class = "curve-bar"
            boost_label = ""
        count_display = f"{count}" if count > 0 else ""
        if boost_label and count > 0:
            count_display = f"{count} <span style='font-size:0.5rem;opacity:0.7'>{boost_label}</span>"
        bars_html += f"""<div class="curve-col">
          <div class="curve-count">{count_display}</div>
          <div class="curve-bar-wrap"><div class="{bar_class}" style="height:{max(height_pct, 2):.0f}%"></div></div>
          <div class="curve-label">{label}</div>
        </div>"""

    # Land recommendation (use effective_lands which counts MDFCs as ~0.5)
    land_count = stats["land_count"]
    mdfc_lands = stats.get("mdfc_land_count", 0)
    effective = stats.get("effective_lands", land_count)
    rec_lands = stats["recommended_lands"]
    diff = effective - rec_lands
    mdfc_note = f" (+{mdfc_lands} MDFC)" if mdfc_lands > 0 else ""
    if abs(diff) <= 1:
        land_class = "ok"
        land_msg = f"Land count looks good ({land_count}{mdfc_note} lands)"
    elif diff > 1:
        land_class = "over"
        land_msg = f"Consider cutting {diff:.0f} lands ({land_count}{mdfc_note} → {rec_lands} recommended)"
    else:
        land_class = "under"
        land_msg = f"Consider adding {-diff:.0f} more lands ({land_count}{mdfc_note} → {rec_lands} recommended)"

    ramp_note = f" ({stats['ramp_count']} ramp spells detected)" if stats["ramp_count"] > 0 else ""

    return f"""
    <div class="curve-chart">{bars_html}</div>
    <div style="margin-top:10px">
      <div class="stat-row">
        <span class="stat-label">Avg CMC (nonland)</span>
        <span class="stat-value neutral">{stats['avg_cmc']:.2f}</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Lands / Spells</span>
        <span class="stat-value neutral">{stats['land_count']}{f" (+{mdfc_lands} MDFC)" if mdfc_lands > 0 else ""} / {stats['nonland_count']}</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Total Cards</span>
        <span class="stat-value neutral">{stats['total']}</span>
      </div>
    </div>
    <div class="land-rec {land_class}">{land_msg}{ramp_note}</div>"""


def _card_html(card_info: dict, rank: int, score: float, mechanics: list[str],
               shared: list[str] | None = None,
               suggested_cut: str | None = None) -> str:
    """Render a single recommendation card."""
    name = card_info.get("name", "Unknown")
    image_uri = card_info.get("image_uri", "")
    type_line = card_info.get("type_line", "")
    oracle_text = card_info.get("oracle_text", "")
    mana_cost = card_info.get("mana_cost", "")
    scryfall_uri = card_info.get("scryfall_uri", "")

    if image_uri:
        img = f'<img class="card-img" src="{image_uri}" alt="{_html_escape(name)}" loading="lazy">'
        if scryfall_uri:
            img_html = f'<a href="{scryfall_uri}" target="_blank" style="cursor:pointer">{img}</a>'
        else:
            img_html = img
    else:
        img_html = '<div class="card-img-placeholder">No image</div>'

    score_pct = min(score * 100, 100)

    skip = {"TRIGGERED_ABILITY", "ACTIVATED_ABILITY", "SORCERY_SPEED", "INSTANT_SPEED"}
    shared_set = set(shared or [])

    # Show shared mechanics first (highlighted), then others
    shared_tags = "".join(f'<span class="mech-tag shared">{m}</span>' for m in mechanics if m in shared_set and m not in skip)
    other_tags = "".join(f'<span class="mech-tag">{m}</span>' for m in mechanics if m not in shared_set and m not in skip)
    # Limit total
    all_tags = shared_tags + other_tags

    name_escaped = _html_escape(name)
    link = f'<a href="{scryfall_uri}" target="_blank">{name_escaped}</a>' if scryfall_uri else name_escaped

    shared_label = ""
    if shared:
        shared_label = f'<div class="shared-label">Matches theme: {", ".join(m.replace("_"," ").title() for m in shared[:3])}</div>'

    return f"""
    <div class="card">
      <div class="card-top">
        <div class="card-img-wrap">
          <div class="rank-badge">{rank}</div>
          {img_html}
        </div>
        <div class="card-info">
          <div class="card-name">{link}</div>
          <div class="card-type">{_html_escape(type_line)}</div>
          <div class="card-mana">{mana_cost}</div>
          <div class="card-oracle">{_html_escape(oracle_text)}</div>
        </div>
      </div>
      <div class="card-bottom">
        <div class="score-row">
          <div class="score-bar-track">
            <div class="score-bar-fill" style="width:{score_pct:.0f}%"></div>
          </div>
          <div class="score-label">{score:.3f}</div>
        </div>
        {shared_label}
        <div class="mechanics-list">{all_tags}</div>
        {'<div class="cut-suggestion">Replaces: ' + _html_escape(suggested_cut) + '</div>' if suggested_cut else ''}
      </div>
    </div>"""


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class RecommenderHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path == "/":
            self._serve_form()
        elif self.path == "/api/health":
            self._json_response({"status": "ok", "cards": len(CARD_INDEX), "vocab_size": DB_VOCAB_SIZE})
        else:
            self.send_error(404)

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode()

        if self.path == "/":
            params = parse_qs(body)
            decklist = params.get("decklist", [""])[0]
            mode = params.get("mode", ["themes"])[0]
            cards_per_theme = int(params.get("cards_per_theme", ["3"])[0])
            self._serve_results(decklist, mode, cards_per_theme=cards_per_theme)
        elif self.path == "/api/recommend":
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                self._json_response({"error": "Invalid JSON"}, 400)
                return
            self._api_recommend(data)
        else:
            self.send_error(404)

    def _json_response(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _html_response(self, body_html: str):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        html = HTML_PAGE.replace("__BODY__", body_html)
        self.wfile.write(html.encode())

    def _serve_form(self):
        body = f"""
        <header>
          <h1>MTG Card Recommender</h1>
          <p>Paste your decklist to get theme-aware, color-filtered card suggestions.</p>
        </header>
        <div class="container">
          <div class="form-panel">
            <h2>Your Decklist</h2>
            <form method="POST" id="deck-form">
              <textarea name="decklist" id="decklist" placeholder="Paste your decklist here (Moxfield / Archidekt / Arena format)

Example:
1 Sol Ring
1 Command Tower
1 Arcane Signet
1 Rhystic Study
1 Swords to Plowshares
1 Counterspell
1 Beast Within
1 Cultivate"></textarea>
              <div class="modes" id="modes">
                <label class="mode-option selected" onclick="selectMode(this)">
                  <input type="radio" name="mode" value="themes" checked>
                  <div>
                    <div class="mode-title">Theme-Based</div>
                    <div class="mode-desc">Clusters your deck into themes (KMeans) and recommends cards for each theme, filtered by color identity</div>
                  </div>
                </label>
                <label class="mode-option" onclick="selectMode(this)">
                  <input type="radio" name="mode" value="centroid">
                  <div>
                    <div class="mode-title">Global Similar</div>
                    <div class="mode-desc">Cards similar to your deck's overall mechanics centroid</div>
                  </div>
                </label>
                <label class="mode-option" onclick="selectMode(this)">
                  <input type="radio" name="mode" value="alternatives">
                  <div>
                    <div class="mode-title">Alternatives</div>
                    <div class="mode-desc">Find substitutes for the first card in your list</div>
                  </div>
                </label>
              </div>
              <button type="submit" class="submit-btn">Get Recommendations</button>
            </form>
          </div>
          <div class="info-panel">
            <strong>Database:</strong> {len(CARD_INDEX):,} Commander-legal cards |
            <strong>Vocabulary:</strong> {DB_VOCAB_SIZE} mechanics primitives |
            <strong>Method:</strong> KMeans clustering + cosine similarity + color identity filtering
          </div>
        </div>
        <script>
          function selectMode(el) {{
            document.querySelectorAll('.mode-option').forEach(m => m.classList.remove('selected'));
            el.classList.add('selected');
            el.querySelector('input').checked = true;
          }}
        </script>"""
        self._html_response(body)

    def _serve_results(self, decklist: str, mode: str, cards_per_theme: int = 3):
        t0 = time.time()

        parsed = parse_decklist(decklist)
        all_cards = []
        for section in ["commander", "mainboard", "sideboard"]:
            for qty, card in parsed[section]:
                all_cards.extend([card] * qty)

        if not all_cards:
            self._html_response("""
            <header><h1>MTG Card Recommender</h1>
              <p>No cards found in decklist. Check your format and try again.</p>
            </header>
            <div class="container"><a class="back-link" href="/">&larr; Back</a></div>""")
            return

        # Analyze deck
        analysis = analyze_deck(all_cards)
        found = analysis["found"]
        missing = analysis["missing"]
        colors = analysis["colors"]

        # Warning for missing cards
        warning_html = ""
        if missing:
            names = ", ".join(missing[:10])
            more = f" ... and {len(missing) - 10} more" if len(missing) > 10 else ""
            warning_html = f'<div class="warning">Cards not found: {names}{more}</div>'

        # Deck analysis panel
        colors_html = _color_pips_html(colors)
        profile_html = _mech_bars_html(analysis["mechanics_profile"])
        stats_html_panel = _deck_stats_html(analysis.get("deck_stats", {}))

        # Themes summary for analysis
        themes_summary = ""
        for t in analysis["themes"]:
            mechs = " / ".join(m[0].replace("_", " ").title() for m, _ in zip(t["top_mechanics"], range(3)))
            themes_summary += f'<div style="margin-bottom:4px"><strong>{t["label"]}</strong> <span style="color:var(--text-muted)">({len(t["cards"])} cards — {mechs})</span></div>'

        analysis_html = f"""
        <div class="analysis-panel">
          <h2>Deck Analysis</h2>
          <div class="analysis-grid">
            <div class="analysis-section">
              <h3>Mana Curve</h3>
              {stats_html_panel}
            </div>
            <div class="analysis-section">
              <h3>Top Mechanics</h3>
              {profile_html}
            </div>
            <div class="analysis-section">
              <h3>Color Identity</h3>
              <div class="color-pips">{colors_html}</div>
              <div style="margin-top:8px;font-size:0.8rem;color:var(--text-muted)">
                {len(found)} cards found, {len(missing)} missing
              </div>
              <h3 style="margin-top:16px">Detected Themes</h3>
              {themes_summary or '<span style="color:var(--text-muted)">Not enough cards for theme detection</span>'}
            </div>
          </div>
        </div>"""

        # Get recommendations based on mode
        if mode == "themes":
            themed_recs = recommend_by_themes(analysis, cards_per_theme=cards_per_theme)
            all_rec_names = [r["name"] for tr in themed_recs for r in tr["recommendations"]]
        elif mode == "alternatives":
            alt_recs = recommend_alternatives(
                all_cards[0], analysis["resolved"], colors, limit=9
            )
            all_rec_names = [r["name"] for r in alt_recs]
        else:  # centroid
            # Global cosine with color filtering
            exclude = _build_exclusion_mask(analysis["resolved"])
            color_ok = _color_filter_mask(colors)
            eligible = ~exclude & color_ok

            centroid = analysis["vecs"].mean(axis=0).astype(np.float64)
            centroid_norm = np.sqrt((centroid ** 2).sum())
            mech_float = MECHANICS.astype(np.float64)
            card_norms = np.sqrt((mech_float ** 2).sum(axis=1))
            dots = mech_float @ centroid
            scores = dots / (card_norms * centroid_norm + 1e-8)
            scores[MECH_COUNTS < 3] *= 0.3
            _apply_tribal_and_land_penalties(scores, analysis.get("creature_types", set()))
            scores *= QUALITY_BOOST
            scores[~eligible] = 0

            top_idx = np.argsort(-scores)[:9]
            centroid_recs = []
            for idx in top_idx:
                if scores[idx] <= 0:
                    break
                centroid_recs.append({
                    "name": IDX_TO_NAME[idx],
                    "score": float(scores[idx]),
                    "mechanics": get_mechanics_for_idx(idx),
                })
            all_rec_names = [r["name"] for r in centroid_recs]

        # Compute cuts early so we can map them to recommendations
        cuts = find_cuts(analysis, limit=5)

        # Map cuts to recommendations (add suggested_cut to each rec)
        if mode == "themes":
            all_recs_flat = [r for tr in themed_recs for r in tr["recommendations"]]
        elif mode == "alternatives":
            all_recs_flat = alt_recs
        else:
            all_recs_flat = centroid_recs
        map_cuts_to_recs(cuts, all_recs_flat)

        # Fetch Scryfall images
        print(f"  Fetching {len(all_rec_names)} card images from Scryfall...")
        card_infos_map = {}
        for info in batch_scryfall_lookup(all_rec_names):
            card_infos_map[info["name"]] = info

        elapsed = time.time() - t0

        # Build recommendation HTML
        recs_html = ""
        if mode == "themes":
            rank = 1
            for theme_result in themed_recs:
                if not theme_result["recommendations"]:
                    continue
                # Theme header
                theme_mechs_html = "".join(
                    f'<span class="mech-tag">{m[0]}</span>'
                    for m in theme_result["top_mechanics"][:5]
                )
                recs_html += f"""
                <div class="theme-section">
                  <div class="theme-header">
                    <div class="theme-label">{_html_escape(theme_result["label"])}</div>
                    <div class="theme-cards-count">{len(theme_result["cards"])} deck cards</div>
                    <div class="theme-mechs">{theme_mechs_html}</div>
                  </div>
                  <div class="grid">"""
                for rec in theme_result["recommendations"]:
                    info = card_infos_map.get(rec["name"], {"name": rec["name"]})
                    recs_html += _card_html(info, rank, rec["score"], rec["mechanics"],
                                            shared=rec.get("shared_with_theme"),
                                            suggested_cut=rec.get("suggested_cut"))
                    rank += 1
                recs_html += "</div></div>"
        elif mode == "alternatives":
            recs_html = '<div class="grid">'
            for i, rec in enumerate(alt_recs):
                info = card_infos_map.get(rec["name"], {"name": rec["name"]})
                recs_html += _card_html(info, i + 1, rec["score"], rec["mechanics"],
                                        suggested_cut=rec.get("suggested_cut"))
            recs_html += "</div>"
        else:
            recs_html = '<div class="grid">'
            for i, rec in enumerate(centroid_recs):
                info = card_infos_map.get(rec["name"], {"name": rec["name"]})
                recs_html += _card_html(info, i + 1, rec["score"], rec["mechanics"],
                                        suggested_cut=rec.get("suggested_cut"))
            recs_html += "</div>"

        mode_names = {"themes": "Theme-Based", "centroid": "Global Similar", "alternatives": "Alternatives"}

        stats_html = f"""
        <div class="stats-bar">
          <div>Mode: <span>{mode_names.get(mode, mode)}</span></div>
          <div>Deck: <span>{len(all_cards)} cards</span></div>
          <div>Colors: <span>{''.join(sorted(colors)) or 'C'}</span></div>
          <div>Themes: <span>{len(analysis['themes'])}</span></div>
          <div>Time: <span>{elapsed:.1f}s</span></div>
        </div>"""

        # View More button (theme mode only)
        view_more_html = ""
        if mode == "themes":
            next_cpt = cards_per_theme + 3
            escaped_decklist = _html_escape(decklist)
            view_more_html = f"""
            <form method="POST" style="text-align:center;margin:20px 0">
              <input type="hidden" name="decklist" value="{escaped_decklist}">
              <input type="hidden" name="mode" value="themes">
              <input type="hidden" name="cards_per_theme" value="{next_cpt}">
              <button type="submit" class="submit-btn" style="background:#0f3460">View More ({next_cpt} per theme)</button>
            </form>"""

        # Cards to Cut section (cuts already computed above for mapping)
        cuts_html = ""
        if cuts:
            cut_names = [c["name"] for c in cuts]
            print(f"  Fetching {len(cut_names)} cut candidate images from Scryfall...")
            cut_infos = batch_scryfall_lookup(cut_names)
            cut_infos_map = {info["name"]: info for info in cut_infos}

            cuts_cards_html = ""
            for i, cut in enumerate(cuts):
                info = cut_infos_map.get(cut["name"], {"name": cut["name"]})
                cuts_cards_html += _card_html(info, i + 1, cut["score"], cut["mechanics"])

            cuts_html = f"""
            <div class="theme-section" style="margin-top:32px">
              <div class="theme-header">
                <div class="theme-label" style="color:var(--bad)">Cards to Consider Cutting</div>
                <div class="theme-cards-count">least synergistic with deck</div>
              </div>
              <div class="grid">{cuts_cards_html}</div>
            </div>"""

        body = f"""
        <header>
          <h1>Recommendations</h1>
          <p>{mode_names.get(mode, mode)} — color-filtered to {''.join(sorted(colors)) or 'Colorless'}</p>
        </header>
        {stats_html}
        <div class="container">
          {warning_html}
          {analysis_html}
          {recs_html}
          {view_more_html}
          {cuts_html}
          <a class="back-link" href="/">&larr; Try another decklist</a>
        </div>"""

        self._html_response(body)
        print(f"  Served recommendations in {elapsed:.1f}s")

    def _api_recommend(self, data: dict):
        decklist = data.get("decklist", "")
        mode = data.get("mode", "themes")
        limit = data.get("limit", 9)

        parsed = parse_decklist(decklist)
        all_cards = []
        for section in ["commander", "mainboard", "sideboard"]:
            for qty, card in parsed[section]:
                all_cards.extend([card] * qty)

        analysis = analyze_deck(all_cards)

        if mode == "themes":
            themed = recommend_by_themes(analysis, cards_per_theme=limit // max(len(analysis["themes"]), 1))
            recs = [r for tr in themed for r in tr["recommendations"]]
        elif mode == "alternatives" and all_cards:
            recs = recommend_alternatives(all_cards[0], analysis["resolved"], analysis["colors"], limit)
        else:
            recs = []  # Use the UI for centroid mode

        self._json_response({
            "recommendations": recs,
            "found_cards": analysis["found"],
            "missing_cards": analysis["missing"],
            "deck_colors": list(analysis["colors"]),
            "themes": [{"label": t["label"], "cards": t["cards"],
                        "top_mechanics": t["top_mechanics"]} for t in analysis["themes"]],
            "deck_stats": analysis.get("deck_stats", {}),
        })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MTG Card Recommender Server")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print("MTG Card Recommender Server")
    print(f"{'=' * 60}")
    print(f"Cards: {len(CARD_INDEX):,}")
    print(f"Vocab: {DB_VOCAB_SIZE}")
    print(f"DFCs:  {len(FRONT_FACE_INDEX)}")
    print(f"Colors: {len(COLOR_IDENTITY):,} cards with color identity")
    print(f"Memory: {MECHANICS.nbytes / 1024 / 1024:.1f} MB")
    print(f"\nServer running at http://localhost:{args.port}")
    print("Press Ctrl+C to stop")
    print(f"{'=' * 60}\n")

    server = HTTPServer(("localhost", args.port), RecommenderHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
