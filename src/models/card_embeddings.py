#!/usr/bin/env python3
"""
MTG Card Embedding System

This module handles the complex task of representing MTG cards as fixed-size vectors
for RL training. The challenges are significant:

CHALLENGE ANALYSIS
==================

1. SCALE: ~30,000 unique cards
   - One-hot encoding = 30,000-dim vectors (impractical)
   - Need learned embeddings that capture card similarity

2. KEYWORD CATEGORIES (from Forge's Keyword.java):

   A. Simple Keywords (binary flags) - 50+ types:
      Flying, Trample, Haste, Vigilance, Lifelink, Deathtouch, Reach,
      First Strike, Double Strike, Menace, Defender, Indestructible,
      Hexproof, Shroud, Flash, Fear, Intimidate, Shadow, Infect, etc.

   B. Keywords with Amount (keyword + numeric value) - 30+ types:
      "Absorb 2", "Afflict 3", "Annihilator 6", "Bushido 1",
      "Crew 3", "Dredge 4", "Fabricate 2", "Toxic 1", etc.

   C. Keywords with Cost (keyword + mana/cost) - 40+ types:
      "Kicker {2}{R}", "Flashback {3}{B}", "Morph {2}{G}",
      "Cycling {2}", "Equip {3}", "Unearth {B}", etc.

   D. Keywords with Type (keyword + card type) - 15+ types:
      "Affinity for artifacts", "Protection from red",
      "Landwalk (Swamp)", "Enchant creature", etc.

   E. Complex/Compound Keywords - 20+ types:
      "Partner with [CardName]", "Companion (deck restriction)",
      "Mutate {cost}", "Emerge {cost}", etc.

3. ABILITIES (beyond keywords):

   A. Triggered Abilities: "When/Whenever X, do Y"
      - ETB (enters the battlefield)
      - Dies triggers
      - Attack/block triggers
      - Upkeep/end step triggers

   B. Activated Abilities: "{cost}: effect"
      - Tap abilities
      - Mana abilities
      - Sacrifice abilities

   C. Static Abilities: Continuous effects
      - "Other creatures you control get +1/+1"
      - "Spells cost {1} more to cast"

   D. Replacement Effects: "If X would happen, Y instead"

4. CONTEXT-DEPENDENT VALUES:
   - X spells: "Deal X damage" where X is chosen at cast time
   - "For each creature you control"
   - "Equal to the number of cards in your graveyard"
   - "*/*" power/toughness (e.g., Tarmogoyf)

5. MODAL SPELLS:
   - "Choose one/two/three"
   - Adventure cards (two modes)
   - Split cards (Fire // Ice)
   - Transform/DFC cards

EMBEDDING STRATEGY
==================

We use a hybrid approach:

1. PRE-TRAINED TEXT EMBEDDINGS
   - Use a language model (SentenceTransformer) on card oracle text
   - Captures semantic meaning of card effects
   - Similar cards get similar embeddings

2. STRUCTURED FEATURE EXTRACTION
   - Parse keywords into categorical + numeric features
   - Encode mana costs as color vectors
   - Encode card types as multi-hot vectors

3. GAME-CONTEXT FEATURES
   - Current power/toughness (after modifications)
   - Counters, damage, tapped state
   - Zone location

The final embedding concatenates:
[text_embedding | keyword_features | mana_features | type_features | context_features]
"""

import re
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum
import hashlib


# =============================================================================
# KEYWORD PARSING
# =============================================================================

class KeywordCategory(Enum):
    SIMPLE = "simple"           # Binary: has or doesn't have
    WITH_AMOUNT = "amount"      # Has numeric parameter
    WITH_COST = "cost"          # Has mana/cost parameter
    WITH_TYPE = "type"          # Has type parameter
    COMPLEX = "complex"         # Special handling needed


# Comprehensive keyword classification based on Forge's Keyword.java
KEYWORD_DEFINITIONS = {
    # Simple Keywords (binary flags)
    "Flying": KeywordCategory.SIMPLE,
    "First strike": KeywordCategory.SIMPLE,
    "First Strike": KeywordCategory.SIMPLE,
    "Double strike": KeywordCategory.SIMPLE,
    "Double Strike": KeywordCategory.SIMPLE,
    "Trample": KeywordCategory.SIMPLE,
    "Haste": KeywordCategory.SIMPLE,
    "Vigilance": KeywordCategory.SIMPLE,
    "Lifelink": KeywordCategory.SIMPLE,
    "Deathtouch": KeywordCategory.SIMPLE,
    "Reach": KeywordCategory.SIMPLE,
    "Menace": KeywordCategory.SIMPLE,
    "Defender": KeywordCategory.SIMPLE,
    "Indestructible": KeywordCategory.SIMPLE,
    "Hexproof": KeywordCategory.SIMPLE,
    "Shroud": KeywordCategory.SIMPLE,
    "Flash": KeywordCategory.SIMPLE,
    "Fear": KeywordCategory.SIMPLE,
    "Intimidate": KeywordCategory.SIMPLE,
    "Shadow": KeywordCategory.SIMPLE,
    "Horsemanship": KeywordCategory.SIMPLE,
    "Skulk": KeywordCategory.SIMPLE,
    "Infect": KeywordCategory.SIMPLE,
    "Wither": KeywordCategory.SIMPLE,
    "Changeling": KeywordCategory.SIMPLE,
    "Devoid": KeywordCategory.SIMPLE,
    "Convoke": KeywordCategory.SIMPLE,
    "Delve": KeywordCategory.SIMPLE,
    "Improvise": KeywordCategory.SIMPLE,
    "Cascade": KeywordCategory.SIMPLE,
    "Storm": KeywordCategory.SIMPLE,
    "Prowess": KeywordCategory.SIMPLE,
    "Exalted": KeywordCategory.SIMPLE,
    "Undying": KeywordCategory.SIMPLE,
    "Persist": KeywordCategory.SIMPLE,
    "Evolve": KeywordCategory.SIMPLE,
    "Exploit": KeywordCategory.SIMPLE,
    "Phasing": KeywordCategory.SIMPLE,
    "Banding": KeywordCategory.SIMPLE,
    "Flanking": KeywordCategory.SIMPLE,
    "Provoke": KeywordCategory.SIMPLE,
    "Battle cry": KeywordCategory.SIMPLE,
    "Ascend": KeywordCategory.SIMPLE,
    "Decayed": KeywordCategory.SIMPLE,
    "Living Weapon": KeywordCategory.SIMPLE,
    "Myriad": KeywordCategory.SIMPLE,
    "Daybound": KeywordCategory.SIMPLE,
    "Nightbound": KeywordCategory.SIMPLE,
    "Training": KeywordCategory.SIMPLE,
    "Soulbond": KeywordCategory.SIMPLE,
    "Enlist": KeywordCategory.SIMPLE,
    "Riot": KeywordCategory.SIMPLE,
    "Mentor": KeywordCategory.SIMPLE,
    "Melee": KeywordCategory.SIMPLE,

    # Keywords with Amount
    "Absorb": KeywordCategory.WITH_AMOUNT,
    "Afflict": KeywordCategory.WITH_AMOUNT,
    "Afterlife": KeywordCategory.WITH_AMOUNT,
    "Annihilator": KeywordCategory.WITH_AMOUNT,
    "Bloodthirst": KeywordCategory.WITH_AMOUNT,
    "Bushido": KeywordCategory.WITH_AMOUNT,
    "Crew": KeywordCategory.WITH_AMOUNT,
    "Dredge": KeywordCategory.WITH_AMOUNT,
    "Fabricate": KeywordCategory.WITH_AMOUNT,
    "Fading": KeywordCategory.WITH_AMOUNT,
    "Frenzy": KeywordCategory.WITH_AMOUNT,
    "Graft": KeywordCategory.WITH_AMOUNT,
    "Hideaway": KeywordCategory.WITH_AMOUNT,
    "Modular": KeywordCategory.WITH_AMOUNT,
    "Poisonous": KeywordCategory.WITH_AMOUNT,
    "Rampage": KeywordCategory.WITH_AMOUNT,
    "Renown": KeywordCategory.WITH_AMOUNT,
    "Ripple": KeywordCategory.WITH_AMOUNT,
    "Soulshift": KeywordCategory.WITH_AMOUNT,
    "Toxic": KeywordCategory.WITH_AMOUNT,
    "Tribute": KeywordCategory.WITH_AMOUNT,
    "Vanishing": KeywordCategory.WITH_AMOUNT,
    "Backup": KeywordCategory.WITH_AMOUNT,
    "Casualty": KeywordCategory.WITH_AMOUNT,
    "Saddle": KeywordCategory.WITH_AMOUNT,

    # Keywords with Cost
    "Kicker": KeywordCategory.WITH_COST,
    "Multikicker": KeywordCategory.WITH_COST,
    "Flashback": KeywordCategory.WITH_COST,
    "Madness": KeywordCategory.WITH_COST,
    "Morph": KeywordCategory.WITH_COST,
    "Megamorph": KeywordCategory.WITH_COST,
    "Disguise": KeywordCategory.WITH_COST,
    "Cycling": KeywordCategory.WITH_COST,
    "Equip": KeywordCategory.WITH_COST,
    "Bestow": KeywordCategory.WITH_COST,
    "Dash": KeywordCategory.WITH_COST,
    "Evoke": KeywordCategory.WITH_COST,
    "Unearth": KeywordCategory.WITH_COST,
    "Encore": KeywordCategory.WITH_COST,
    "Escape": KeywordCategory.WITH_COST,
    "Foretell": KeywordCategory.WITH_COST,
    "Suspend": KeywordCategory.WITH_COST,
    "Buyback": KeywordCategory.WITH_COST,
    "Echo": KeywordCategory.WITH_COST,
    "Cumulative upkeep": KeywordCategory.WITH_COST,
    "Ninjutsu": KeywordCategory.WITH_COST,
    "Outlast": KeywordCategory.WITH_COST,
    "Scavenge": KeywordCategory.WITH_COST,
    "Embalm": KeywordCategory.WITH_COST,
    "Eternalize": KeywordCategory.WITH_COST,
    "Blitz": KeywordCategory.WITH_COST,
    "Ward": KeywordCategory.WITH_COST,
    "Reconfigure": KeywordCategory.WITH_COST,
    "Spectacle": KeywordCategory.WITH_COST,
    "Surge": KeywordCategory.WITH_COST,
    "Prowl": KeywordCategory.WITH_COST,
    "Miracle": KeywordCategory.WITH_COST,
    "Overload": KeywordCategory.WITH_COST,
    "Awaken": KeywordCategory.WITH_COST,
    "Disturb": KeywordCategory.WITH_COST,
    "Plot": KeywordCategory.WITH_COST,
    "Craft": KeywordCategory.WITH_COST,
    "Offspring": KeywordCategory.WITH_COST,

    # Keywords with Type
    "Affinity": KeywordCategory.WITH_TYPE,
    "Protection": KeywordCategory.WITH_TYPE,
    "Landwalk": KeywordCategory.WITH_TYPE,
    "Enchant": KeywordCategory.WITH_TYPE,
    "Champion": KeywordCategory.WITH_TYPE,
    "Offering": KeywordCategory.WITH_TYPE,
    "Amplify": KeywordCategory.WITH_TYPE,

    # Complex Keywords
    "Partner": KeywordCategory.COMPLEX,
    "Partner with": KeywordCategory.COMPLEX,
    "Companion": KeywordCategory.COMPLEX,
    "Mutate": KeywordCategory.COMPLEX,
    "Emerge": KeywordCategory.COMPLEX,
    "Splice": KeywordCategory.COMPLEX,
}

# Simple keywords for one-hot encoding (most common combat-relevant ones)
SIMPLE_KEYWORD_LIST = [
    "Flying", "First strike", "Double strike", "Trample", "Haste",
    "Vigilance", "Lifelink", "Deathtouch", "Reach", "Menace",
    "Defender", "Indestructible", "Hexproof", "Shroud", "Flash",
    "Fear", "Intimidate", "Shadow", "Infect", "Wither",
    "Prowess", "Exalted", "Undying", "Persist", "Flanking",
    "Changeling", "Devoid", "Decayed", "Riot", "Training",
]


@dataclass
class ParsedKeyword:
    """Parsed keyword with its parameters."""
    name: str
    category: KeywordCategory
    amount: Optional[int] = None
    cost: Optional[str] = None
    type_param: Optional[str] = None
    raw: str = ""


def parse_keyword(keyword_str: str) -> ParsedKeyword:
    """Parse a keyword string into structured form.

    Examples:
        "Flying" -> ParsedKeyword(name="Flying", category=SIMPLE)
        "Toxic 2" -> ParsedKeyword(name="Toxic", category=WITH_AMOUNT, amount=2)
        "Kicker {2}{R}" -> ParsedKeyword(name="Kicker", category=WITH_COST, cost="{2}{R}")
        "Protection from red" -> ParsedKeyword(name="Protection", category=WITH_TYPE, type_param="red")
    """
    keyword_str = keyword_str.strip()

    # Try to match known keywords
    for kw_name, category in KEYWORD_DEFINITIONS.items():
        if keyword_str.lower().startswith(kw_name.lower()):
            remainder = keyword_str[len(kw_name):].strip()

            if category == KeywordCategory.SIMPLE:
                return ParsedKeyword(name=kw_name, category=category, raw=keyword_str)

            elif category == KeywordCategory.WITH_AMOUNT:
                # Extract number
                match = re.search(r'(\d+)', remainder)
                amount = int(match.group(1)) if match else 1
                return ParsedKeyword(name=kw_name, category=category, amount=amount, raw=keyword_str)

            elif category == KeywordCategory.WITH_COST:
                # Extract mana cost (anything in braces or remaining text)
                cost = remainder if remainder else "{0}"
                return ParsedKeyword(name=kw_name, category=category, cost=cost, raw=keyword_str)

            elif category == KeywordCategory.WITH_TYPE:
                # Extract type (e.g., "from red", "for artifacts", "creature")
                type_param = remainder.replace("from ", "").replace("for ", "")
                return ParsedKeyword(name=kw_name, category=category, type_param=type_param, raw=keyword_str)

            elif category == KeywordCategory.COMPLEX:
                return ParsedKeyword(name=kw_name, category=category, raw=keyword_str)

    # Unknown keyword
    return ParsedKeyword(name=keyword_str, category=KeywordCategory.SIMPLE, raw=keyword_str)


# =============================================================================
# MANA COST PARSING
# =============================================================================

@dataclass
class ManaCostVector:
    """Parsed mana cost as a vector."""
    total_cmc: int = 0
    white: int = 0
    blue: int = 0
    black: int = 0
    red: int = 0
    green: int = 0
    colorless: int = 0
    generic: int = 0
    hybrid: int = 0  # Count of hybrid symbols
    phyrexian: int = 0  # Count of phyrexian symbols
    x_count: int = 0  # Number of X in cost

    def to_vector(self) -> np.ndarray:
        """Convert to normalized feature vector."""
        return np.array([
            self.total_cmc / 15.0,  # Normalize by max typical CMC
            self.white / 5.0,
            self.blue / 5.0,
            self.black / 5.0,
            self.red / 5.0,
            self.green / 5.0,
            self.colorless / 5.0,
            self.generic / 10.0,
            self.hybrid / 3.0,
            self.phyrexian / 3.0,
            self.x_count / 2.0,
        ], dtype=np.float32)


def parse_mana_cost(cost_str: str) -> ManaCostVector:
    """Parse mana cost string like '{2}{W}{W}' into vector.

    Handles:
    - Basic colors: {W}, {U}, {B}, {R}, {G}
    - Generic: {1}, {2}, etc.
    - Colorless: {C}
    - Hybrid: {W/U}, {2/W}
    - Phyrexian: {W/P}, {U/P}
    - X costs: {X}
    - Snow: {S}
    """
    result = ManaCostVector()

    if not cost_str or cost_str == "no cost":
        return result

    # Find all {X} patterns
    symbols = re.findall(r'\{([^}]+)\}', cost_str)

    for symbol in symbols:
        symbol = symbol.upper()

        if symbol == 'W':
            result.white += 1
            result.total_cmc += 1
        elif symbol == 'U':
            result.blue += 1
            result.total_cmc += 1
        elif symbol == 'B':
            result.black += 1
            result.total_cmc += 1
        elif symbol == 'R':
            result.red += 1
            result.total_cmc += 1
        elif symbol == 'G':
            result.green += 1
            result.total_cmc += 1
        elif symbol == 'C':
            result.colorless += 1
            result.total_cmc += 1
        elif symbol == 'X':
            result.x_count += 1
            # X doesn't add to CMC directly
        elif symbol.isdigit():
            result.generic += int(symbol)
            result.total_cmc += int(symbol)
        elif '/' in symbol:
            # Hybrid or Phyrexian
            if 'P' in symbol:
                result.phyrexian += 1
                result.total_cmc += 1
            else:
                result.hybrid += 1
                result.total_cmc += 1
        elif symbol == 'S':
            # Snow mana
            result.colorless += 1
            result.total_cmc += 1

    return result


# =============================================================================
# CARD TYPE PARSING
# =============================================================================

# Major card types
CARD_TYPES = [
    "Creature", "Instant", "Sorcery", "Enchantment", "Artifact",
    "Land", "Planeswalker", "Battle", "Tribal", "Kindred"
]

# Common creature types (most relevant for gameplay)
CREATURE_TYPES = [
    "Human", "Soldier", "Wizard", "Warrior", "Knight", "Cleric",
    "Zombie", "Vampire", "Elf", "Goblin", "Merfolk", "Spirit",
    "Dragon", "Angel", "Demon", "Beast", "Elemental", "Giant",
    "Bird", "Cat", "Dog", "Snake", "Spider", "Rat",
    "Artificer", "Rogue", "Shaman", "Druid", "Monk", "Assassin",
    "Artifact",  # For artifact creatures
]

# Common subtypes for non-creatures
OTHER_SUBTYPES = [
    "Aura", "Equipment", "Vehicle", "Saga", "Food", "Treasure",
    "Clue", "Blood", "Plains", "Island", "Swamp", "Mountain", "Forest",
]


@dataclass
class CardTypeVector:
    """Parsed card types as a vector."""
    is_creature: bool = False
    is_instant: bool = False
    is_sorcery: bool = False
    is_enchantment: bool = False
    is_artifact: bool = False
    is_land: bool = False
    is_planeswalker: bool = False
    is_legendary: bool = False
    is_token: bool = False
    is_basic: bool = False
    creature_types: Set[str] = field(default_factory=set)

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector."""
        base = np.array([
            float(self.is_creature),
            float(self.is_instant),
            float(self.is_sorcery),
            float(self.is_enchantment),
            float(self.is_artifact),
            float(self.is_land),
            float(self.is_planeswalker),
            float(self.is_legendary),
            float(self.is_token),
            float(self.is_basic),
        ], dtype=np.float32)

        # Add creature type indicators (top 20 types)
        creature_type_features = np.array([
            float(ct in self.creature_types) for ct in CREATURE_TYPES[:20]
        ], dtype=np.float32)

        return np.concatenate([base, creature_type_features])


def parse_card_types(type_str: str) -> CardTypeVector:
    """Parse card type string like 'Legendary Creature - Human Soldier'."""
    result = CardTypeVector()
    type_str = type_str.lower()

    # Check supertypes
    result.is_legendary = "legendary" in type_str
    result.is_basic = "basic" in type_str
    result.is_token = "token" in type_str

    # Check main types
    result.is_creature = "creature" in type_str
    result.is_instant = "instant" in type_str
    result.is_sorcery = "sorcery" in type_str
    result.is_enchantment = "enchantment" in type_str
    result.is_artifact = "artifact" in type_str
    result.is_land = "land" in type_str
    result.is_planeswalker = "planeswalker" in type_str

    # Extract subtypes (after the dash)
    if " - " in type_str or " — " in type_str:
        parts = re.split(r' [-—] ', type_str)
        if len(parts) > 1:
            subtypes = parts[1].split()
            for subtype in subtypes:
                subtype = subtype.strip().title()
                if subtype in CREATURE_TYPES:
                    result.creature_types.add(subtype)

    return result


# =============================================================================
# CARD EMBEDDING
# =============================================================================

class CardEmbedding:
    """
    Generates fixed-size embeddings for MTG cards.

    Embedding structure:
    - Keyword features: 64 dims (30 simple keywords + parsed amounts/counts)
    - Mana features: 11 dims (color breakdown, CMC, hybrid, etc.)
    - Type features: 30 dims (card types + creature types)
    - Stats features: 8 dims (P/T, loyalty, CMC context)
    - Text embedding: 128 dims (from language model, when available)

    Total: 241 dims (or 113 without text embedding)
    """

    def __init__(self, use_text_embeddings: bool = False, text_embedding_dim: int = 128):
        self.use_text_embeddings = use_text_embeddings
        self.text_embedding_dim = text_embedding_dim
        self.simple_keyword_to_idx = {kw: i for i, kw in enumerate(SIMPLE_KEYWORD_LIST)}

        # Cache for text embeddings
        self._text_embedding_cache: Dict[str, np.ndarray] = {}
        self._sentence_transformer = None

        # Dimension breakdown
        self.keyword_dim = len(SIMPLE_KEYWORD_LIST) + 10  # Simple + amount keywords
        self.mana_dim = 11
        self.type_dim = 10 + 20  # Base types + creature types
        self.stats_dim = 8
        self.text_dim = text_embedding_dim if use_text_embeddings else 0

        self.total_dim = self.keyword_dim + self.mana_dim + self.type_dim + self.stats_dim + self.text_dim

    def _get_text_embedding(self, card_text: str) -> np.ndarray:
        """Get text embedding for card oracle text."""
        if not self.use_text_embeddings:
            return np.zeros(0, dtype=np.float32)

        # Cache lookup
        text_hash = hashlib.md5(card_text.encode()).hexdigest()
        if text_hash in self._text_embedding_cache:
            return self._text_embedding_cache[text_hash]

        # Lazy load sentence transformer
        if self._sentence_transformer is None:
            try:
                from sentence_transformers import SentenceTransformer
                # Use a small, fast model
                self._sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                print("Warning: sentence-transformers not installed. Using zero embeddings.")
                self._sentence_transformer = "unavailable"

        if self._sentence_transformer == "unavailable":
            embedding = np.zeros(self.text_embedding_dim, dtype=np.float32)
        else:
            # Get embedding and reduce dimensions if needed
            full_embedding = self._sentence_transformer.encode(card_text)
            if len(full_embedding) > self.text_embedding_dim:
                # Simple truncation (PCA would be better for production)
                embedding = full_embedding[:self.text_embedding_dim]
            else:
                embedding = np.pad(full_embedding, (0, self.text_embedding_dim - len(full_embedding)))
            embedding = embedding.astype(np.float32)

        self._text_embedding_cache[text_hash] = embedding
        return embedding

    def embed_card(
        self,
        name: str,
        mana_cost: str,
        type_line: str,
        oracle_text: str = "",
        power: Optional[int] = None,
        toughness: Optional[int] = None,
        loyalty: Optional[int] = None,
        keywords: Optional[List[str]] = None,
        cmc: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate embedding vector for a card.

        Args:
            name: Card name
            mana_cost: Mana cost string like "{2}{W}{W}"
            type_line: Type line like "Creature - Human Soldier"
            oracle_text: Card rules text
            power: Creature power (None if not creature)
            toughness: Creature toughness
            loyalty: Planeswalker loyalty
            keywords: List of keyword strings
            cmc: Converted mana cost (calculated from mana_cost if not provided)

        Returns:
            numpy array of shape (total_dim,)
        """
        features = []

        # 1. Keyword features
        keyword_features = np.zeros(self.keyword_dim, dtype=np.float32)

        if keywords:
            for kw_str in keywords:
                parsed = parse_keyword(kw_str)

                # Simple keywords: one-hot
                if parsed.category == KeywordCategory.SIMPLE:
                    for variant in [parsed.name, parsed.name.lower(), parsed.name.title()]:
                        if variant in self.simple_keyword_to_idx:
                            idx = self.simple_keyword_to_idx[variant]
                            keyword_features[idx] = 1.0
                            break

                # Keywords with amount: encode in extra slots
                elif parsed.category == KeywordCategory.WITH_AMOUNT and parsed.amount:
                    # Use extra slots for amount keywords
                    base_idx = len(SIMPLE_KEYWORD_LIST)
                    if parsed.name == "Toxic":
                        keyword_features[base_idx] = parsed.amount / 5.0
                    elif parsed.name == "Annihilator":
                        keyword_features[base_idx + 1] = parsed.amount / 6.0
                    elif parsed.name == "Crew":
                        keyword_features[base_idx + 2] = parsed.amount / 10.0
                    elif parsed.name == "Fabricate":
                        keyword_features[base_idx + 3] = parsed.amount / 5.0
                    elif parsed.name == "Modular":
                        keyword_features[base_idx + 4] = parsed.amount / 5.0
                    elif parsed.name == "Afflict":
                        keyword_features[base_idx + 5] = parsed.amount / 5.0
                    elif parsed.name == "Bushido":
                        keyword_features[base_idx + 6] = parsed.amount / 5.0
                    elif parsed.name == "Renown":
                        keyword_features[base_idx + 7] = parsed.amount / 5.0
                    elif parsed.name == "Dredge":
                        keyword_features[base_idx + 8] = parsed.amount / 6.0
                    else:
                        keyword_features[base_idx + 9] = parsed.amount / 5.0  # Generic amount slot

        features.append(keyword_features)

        # 2. Mana features
        mana_vec = parse_mana_cost(mana_cost)
        features.append(mana_vec.to_vector())

        # 3. Type features
        type_vec = parse_card_types(type_line)
        features.append(type_vec.to_vector())

        # 4. Stats features
        stats = np.zeros(self.stats_dim, dtype=np.float32)
        if power is not None:
            stats[0] = power / 15.0
        if toughness is not None:
            stats[1] = toughness / 15.0
        if loyalty is not None:
            stats[2] = loyalty / 10.0
        if cmc is not None:
            stats[3] = cmc / 15.0
        else:
            stats[3] = mana_vec.total_cmc / 15.0

        # Power/toughness ratio (for creature evaluation)
        if power is not None and toughness is not None and toughness > 0:
            stats[4] = power / toughness  # Attack/defense ratio

        # Efficiency metrics
        if mana_vec.total_cmc > 0:
            if power is not None:
                stats[5] = power / mana_vec.total_cmc  # Power per mana
            if power is not None and toughness is not None:
                stats[6] = (power + toughness) / mana_vec.total_cmc  # Stats per mana

        # Is multicolor?
        color_count = sum([mana_vec.white > 0, mana_vec.blue > 0, mana_vec.black > 0,
                         mana_vec.red > 0, mana_vec.green > 0])
        stats[7] = color_count / 5.0

        features.append(stats)

        # 5. Text embedding (if enabled)
        if self.use_text_embeddings and oracle_text:
            text_emb = self._get_text_embedding(oracle_text)
            features.append(text_emb)

        return np.concatenate(features)

    def embed_from_game_state(self, card_data: Dict) -> np.ndarray:
        """
        Embed a card from game state JSON.

        Args:
            card_data: Dict with keys like 'name', 'mana_cost', 'types', 'power', 'toughness', 'keywords'

        Returns:
            numpy array embedding
        """
        return self.embed_card(
            name=card_data.get('name', ''),
            mana_cost=card_data.get('mana_cost', ''),
            type_line=card_data.get('types', ''),
            oracle_text=card_data.get('oracle_text', ''),
            power=card_data.get('power'),
            toughness=card_data.get('toughness'),
            loyalty=card_data.get('loyalty'),
            keywords=card_data.get('keywords', []),
            cmc=card_data.get('cmc'),
        )


# =============================================================================
# GAME CONTEXT FEATURES
# =============================================================================

def embed_battlefield_card(card_data: Dict, embedder: CardEmbedding) -> np.ndarray:
    """
    Embed a card on the battlefield with game context.

    Additional context features beyond base card embedding:
    - Tapped state
    - Summoning sickness
    - Current damage
    - Counters
    - Is attacking/blocking (if in combat)
    """
    # Base card embedding
    base_embedding = embedder.embed_from_game_state(card_data)

    # Context features
    context = np.zeros(8, dtype=np.float32)
    context[0] = float(card_data.get('tapped', False))
    context[1] = float(card_data.get('summoning_sick', False))
    context[2] = card_data.get('damage', 0) / 10.0

    # Counter types (simplified)
    counters = card_data.get('counters', '')
    if '+1/+1' in counters:
        # Extract count
        match = re.search(r'(\d+) \+1/\+1', counters)
        context[3] = int(match.group(1)) / 10.0 if match else 0.1
    if '-1/-1' in counters:
        match = re.search(r'(\d+) -1/-1', counters)
        context[4] = int(match.group(1)) / 10.0 if match else 0.1

    # Is creature/can attack/can block
    context[5] = float(card_data.get('is_creature', False))
    context[6] = float(card_data.get('is_creature', False) and not card_data.get('tapped', False)
                       and not card_data.get('summoning_sick', False))
    context[7] = float(card_data.get('is_creature', False) and not card_data.get('tapped', False))

    return np.concatenate([base_embedding, context])


# =============================================================================
# TESTING
# =============================================================================

def test_embeddings():
    """Test the embedding system with sample cards."""
    embedder = CardEmbedding(use_text_embeddings=False)

    print(f"Embedding dimension: {embedder.total_dim}")
    print()

    # Test cards
    test_cards = [
        {
            "name": "Lightning Bolt",
            "mana_cost": "{R}",
            "types": "Instant",
            "oracle_text": "Lightning Bolt deals 3 damage to any target.",
            "keywords": [],
        },
        {
            "name": "Tarmogoyf",
            "mana_cost": "{1}{G}",
            "types": "Creature - Lhurgoyf",
            "power": 0, "toughness": 1,
            "oracle_text": "Tarmogoyf's power is equal to the number of card types among cards in all graveyards and its toughness is equal to that number plus 1.",
            "keywords": [],
        },
        {
            "name": "Ragavan, Nimble Pilferer",
            "mana_cost": "{R}",
            "types": "Legendary Creature - Monkey Pirate",
            "power": 2, "toughness": 1,
            "keywords": ["Dash {1}{R}"],
        },
        {
            "name": "Sheoldred, the Apocalypse",
            "mana_cost": "{2}{B}{B}",
            "types": "Legendary Creature - Phyrexian Praetor",
            "power": 4, "toughness": 5,
            "keywords": ["Deathtouch"],
        },
        {
            "name": "Counterspell",
            "mana_cost": "{U}{U}",
            "types": "Instant",
            "keywords": [],
        },
        {
            "name": "Toxic Deluge",
            "mana_cost": "{2}{B}",
            "types": "Sorcery",
            "keywords": [],
        },
        {
            "name": "Atraxa, Grand Unifier",
            "mana_cost": "{3}{G}{W}{U}{B}",
            "types": "Legendary Creature - Phyrexian Angel",
            "power": 7, "toughness": 7,
            "keywords": ["Flying", "Vigilance", "Deathtouch", "Lifelink"],
        },
        {
            "name": "Blightsteel Colossus",
            "mana_cost": "{12}",
            "types": "Artifact Creature - Phyrexian Golem",
            "power": 11, "toughness": 11,
            "keywords": ["Trample", "Infect", "Indestructible"],
        },
    ]

    embeddings = []
    for card in test_cards:
        emb = embedder.embed_from_game_state(card)
        embeddings.append(emb)
        print(f"{card['name']}: shape={emb.shape}, norm={np.linalg.norm(emb):.2f}")

    # Compute similarity matrix
    print("\nSimilarity matrix (cosine):")
    embeddings = np.array(embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    similarity = normalized @ normalized.T

    names = [c['name'][:15] for c in test_cards]
    print(f"{'':>16}", end="")
    for name in names:
        print(f"{name:>16}", end="")
    print()

    for i, name in enumerate(names):
        print(f"{name:>16}", end="")
        for j in range(len(names)):
            print(f"{similarity[i,j]:>16.2f}", end="")
        print()


if __name__ == "__main__":
    test_embeddings()
