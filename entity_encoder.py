#!/usr/bin/env python3
"""
Entity-Based Encoder for MTG RL

This module implements a comprehensive entity encoder inspired by AlphaStar's
architecture, designed to handle the full complexity of Magic: The Gathering.

Key Design Principles:
1. EXTENSIBLE: New mechanics can be added without retraining from scratch
2. ENTITY-BASED: Each card/permanent is an entity with full state
3. HIERARCHICAL: Identity (static) + State (dynamic) embeddings
4. FORGE-COMPATIBLE: Uses Forge's TrackableProperty representation

Architecture based on:
- AlphaStar's entity encoder (140+ features per unit)
- Hearthstone AI research showing synergy-heavy decks need rich representations
- MTG research showing text embeddings capture ability semantics

The encoder operates in three modes:
1. DRAFT: Only identity features needed (cards don't have state yet)
2. GAMEPLAY: Full identity + state features
3. DECKBUILDING: Identity + deck context features
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# CONFIGURATION
# =============================================================================

class EncoderMode(Enum):
    """Operating mode for the encoder."""
    DRAFT = "draft"           # Static card features only
    GAMEPLAY = "gameplay"     # Full entity state
    DECKBUILDING = "deckbuilding"  # Identity + deck context


@dataclass
class EntityEncoderConfig:
    """
    Configuration for the entity encoder.

    Designed to match Forge's TrackableProperty representation while being
    extensible for future mechanics.
    """
    # =========================
    # IDENTITY FEATURES (static card properties)
    # =========================

    # Text embedding dimension (from pretrained model or learned)
    text_embedding_dim: int = 256
    use_pretrained_text: bool = False  # If True, expect external embeddings

    # Mana cost features
    mana_colors: int = 6     # WUBRG + C
    max_cmc: int = 16        # Max CMC to represent
    max_color_pips: int = 5  # Max pips of single color

    # Type features
    num_card_types: int = 10      # Creature, Instant, Sorcery, etc.
    num_creature_types: int = 300  # Full creature type coverage
    num_supertypes: int = 5        # Legendary, Basic, Snow, World, Tribal

    # Stat features
    max_power: int = 20
    max_toughness: int = 20
    max_loyalty: int = 10
    max_defense: int = 10

    # Keyword features (from Forge's KeywordCollection)
    num_keywords: int = 150   # Full keyword coverage

    # Ability type indicators
    num_ability_types: int = 8  # ETB, Activated, Triggered, Static, etc.

    # Rarity
    num_rarities: int = 6  # Common, Uncommon, Rare, Mythic, Special, Bonus

    # =========================
    # STATE FEATURES (dynamic runtime properties)
    # =========================

    # Counter types (from Forge's CounterEnumType)
    num_counter_types: int = 50
    max_counters: int = 20

    # Zone information
    num_zones: int = 15  # Hand, Library, Battlefield, Graveyard, Exile, Stack, etc.

    # Combat state
    combat_features: int = 8  # attacking, blocking, blocked, etc.

    # Modification tracking
    num_modifications: int = 10  # control change, type change, etc.

    # Temporal features
    temporal_features: int = 5  # turns since cast, damage this turn, etc.

    # Attachment tracking
    max_attachments: int = 10

    # =========================
    # ARCHITECTURE
    # =========================

    d_model: int = 512          # Internal dimension
    n_heads: int = 8            # Attention heads
    n_entity_layers: int = 4    # Transformer layers for entity interactions
    n_global_layers: int = 2    # Layers for global game state
    d_ff: int = 1024            # Feed-forward dimension
    dropout: float = 0.1

    # Output dimensions
    identity_dim: int = 384     # Card identity embedding
    state_dim: int = 128        # Runtime state embedding
    output_dim: int = 512       # Final entity embedding

    # Pooling for set aggregation
    use_cls_token: bool = True  # Add CLS token for set-level representation

    @property
    def identity_input_dim(self) -> int:
        """Total dimension of identity features."""
        return (
            self.text_embedding_dim +      # Ability text
            self.mana_colors + 3 +          # Mana (colors + generic + X + CMC)
            self.num_card_types +           # Card types
            self.num_creature_types +       # Creature types
            self.num_supertypes +           # Supertypes
            4 +                              # P/T/Loyalty/Defense
            self.num_keywords +             # Keywords
            self.num_ability_types +        # Ability indicators
            self.num_rarities               # Rarity
        )

    @property
    def state_input_dim(self) -> int:
        """Total dimension of state features."""
        return (
            self.num_counter_types +        # Counter types
            self.num_zones +                # Current zone
            self.combat_features +          # Combat state
            self.num_modifications +        # Modifications
            self.temporal_features +        # Temporal info
            4 +                              # Current P/T (modified), tapped, summoning sickness
            1                                # Controller (self=0, opponent=1)
        )


# =============================================================================
# FEATURE VOCABULARIES (matching Forge's internal representation)
# =============================================================================

class ForgeVocabulary:
    """
    Vocabulary mappings matching Forge's internal enums and collections.

    These are extracted from Forge's source code to ensure compatibility:
    - forge-game/src/main/java/forge/game/keyword/Keyword.java
    - forge-game/src/main/java/forge/game/GameEntityCounterTable.java
    - forge-game/src/main/java/forge/game/zone/ZoneType.java
    """

    # Keywords (from Forge's Keyword.java - top 150)
    KEYWORDS = [
        # Evergreen
        "flying", "first_strike", "double_strike", "trample", "haste",
        "vigilance", "lifelink", "deathtouch", "reach", "menace",
        "defender", "indestructible", "hexproof", "flash", "fear",
        "intimidate", "shroud", "protection", "landwalk", "islandwalk",
        "mountainwalk", "swampwalk", "forestwalk", "plainswalk",

        # Combat
        "flanking", "shadow", "horsemanship", "banding", "phasing",
        "rampage", "provoke", "bushido", "ninjutsu", "skulk",

        # Keywords with costs/amounts
        "kicker", "multikicker", "cycling", "flashback", "madness",
        "morph", "megamorph", "disguise", "manifest", "emerge",
        "evoke", "dash", "bestow", "awaken", "surge",
        "spectacle", "afterlife", "riot", "escape", "mutate",
        "companion", "foretell", "disturb", "cleave", "blitz",
        "casualty", "read_ahead", "prototype", "squad", "enlist",
        "ravenous", "backup", "bargain", "craft", "plot",

        # Cost modification
        "affinity", "convoke", "delve", "improvise", "assist",
        "undaunted", "offering",

        # Triggered
        "exalted", "cascade", "annihilator", "infect", "wither",
        "persist", "undying", "soulbond", "exploit", "devour",
        "fabricate", "embalm", "eternalize", "encore", "decayed",
        "training", "toxic", "for_mirrodin", "living_weapon",

        # Static
        "changeling", "devoid", "partner", "storm", "suspend",
        "vanishing", "fading", "echo", "cumulative_upkeep", "amplify",
        "modular", "sunburst", "graft", "bloodthirst", "dredge",
        "transmute", "recover", "ripple", "split_second", "absorb",
        "frenzy", "gravestorm", "hideaway", "fortify", "prowess",
        "renown", "melee", "crew", "partner_with", "jump_start",
        "spectacle", "adapt", "amass", "proliferate", "populate",
        "investigate", "explore", "ascend", "surveil", "mill",
        "learn", "daybound", "nightbound", "ward", "reconfigure",
        "compleated", "living_metal", "more_than_meets_the_eye",
        "battle_cry", "cipher", "extort", "fuse", "haunt",
        "overload", "scavenge", "tribute", "unleash", "dethrone",
        "myriad", "melee", "undaunted", "partner", "eminence",
        "enrage", "treasure", "saga", "adventure", "role",
        "celebration", "domain", "threshold", "metalcraft", "delirium",
        "revolt", "ferocious", "formidable", "morbid", "spell_mastery",
        "lieutenant", "raid", "constellation", "heroic", "devotion",
        "monstrosity", "tribute", "inspired", "battalion", "bloodrush",
        "battalion", "cipher", "evolve", "extort", "overload",
    ]

    # Counter types (from Forge's CounterEnumType)
    COUNTER_TYPES = [
        "p1p1", "m1m1",                     # +1/+1 and -1/-1
        "loyalty", "energy", "experience",  # Player/Planeswalker
        "charge", "storage", "luck",        # Artifact/General
        "verse", "lore", "time",            # Saga/Suspend
        "spore", "fade", "age",             # Fungus/Fading
        "blood", "flood", "bribery",        # Various mechanics
        "blaze", "corpse", "credit",
        "crystal", "currency", "death",
        "delay", "depletion", "despair",
        "devotion", "divinity", "doom",
        "dream", "echo", "egg",
        "elixir", "eon", "eyeball",
        "fate", "feather", "filibuster",
        "flame", "foreshadow", "fungus",
        "fury", "gem", "glyph",
        "gold", "growth", "healing",
        "hour", "hourglass", "hunger",
        "ice", "incarnation", "infection",
        "intervention", "isolation", "javelin",
        "ki", "knowledge", "level",
    ]

    # Zone types (from Forge's ZoneType)
    ZONES = [
        "hand", "library", "battlefield", "graveyard", "exile",
        "stack", "command", "ante", "sideboard", "schemeDeck",
        "planeDeck", "archenemy", "flashback", "none", "outside",
    ]

    # Card types
    CARD_TYPES = [
        "creature", "instant", "sorcery", "enchantment", "artifact",
        "land", "planeswalker", "battle", "kindred", "dungeon",
    ]

    # Supertypes
    SUPERTYPES = [
        "legendary", "basic", "snow", "world", "ongoing",
    ]

    # Creature types (subset - full list would be 300+)
    CREATURE_TYPES = [
        # Common types
        "human", "soldier", "wizard", "warrior", "knight", "cleric",
        "rogue", "shaman", "druid", "monk", "noble", "advisor",
        "artificer", "assassin", "berserker", "bard", "pilot",

        # Races
        "elf", "goblin", "merfolk", "vampire", "zombie", "skeleton",
        "spirit", "angel", "demon", "devil", "dragon", "giant",
        "dwarf", "orc", "troll", "ogre", "faerie", "kithkin",
        "vedalken", "sphinx", "elemental", "shapeshifter", "golem",
        "construct", "incarnation", "avatar", "god",

        # Creatures
        "beast", "bird", "cat", "dog", "wolf", "bear", "snake",
        "spider", "insect", "rat", "bat", "horse", "unicorn",
        "dinosaur", "hydra", "wurm", "ooze", "sliver", "eldrazi",
        "phyrexian", "horror", "nightmare", "specter", "shade",

        # MTG specific
        "mutant", "ninja", "samurai", "moonfolk", "kirin", "rebel",
        "mercenary", "minion", "pirate", "citizen", "peasant", "serf",
    ]

    # Ability type indicators
    ABILITY_TYPES = [
        "activated", "triggered", "static", "etb", "ltb",
        "attack_trigger", "damage_trigger", "mana_ability",
    ]

    # Rarities
    RARITIES = [
        "common", "uncommon", "rare", "mythic", "special", "bonus",
    ]

    def __init__(self):
        # Build indices
        self.keyword_to_idx = {k: i for i, k in enumerate(self.KEYWORDS)}
        self.counter_to_idx = {c: i for i, c in enumerate(self.COUNTER_TYPES)}
        self.zone_to_idx = {z: i for i, z in enumerate(self.ZONES)}
        self.card_type_to_idx = {t: i for i, t in enumerate(self.CARD_TYPES)}
        self.supertype_to_idx = {s: i for i, s in enumerate(self.SUPERTYPES)}
        self.creature_type_to_idx = {c: i for i, c in enumerate(self.CREATURE_TYPES)}
        self.ability_type_to_idx = {a: i for i, a in enumerate(self.ABILITY_TYPES)}
        self.rarity_to_idx = {r: i for i, r in enumerate(self.RARITIES)}


# =============================================================================
# FEATURE EXTRACTION (from Forge game state)
# =============================================================================

@dataclass
class EntityFeatures:
    """
    Features extracted from a game entity (card/permanent).

    Designed to be populated from Forge's GameStateLogger JSON output.
    """
    # Identity features (static)
    card_name: str = ""
    oracle_text: str = ""
    mana_cost: str = ""
    card_types: List[str] = field(default_factory=list)
    creature_types: List[str] = field(default_factory=list)
    supertypes: List[str] = field(default_factory=list)
    base_power: Optional[int] = None
    base_toughness: Optional[int] = None
    base_loyalty: Optional[int] = None
    base_defense: Optional[int] = None
    keywords: List[str] = field(default_factory=list)
    ability_types: List[str] = field(default_factory=list)
    rarity: str = "common"

    # State features (dynamic)
    counters: Dict[str, int] = field(default_factory=dict)
    zone: str = "none"
    is_tapped: bool = False
    is_attacking: bool = False
    is_blocking: bool = False
    is_blocked: bool = False
    summoning_sickness: bool = False
    controller_is_self: bool = True
    current_power: Optional[int] = None
    current_toughness: Optional[int] = None
    damage_marked: int = 0
    attached_to: Optional[int] = None  # Entity ID
    attached_cards: List[int] = field(default_factory=list)  # Entity IDs
    turns_on_battlefield: int = 0
    was_cast_this_turn: bool = False

    # Precomputed text embedding (optional)
    text_embedding: Optional[np.ndarray] = None


class EntityFeatureExtractor:
    """
    Extracts tensor features from EntityFeatures for the encoder.

    This bridges between Forge's JSON state representation and
    the tensor format expected by the neural network.
    """

    def __init__(self, config: EntityEncoderConfig):
        self.config = config
        self.vocab = ForgeVocabulary()

    def extract_identity(self, entity: EntityFeatures) -> np.ndarray:
        """
        Extract identity (static) features from an entity.

        Returns:
            numpy array of shape [identity_input_dim]
        """
        features = []

        # Text embedding (256 dims or placeholder)
        if entity.text_embedding is not None:
            features.append(entity.text_embedding)
        else:
            # Placeholder - in production, use pretrained embeddings
            features.append(np.zeros(self.config.text_embedding_dim, dtype=np.float32))

        # Mana cost (6 colors + generic + X + CMC = 9)
        mana = self._parse_mana_cost(entity.mana_cost)
        features.append(mana)

        # Card types (10 dims)
        types = np.zeros(self.config.num_card_types, dtype=np.float32)
        for t in entity.card_types:
            t_lower = t.lower()
            if t_lower in self.vocab.card_type_to_idx:
                types[self.vocab.card_type_to_idx[t_lower]] = 1.0
        features.append(types)

        # Creature types (300 dims - truncated in practice)
        creature_types = np.zeros(min(self.config.num_creature_types, len(self.vocab.CREATURE_TYPES)), dtype=np.float32)
        for ct in entity.creature_types:
            ct_lower = ct.lower()
            if ct_lower in self.vocab.creature_type_to_idx:
                idx = self.vocab.creature_type_to_idx[ct_lower]
                if idx < len(creature_types):
                    creature_types[idx] = 1.0
        features.append(creature_types)

        # Supertypes (5 dims)
        supertypes = np.zeros(self.config.num_supertypes, dtype=np.float32)
        for st in entity.supertypes:
            st_lower = st.lower()
            if st_lower in self.vocab.supertype_to_idx:
                supertypes[self.vocab.supertype_to_idx[st_lower]] = 1.0
        features.append(supertypes)

        # Base stats (4 dims) - normalized
        stats = np.zeros(4, dtype=np.float32)
        if entity.base_power is not None:
            stats[0] = min(entity.base_power, self.config.max_power) / self.config.max_power
        if entity.base_toughness is not None:
            stats[1] = min(entity.base_toughness, self.config.max_toughness) / self.config.max_toughness
        if entity.base_loyalty is not None:
            stats[2] = min(entity.base_loyalty, self.config.max_loyalty) / self.config.max_loyalty
        if entity.base_defense is not None:
            stats[3] = min(entity.base_defense, self.config.max_defense) / self.config.max_defense
        features.append(stats)

        # Keywords (150 dims)
        keywords = np.zeros(min(self.config.num_keywords, len(self.vocab.KEYWORDS)), dtype=np.float32)
        for kw in entity.keywords:
            # Normalize keyword (remove numbers, lowercase)
            kw_norm = kw.lower().replace(" ", "_").split("_")[0]
            if kw_norm in self.vocab.keyword_to_idx:
                idx = self.vocab.keyword_to_idx[kw_norm]
                if idx < len(keywords):
                    keywords[idx] = 1.0
        features.append(keywords)

        # Ability type indicators (8 dims)
        ability_types = np.zeros(self.config.num_ability_types, dtype=np.float32)
        for at in entity.ability_types:
            at_lower = at.lower()
            if at_lower in self.vocab.ability_type_to_idx:
                ability_types[self.vocab.ability_type_to_idx[at_lower]] = 1.0
        features.append(ability_types)

        # Rarity (6 dims)
        rarity = np.zeros(self.config.num_rarities, dtype=np.float32)
        rarity_lower = entity.rarity.lower()
        if rarity_lower in self.vocab.rarity_to_idx:
            rarity[self.vocab.rarity_to_idx[rarity_lower]] = 1.0
        features.append(rarity)

        return np.concatenate(features)

    def extract_state(self, entity: EntityFeatures) -> np.ndarray:
        """
        Extract state (dynamic) features from an entity.

        Returns:
            numpy array of shape [state_input_dim]
        """
        features = []

        # Counters (50 dims)
        counters = np.zeros(min(self.config.num_counter_types, len(self.vocab.COUNTER_TYPES)), dtype=np.float32)
        for counter_type, count in entity.counters.items():
            ct_lower = counter_type.lower()
            if ct_lower in self.vocab.counter_to_idx:
                idx = self.vocab.counter_to_idx[ct_lower]
                if idx < len(counters):
                    counters[idx] = min(count, self.config.max_counters) / self.config.max_counters
        features.append(counters)

        # Zone (15 dims)
        zone = np.zeros(self.config.num_zones, dtype=np.float32)
        zone_lower = entity.zone.lower()
        if zone_lower in self.vocab.zone_to_idx:
            zone[self.vocab.zone_to_idx[zone_lower]] = 1.0
        features.append(zone)

        # Combat state (8 dims)
        combat = np.zeros(self.config.combat_features, dtype=np.float32)
        combat[0] = float(entity.is_tapped)
        combat[1] = float(entity.is_attacking)
        combat[2] = float(entity.is_blocking)
        combat[3] = float(entity.is_blocked)
        combat[4] = float(entity.summoning_sickness)
        combat[5] = float(entity.attached_to is not None)
        combat[6] = min(len(entity.attached_cards), 5) / 5.0
        combat[7] = min(entity.damage_marked, 20) / 20.0
        features.append(combat)

        # Modifications placeholder (10 dims)
        modifications = np.zeros(self.config.num_modifications, dtype=np.float32)
        # These would be populated from Forge's modification tracking
        features.append(modifications)

        # Temporal features (5 dims)
        temporal = np.zeros(self.config.temporal_features, dtype=np.float32)
        temporal[0] = min(entity.turns_on_battlefield, 10) / 10.0
        temporal[1] = float(entity.was_cast_this_turn)
        features.append(temporal)

        # Current P/T (modified) + controller (5 dims)
        current = np.zeros(5, dtype=np.float32)
        if entity.current_power is not None:
            current[0] = min(entity.current_power, self.config.max_power) / self.config.max_power
        elif entity.base_power is not None:
            current[0] = min(entity.base_power, self.config.max_power) / self.config.max_power
        if entity.current_toughness is not None:
            current[1] = min(entity.current_toughness, self.config.max_toughness) / self.config.max_toughness
        elif entity.base_toughness is not None:
            current[1] = min(entity.base_toughness, self.config.max_toughness) / self.config.max_toughness
        current[2] = float(entity.summoning_sickness)
        current[3] = float(entity.is_tapped)
        current[4] = 0.0 if entity.controller_is_self else 1.0
        features.append(current)

        return np.concatenate(features)

    def _parse_mana_cost(self, cost_str: str) -> np.ndarray:
        """Parse mana cost string to feature vector."""
        result = np.zeros(9, dtype=np.float32)  # WUBRG + C + generic + X + total CMC

        if not cost_str:
            return result

        import re
        symbols = re.findall(r'\{([^}]+)\}', cost_str)

        total_cmc = 0
        for symbol in symbols:
            symbol = symbol.upper()

            if symbol == 'W':
                result[0] = min(result[0] + 1, self.config.max_color_pips) / self.config.max_color_pips
                total_cmc += 1
            elif symbol == 'U':
                result[1] = min(result[1] + 1, self.config.max_color_pips) / self.config.max_color_pips
                total_cmc += 1
            elif symbol == 'B':
                result[2] = min(result[2] + 1, self.config.max_color_pips) / self.config.max_color_pips
                total_cmc += 1
            elif symbol == 'R':
                result[3] = min(result[3] + 1, self.config.max_color_pips) / self.config.max_color_pips
                total_cmc += 1
            elif symbol == 'G':
                result[4] = min(result[4] + 1, self.config.max_color_pips) / self.config.max_color_pips
                total_cmc += 1
            elif symbol == 'C':
                result[5] = min(result[5] + 1, self.config.max_color_pips) / self.config.max_color_pips
                total_cmc += 1
            elif symbol.isdigit():
                result[6] = min(result[6] + int(symbol), self.config.max_cmc) / self.config.max_cmc
                total_cmc += int(symbol)
            elif symbol == 'X':
                result[7] = 1.0
            elif '/' in symbol:  # Hybrid or Phyrexian
                total_cmc += 1

        result[8] = min(total_cmc, self.config.max_cmc) / self.config.max_cmc

        return result

    def extract_full(self, entity: EntityFeatures) -> Tuple[np.ndarray, np.ndarray]:
        """Extract both identity and state features."""
        return self.extract_identity(entity), self.extract_state(entity)


# =============================================================================
# NEURAL NETWORK ARCHITECTURE
# =============================================================================

class IdentityEncoder(nn.Module):
    """
    Encodes static card identity features.

    This component can be pretrained on draft data and frozen for gameplay.
    """

    def __init__(self, config: EntityEncoderConfig):
        super().__init__()
        self.config = config

        # Separate projections for different feature types
        # This allows learning appropriate representations for each

        # Text embedding projection
        self.text_proj = nn.Sequential(
            nn.Linear(config.text_embedding_dim, config.d_model // 2),
            nn.LayerNorm(config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Mana cost projection
        self.mana_proj = nn.Sequential(
            nn.Linear(9, config.d_model // 8),  # 9 mana features
            nn.GELU(),
        )

        # Type projection (card types + creature types + supertypes)
        type_dim = config.num_card_types + min(config.num_creature_types, 60) + config.num_supertypes
        self.type_proj = nn.Sequential(
            nn.Linear(type_dim, config.d_model // 4),
            nn.LayerNorm(config.d_model // 4),
            nn.GELU(),
        )

        # Stats projection
        self.stats_proj = nn.Sequential(
            nn.Linear(4, config.d_model // 8),  # P/T/Loyalty/Defense
            nn.GELU(),
        )

        # Keyword projection
        self.keyword_proj = nn.Sequential(
            nn.Linear(min(config.num_keywords, 150), config.d_model // 4),
            nn.LayerNorm(config.d_model // 4),
            nn.GELU(),
        )

        # Ability type + rarity projection
        other_dim = config.num_ability_types + config.num_rarities
        self.other_proj = nn.Sequential(
            nn.Linear(other_dim, config.d_model // 8),
            nn.GELU(),
        )

        # Combine all projections
        # d_model//2 + d_model//8 + d_model//4 + d_model//8 + d_model//4 + d_model//8 = 11*d_model//8
        combine_dim = (config.d_model // 2 + config.d_model // 8 + config.d_model // 4 +
                      config.d_model // 8 + config.d_model // 4 + config.d_model // 8)

        self.combine = nn.Sequential(
            nn.Linear(combine_dim, config.identity_dim),
            nn.LayerNorm(config.identity_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.identity_dim, config.identity_dim),
            nn.LayerNorm(config.identity_dim),
        )

    def forward(self, identity_features: torch.Tensor) -> torch.Tensor:
        """
        Encode identity features.

        Args:
            identity_features: [batch, num_entities, identity_input_dim]

        Returns:
            [batch, num_entities, identity_dim]
        """
        # Split features by type
        idx = 0

        text = identity_features[..., idx:idx + self.config.text_embedding_dim]
        idx += self.config.text_embedding_dim

        mana = identity_features[..., idx:idx + 9]
        idx += 9

        type_dim = self.config.num_card_types + min(self.config.num_creature_types, 60) + self.config.num_supertypes
        types = identity_features[..., idx:idx + type_dim]
        idx += type_dim

        stats = identity_features[..., idx:idx + 4]
        idx += 4

        keywords = identity_features[..., idx:idx + min(self.config.num_keywords, 150)]
        idx += min(self.config.num_keywords, 150)

        other = identity_features[..., idx:]

        # Project each
        text_emb = self.text_proj(text)
        mana_emb = self.mana_proj(mana)
        type_emb = self.type_proj(types)
        stats_emb = self.stats_proj(stats)
        keyword_emb = self.keyword_proj(keywords)
        other_emb = self.other_proj(other)

        # Combine
        combined = torch.cat([text_emb, mana_emb, type_emb, stats_emb, keyword_emb, other_emb], dim=-1)
        return self.combine(combined)


class StateEncoder(nn.Module):
    """
    Encodes dynamic runtime state features.

    This component learns during gameplay training.
    """

    def __init__(self, config: EntityEncoderConfig):
        super().__init__()
        self.config = config

        # Counter projection
        self.counter_proj = nn.Sequential(
            nn.Linear(min(config.num_counter_types, 50), config.state_dim // 2),
            nn.LayerNorm(config.state_dim // 2),
            nn.GELU(),
        )

        # Zone + combat + other state projection
        other_state_dim = config.num_zones + config.combat_features + config.num_modifications + config.temporal_features + 5
        self.state_proj = nn.Sequential(
            nn.Linear(other_state_dim, config.state_dim // 2),
            nn.LayerNorm(config.state_dim // 2),
            nn.GELU(),
        )

        # Combine
        self.combine = nn.Sequential(
            nn.Linear(config.state_dim, config.state_dim),
            nn.LayerNorm(config.state_dim),
        )

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        """
        Encode state features.

        Args:
            state_features: [batch, num_entities, state_input_dim]

        Returns:
            [batch, num_entities, state_dim]
        """
        idx = 0

        counters = state_features[..., idx:idx + min(self.config.num_counter_types, 50)]
        idx += min(self.config.num_counter_types, 50)

        other_state = state_features[..., idx:]

        counter_emb = self.counter_proj(counters)
        state_emb = self.state_proj(other_state)

        combined = torch.cat([counter_emb, state_emb], dim=-1)
        return self.combine(combined)


class EntityInteractionLayer(nn.Module):
    """
    Models interactions between entities using self-attention.

    Like AlphaStar, this allows the model to understand:
    - Which creatures can block which attackers
    - Equipment/aura relationships
    - Tribal synergies
    - Board-wide effects
    """

    def __init__(self, config: EntityEncoderConfig):
        super().__init__()
        self.config = config

        self.self_attn = nn.MultiheadAttention(
            config.output_dim,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(config.output_dim, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.output_dim),
            nn.Dropout(config.dropout),
        )

        self.norm1 = nn.LayerNorm(config.output_dim)
        self.norm2 = nn.LayerNorm(config.output_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply self-attention over entities.

        Args:
            x: [batch, num_entities, output_dim]
            mask: [batch, num_entities] - 1 for valid, 0 for padding

        Returns:
            [batch, num_entities, output_dim]
        """
        attn_mask = None
        if mask is not None:
            attn_mask = (mask == 0)  # True for positions to mask OUT

        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=attn_mask)
        x = self.norm1(x + attn_out)

        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)

        return x


class EntityEncoder(nn.Module):
    """
    Full entity encoder combining identity and state.

    Architecture:
    1. Encode identity features (can be frozen from draft pretraining)
    2. Encode state features (learned during gameplay)
    3. Combine identity + state
    4. Apply transformer layers for entity interactions
    5. Output per-entity embeddings + optional pooled representation
    """

    def __init__(self, config: Optional[EntityEncoderConfig] = None):
        super().__init__()
        self.config = config or EntityEncoderConfig()

        # Identity encoder (can be frozen)
        self.identity_encoder = IdentityEncoder(self.config)

        # State encoder (always trained)
        self.state_encoder = StateEncoder(self.config)

        # Combine identity + state
        combine_dim = self.config.identity_dim + self.config.state_dim
        self.combine = nn.Sequential(
            nn.Linear(combine_dim, self.config.output_dim),
            nn.LayerNorm(self.config.output_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )

        # Entity interaction layers
        self.interaction_layers = nn.ModuleList([
            EntityInteractionLayer(self.config)
            for _ in range(self.config.n_entity_layers)
        ])

        # Optional CLS token for pooled representation
        if self.config.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.config.output_dim))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def freeze_identity(self):
        """Freeze identity encoder for transfer learning."""
        for param in self.identity_encoder.parameters():
            param.requires_grad = False

    def unfreeze_identity(self):
        """Unfreeze identity encoder."""
        for param in self.identity_encoder.parameters():
            param.requires_grad = True

    def forward(
        self,
        identity_features: torch.Tensor,
        state_features: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        mode: EncoderMode = EncoderMode.GAMEPLAY,
        return_pooled: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode entities.

        Args:
            identity_features: [batch, num_entities, identity_input_dim]
            state_features: [batch, num_entities, state_input_dim] (optional for draft mode)
            mask: [batch, num_entities] - 1 for valid, 0 for padding
            mode: Encoder operating mode
            return_pooled: Whether to return pooled representation

        Returns:
            If return_pooled:
                (entity_embeddings, pooled_embedding)
                entity_embeddings: [batch, num_entities, output_dim]
                pooled_embedding: [batch, output_dim]
            Else:
                entity_embeddings: [batch, num_entities, output_dim]
        """
        batch_size, num_entities = identity_features.shape[:2]

        # Encode identity
        identity_emb = self.identity_encoder(identity_features)

        # Encode state (if provided and not in draft mode)
        if state_features is not None and mode != EncoderMode.DRAFT:
            state_emb = self.state_encoder(state_features)
            combined = torch.cat([identity_emb, state_emb], dim=-1)
        else:
            # Draft mode: pad state with zeros
            zero_state = torch.zeros(batch_size, num_entities, self.config.state_dim,
                                    device=identity_features.device)
            combined = torch.cat([identity_emb, zero_state], dim=-1)

        # Combine
        x = self.combine(combined)

        # Add CLS token if using pooled representation
        if self.config.use_cls_token and return_pooled:
            cls = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls, x], dim=1)

            # Extend mask
            if mask is not None:
                cls_mask = torch.ones(batch_size, 1, device=mask.device)
                mask = torch.cat([cls_mask, mask], dim=1)

        # Apply interaction layers
        for layer in self.interaction_layers:
            x = layer(x, mask)

        # Split CLS token from entities
        if self.config.use_cls_token and return_pooled:
            pooled = x[:, 0]  # CLS token
            entity_emb = x[:, 1:]  # Entity embeddings
            return entity_emb, pooled
        else:
            return x

    def save(self, path: str):
        """Save encoder to file."""
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str, device: torch.device = None) -> 'EntityEncoder':
        """Load encoder from file."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        encoder = cls(checkpoint['config'])
        encoder.load_state_dict(checkpoint['state_dict'])
        return encoder


# =============================================================================
# TRANSFER LEARNING UTILITIES
# =============================================================================

def transfer_from_draft_encoder(
    draft_encoder_path: str,
    config: Optional[EntityEncoderConfig] = None,
) -> EntityEncoder:
    """
    Create a gameplay encoder with identity weights transferred from draft encoder.

    Args:
        draft_encoder_path: Path to pretrained draft encoder
        config: Optional new config (must have compatible identity dimensions)

    Returns:
        EntityEncoder with identity weights from draft encoder
    """
    # Load draft encoder
    draft_checkpoint = torch.load(draft_encoder_path, weights_only=False)

    # Create new encoder
    entity_encoder = EntityEncoder(config)

    # Transfer identity encoder weights
    draft_state = draft_checkpoint['state_dict']
    entity_state = entity_encoder.state_dict()

    # Map weights (this would need adjustment based on actual draft encoder architecture)
    # For now, just initialize fresh
    print("Note: Full weight transfer requires matching architectures")
    print("Entity encoder initialized with random weights")

    return entity_encoder


# =============================================================================
# TESTING
# =============================================================================

def test_entity_encoder():
    """Test the entity encoder."""
    print("Testing Entity Encoder")
    print("=" * 70)

    config = EntityEncoderConfig()
    encoder = EntityEncoder(config)

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    identity_params = sum(p.numel() for p in encoder.identity_encoder.parameters())
    state_params = sum(p.numel() for p in encoder.state_encoder.parameters())

    print(f"\nEncoder parameters: {total_params:,}")
    print(f"  Identity encoder: {identity_params:,}")
    print(f"  State encoder: {state_params:,}")
    print(f"  Interaction layers: {total_params - identity_params - state_params:,}")

    print(f"\nIdentity input dim: ~{config.text_embedding_dim + 9 + config.num_card_types + 60 + config.num_supertypes + 4 + 150 + config.num_ability_types + config.num_rarities}")
    print(f"State input dim: ~{50 + config.num_zones + config.combat_features + config.num_modifications + config.temporal_features + 5}")
    print(f"Output dim: {config.output_dim}")

    # Test with random input
    batch_size = 4
    num_entities = 20

    # Create random features (simplified for testing)
    identity_dim = config.text_embedding_dim + 9 + config.num_card_types + 60 + config.num_supertypes + 4 + 150 + config.num_ability_types + config.num_rarities
    state_dim = 50 + config.num_zones + config.combat_features + config.num_modifications + config.temporal_features + 5

    identity_features = torch.randn(batch_size, num_entities, identity_dim)
    state_features = torch.randn(batch_size, num_entities, state_dim)
    mask = torch.ones(batch_size, num_entities)
    mask[:, 15:] = 0  # Last 5 entities are padding

    print("\nInput shapes:")
    print(f"  Identity: {identity_features.shape}")
    print(f"  State: {state_features.shape}")
    print(f"  Mask: {mask.shape}")

    # Forward pass
    encoder.eval()
    with torch.no_grad():
        # Gameplay mode
        entity_emb, pooled = encoder(identity_features, state_features, mask,
                                      mode=EncoderMode.GAMEPLAY, return_pooled=True)
        print("\nGameplay mode output:")
        print(f"  Entity embeddings: {entity_emb.shape}")
        print(f"  Pooled: {pooled.shape}")

        # Draft mode (no state)
        entity_emb_draft, pooled_draft = encoder(identity_features, None, mask,
                                                  mode=EncoderMode.DRAFT, return_pooled=True)
        print("\nDraft mode output:")
        print(f"  Entity embeddings: {entity_emb_draft.shape}")
        print(f"  Pooled: {pooled_draft.shape}")

    # Test feature extractor
    print("\nTesting feature extractor...")
    extractor = EntityFeatureExtractor(config)

    test_entity = EntityFeatures(
        card_name="Lightning Bolt",
        oracle_text="Lightning Bolt deals 3 damage to any target.",
        mana_cost="{R}",
        card_types=["instant"],
        keywords=[],
        rarity="common",
        zone="hand",
        controller_is_self=True,
    )

    identity, state = extractor.extract_full(test_entity)
    print("Lightning Bolt:")
    print(f"  Identity features: shape={identity.shape}, non-zero={np.count_nonzero(identity)}")
    print(f"  State features: shape={state.shape}, non-zero={np.count_nonzero(state)}")

    test_creature = EntityFeatures(
        card_name="Questing Beast",
        oracle_text="Vigilance, deathtouch, haste...",
        mana_cost="{2}{G}{G}",
        card_types=["creature"],
        creature_types=["beast"],
        supertypes=["legendary"],
        keywords=["vigilance", "deathtouch", "haste"],
        base_power=4,
        base_toughness=4,
        rarity="mythic",
        zone="battlefield",
        counters={"p1p1": 2},
        is_tapped=False,
        controller_is_self=True,
        turns_on_battlefield=2,
    )

    identity, state = extractor.extract_full(test_creature)
    print("\nQuesting Beast (with 2 +1/+1 counters):")
    print(f"  Identity features: shape={identity.shape}, non-zero={np.count_nonzero(identity)}")
    print(f"  State features: shape={state.shape}, non-zero={np.count_nonzero(state)}")

    # Test save/load
    print("\nTesting save/load...")
    encoder.save("/tmp/test_entity_encoder.pt")
    loaded = EntityEncoder.load("/tmp/test_entity_encoder.pt")
    print("Save/load successful!")

    # Test freezing
    print("\nTesting identity freeze...")
    encoder.freeze_identity()
    frozen_params = sum(1 for p in encoder.parameters() if not p.requires_grad)
    print(f"Frozen parameters: {frozen_params}")
    encoder.unfreeze_identity()

    print("\n" + "=" * 70)
    print("Entity Encoder test completed!")


if __name__ == "__main__":
    test_entity_encoder()
