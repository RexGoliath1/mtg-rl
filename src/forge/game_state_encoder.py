"""
Forge Game State Encoder

Converts Forge's JSON game state representation into tensors for the neural network.
Uses pre-computed mechanics embeddings from HDF5 for efficient card encoding.

The encoder handles:
1. Card identity (mechanics-based, loaded from HDF5)
2. Dynamic game state (zone, counters, tapped state, etc.)
3. Global game information (life totals, turn, phase, stack)
4. Action masking for legal moves

Architecture matches AlphaZero-style:
- State representation for policy/value network input
- Separate encoding per zone (hand, battlefield, graveyard, etc.)
- Attention-based aggregation

Usage:
    encoder = ForgeGameStateEncoder()
    state_tensor = encoder.encode(forge_json_state)
"""

import json
import os
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Try to import h5py for loading pre-computed embeddings
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("Warning: h5py not installed. Install with: pip install h5py")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class GameStateConfig:
    """Configuration for the game state encoder."""

    # Card mechanics embedding
    mechanics_h5_path: str = "data/card_mechanics_commander.h5"
    vocab_size: int = 1403  # From vocabulary.py VOCAB_SIZE
    max_params: int = 37    # From precompute_embeddings.py

    # Zone capacities (max cards per zone to encode)
    max_hand_size: int = 15
    max_battlefield: int = 50  # Includes lands, creatures, etc.
    max_graveyard: int = 30
    max_exile: int = 20
    max_stack: int = 10
    max_command: int = 2  # Commander zone

    # Game state dimensions
    life_bits: int = 8  # Binary encoding up to 255 life
    mana_colors: int = 6  # WUBRG + C
    max_mana: int = 20  # Max mana of any color to encode

    # Player count (for Commander)
    max_players: int = 4

    # Architecture
    d_model: int = 512  # Internal embedding dimension
    n_heads: int = 8
    n_layers: int = 3
    d_ff: int = 1024
    dropout: float = 0.1

    # Output dimensions
    card_embedding_dim: int = 512  # Per-card encoding
    zone_embedding_dim: int = 512  # Per-zone aggregated encoding
    global_embedding_dim: int = 192  # Game-level features
    output_dim: int = 768  # Final combined state embedding

    @property
    def total_max_cards(self) -> int:
        """Total maximum cards to encode across all zones."""
        return (self.max_hand_size + self.max_battlefield +
                self.max_graveyard + self.max_exile +
                self.max_stack + self.max_command)


# =============================================================================
# ZONES AND PHASES
# =============================================================================

class Zone(IntEnum):
    """Game zones matching Forge's ZoneType."""
    HAND = 0
    LIBRARY = 1
    BATTLEFIELD = 2
    GRAVEYARD = 3
    EXILE = 4
    STACK = 5
    COMMAND = 6
    UNKNOWN = 7


class Phase(IntEnum):
    """Game phases matching Forge's PhaseType."""
    UNTAP = 0
    UPKEEP = 1
    DRAW = 2
    MAIN1 = 3
    COMBAT_BEGIN = 4
    COMBAT_ATTACKERS = 5
    COMBAT_BLOCKERS = 6
    COMBAT_FIRST_STRIKE = 7
    COMBAT_DAMAGE = 8
    COMBAT_END = 9
    MAIN2 = 10
    END = 11
    CLEANUP = 12
    UNKNOWN = 13


PHASE_NAMES = {
    "untap": Phase.UNTAP,
    "upkeep": Phase.UPKEEP,
    "draw": Phase.DRAW,
    "main1": Phase.MAIN1,
    "precombatmain": Phase.MAIN1,
    "combat_begin": Phase.COMBAT_BEGIN,
    "beginningofcombat": Phase.COMBAT_BEGIN,
    "combat_attackers": Phase.COMBAT_ATTACKERS,
    "declareattackers": Phase.COMBAT_ATTACKERS,
    "combat_blockers": Phase.COMBAT_BLOCKERS,
    "declareblockers": Phase.COMBAT_BLOCKERS,
    "combat_first_strike": Phase.COMBAT_FIRST_STRIKE,
    "firststrikedamage": Phase.COMBAT_FIRST_STRIKE,
    "combat_damage": Phase.COMBAT_DAMAGE,
    "combatdamage": Phase.COMBAT_DAMAGE,
    "combat_end": Phase.COMBAT_END,
    "endofcombat": Phase.COMBAT_END,
    "main2": Phase.MAIN2,
    "postcombatmain": Phase.MAIN2,
    "end": Phase.END,
    "endofturn": Phase.END,
    "cleanup": Phase.CLEANUP,
}


# =============================================================================
# MECHANICS CACHE (from HDF5)
# =============================================================================

class MechanicsCache:
    """
    Cache for pre-computed card mechanics embeddings.

    Loads from HDF5 and provides fast lookup by card name.
    """

    def __init__(self, h5_path: str, vocab_size: int = 1403, max_params: int = 37):
        """
        Load mechanics embeddings from HDF5 file.

        Args:
            h5_path: Path to card_mechanics_*.h5 file
            vocab_size: Fallback vocab size when HDF5 is unavailable
            max_params: Fallback max params when HDF5 is unavailable
        """
        self.h5_path = h5_path
        self.mechanics_matrix = None
        self.params_matrix = None
        self.card_index = {}
        self.vocab_size = vocab_size
        self.max_params = max_params
        self._loaded = False

        self._load()

    def _load(self):
        """Load data from HDF5."""
        if not HAS_H5PY:
            print("Warning: h5py not available, using zero embeddings")
            return

        if not os.path.exists(self.h5_path):
            print(f"Warning: {self.h5_path} not found, using zero embeddings")
            return

        with h5py.File(self.h5_path, 'r') as f:
            self.mechanics_matrix = f['mechanics'][:]
            self.params_matrix = f['parameters'][:]
            self.card_index = json.loads(f.attrs['card_index'])
            self.vocab_size = f.attrs['vocab_size']
            self.max_params = f.attrs['max_params']

        self._loaded = True
        print(f"Loaded mechanics for {len(self.card_index)} cards from {self.h5_path}")

    def get_embedding(self, card_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get mechanics embedding for a card.

        Args:
            card_name: Card name (case-sensitive, matches Scryfall)

        Returns:
            (mechanics_vector, params_vector) or zeros if not found
        """
        if not self._loaded:
            return (
                np.zeros(self.vocab_size, dtype=np.float32),
                np.zeros(self.max_params, dtype=np.float32)
            )

        idx = self.card_index.get(card_name)
        if idx is None:
            # Try case-insensitive match
            for name, i in self.card_index.items():
                if name.lower() == card_name.lower():
                    idx = i
                    break

        if idx is None:
            return (
                np.zeros(self.vocab_size, dtype=np.float32),
                np.zeros(self.max_params, dtype=np.float32)
            )

        return (
            self.mechanics_matrix[idx].astype(np.float32),
            self.params_matrix[idx].astype(np.float32)
        )

    def batch_get_embeddings(self, card_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get embeddings for multiple cards.

        Args:
            card_names: List of card names

        Returns:
            (mechanics_matrix, params_matrix) - shape [num_cards, dim]
        """
        mechanics = []
        params = []
        for name in card_names:
            m, p = self.get_embedding(name)
            mechanics.append(m)
            params.append(p)
        return np.stack(mechanics), np.stack(params)


# =============================================================================
# CARD STATE ENCODING
# =============================================================================

@dataclass
class CardState:
    """
    Runtime state for a card in a specific zone.

    Populated from Forge's JSON game state.
    """
    # Identity
    name: str = ""
    card_id: int = 0

    # Zone
    zone: Zone = Zone.UNKNOWN

    # Battlefield state (if applicable)
    is_tapped: bool = False
    is_attacking: bool = False
    is_blocking: bool = False
    is_blocked: bool = False
    summoning_sickness: bool = True

    # Counters (type -> count)
    counters: Dict[str, int] = field(default_factory=dict)

    # Combat info
    damage_marked: int = 0
    blocking_creature_id: Optional[int] = None
    blocked_by_ids: List[int] = field(default_factory=list)

    # Modifiers
    power_modifier: int = 0
    toughness_modifier: int = 0

    # Attachments
    attached_to_id: Optional[int] = None
    attached_ids: List[int] = field(default_factory=list)

    # Control
    controller_is_self: bool = True
    owner_is_self: bool = True

    # Temporal
    turns_on_battlefield: int = 0
    cast_this_turn: bool = False

    # Card identity features (from Forge JSON)
    is_creature: bool = False
    is_land: bool = False
    is_artifact: bool = False
    is_enchantment: bool = False
    is_planeswalker: bool = False
    is_instant: bool = False
    is_sorcery: bool = False
    power: int = 0
    toughness: int = 0
    cmc: int = 0

    # Stack-specific (for spells on the stack)
    targets: List[int] = field(default_factory=list)


def encode_card_state(
    card: CardState,
    mechanics_cache: MechanicsCache,
    config: GameStateConfig
) -> np.ndarray:
    """
    Encode a single card's complete state.

    Combines:
    1. Mechanics embedding (from HDF5)
    2. Dynamic state features

    Returns:
        numpy array of shape [mechanics_dim + state_dim]
    """
    # Get mechanics embedding
    mechanics, params = mechanics_cache.get_embedding(card.name)

    # Encode dynamic state
    state_features = []

    # Zone (one-hot, 8 dims)
    zone_enc = np.zeros(8, dtype=np.float32)
    zone_enc[min(card.zone, 7)] = 1.0
    state_features.append(zone_enc)

    # Boolean states (7 dims)
    bools = np.array([
        float(card.is_tapped),
        float(card.is_attacking),
        float(card.is_blocking),
        float(card.is_blocked),
        float(card.summoning_sickness),
        float(card.controller_is_self),
        float(card.cast_this_turn),
    ], dtype=np.float32)
    state_features.append(bools)

    # Counters (10 dims for common counter types)
    counter_types = ["p1p1", "m1m1", "loyalty", "charge", "time",
                     "lore", "verse", "blood", "energy", "experience"]
    counter_enc = np.zeros(10, dtype=np.float32)
    for i, ct in enumerate(counter_types):
        if ct in card.counters:
            counter_enc[i] = min(card.counters[ct], 20) / 20.0
    state_features.append(counter_enc)

    # Combat info (4 dims)
    combat = np.array([
        min(card.damage_marked, 20) / 20.0,
        min(len(card.blocked_by_ids), 5) / 5.0,
        float(card.attached_to_id is not None),
        min(len(card.attached_ids), 5) / 5.0,
    ], dtype=np.float32)
    state_features.append(combat)

    # Modifiers (2 dims)
    modifiers = np.array([
        np.clip(card.power_modifier, -10, 10) / 10.0,
        np.clip(card.toughness_modifier, -10, 10) / 10.0,
    ], dtype=np.float32)
    state_features.append(modifiers)

    # Temporal (1 dim)
    temporal = np.array([
        min(card.turns_on_battlefield, 10) / 10.0,
    ], dtype=np.float32)
    state_features.append(temporal)

    # Card type flags (7 dims)
    type_flags = np.array([
        float(card.is_creature),
        float(card.is_land),
        float(card.is_artifact),
        float(card.is_enchantment),
        float(card.is_planeswalker),
        float(card.is_instant),
        float(card.is_sorcery),
    ], dtype=np.float32)
    state_features.append(type_flags)

    # Power/toughness (2 dims, normalized)
    pt = np.array([
        np.clip(card.power, -1, 20) / 20.0,
        np.clip(card.toughness, -1, 20) / 20.0,
    ], dtype=np.float32)
    state_features.append(pt)

    # CMC (1 dim, normalized)
    cmc = np.array([
        np.clip(card.cmc, 0, 16) / 16.0,
    ], dtype=np.float32)
    state_features.append(cmc)

    # Combine all state features
    # 8 + 7 + 10 + 4 + 2 + 1 + 7 + 2 + 1 = 42 dims
    state = np.concatenate(state_features)

    # Combine mechanics + params + state
    return np.concatenate([mechanics, params, state])


# =============================================================================
# GAME STATE PARSING (from Forge JSON)
# =============================================================================

def parse_forge_json(json_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse Forge's JSON game state into structured data.

    Expected Forge JSON format (from ForgeDraftDaemon):
    {
        "turn": 5,
        "phase": "main1",
        "activePlayer": 0,
        "priorityPlayer": 0,
        "players": [
            {
                "id": 0,
                "name": "Player1",
                "life": 20,
                "mana": {"W": 2, "U": 0, "B": 0, "R": 0, "G": 1, "C": 3},
                "hand": [{"name": "Lightning Bolt", "id": 1}, ...],
                "battlefield": [{"name": "Grizzly Bears", "id": 2, "tapped": false, ...}, ...],
                "graveyard": [...],
                "exile": [...],
                "command": [...]
            },
            ...
        ],
        "stack": [{"name": "Counterspell", "id": 10, "targets": [1]}, ...],
        "legalActions": [...]
    }
    """
    parsed = {
        "turn": json_state.get("turn", 1),
        "phase": _parse_phase(json_state.get("phase", "main1")),
        "active_player": json_state.get("activePlayer", 0),
        "priority_player": json_state.get("priorityPlayer", 0),
        "players": [],
        "stack": [],
        "legal_actions": json_state.get("legalActions", []),
    }

    # Parse players
    for player_data in json_state.get("players", []):
        player = {
            "id": player_data.get("id", 0),
            "name": player_data.get("name", "Unknown"),
            "life": player_data.get("life", 20),
            "mana": player_data.get("mana", {}),
            "poison": player_data.get("poison", 0),
            "library_size": player_data.get("library_size", 0),
            "hand_size": player_data.get("hand_size", len(player_data.get("hand", []))),
            "lands_played_this_turn": player_data.get("lands_played_this_turn", 0),
            "max_land_plays": player_data.get("max_land_plays", 1),
            "has_lost": player_data.get("has_lost", False),
            "cards": {
                Zone.HAND: _parse_cards(player_data.get("hand", []), Zone.HAND),
                Zone.BATTLEFIELD: _parse_cards(player_data.get("battlefield", []), Zone.BATTLEFIELD),
                Zone.GRAVEYARD: _parse_cards(player_data.get("graveyard", []), Zone.GRAVEYARD),
                Zone.EXILE: _parse_cards(player_data.get("exile", []), Zone.EXILE),
                Zone.COMMAND: _parse_cards(player_data.get("command", []), Zone.COMMAND),
            },
        }
        parsed["players"].append(player)

    # Parse stack
    for stack_item in json_state.get("stack", []):
        card = _parse_card(stack_item, Zone.STACK)
        card.targets = stack_item.get("targets", [])
        parsed["stack"].append(card)

    return parsed


def _parse_phase(phase_str: str) -> Phase:
    """Parse phase string to Phase enum."""
    key = phase_str.lower().replace(" ", "").replace("_", "")
    return PHASE_NAMES.get(key, Phase.UNKNOWN)


def _parse_cards(cards_data: List[Dict], zone: Zone) -> List[CardState]:
    """Parse a list of card JSON objects."""
    return [_parse_card(c, zone) for c in cards_data]


def _parse_card(card_data: Dict, zone: Zone) -> CardState:
    """Parse a single card JSON object."""
    card = CardState(
        name=card_data.get("name", "Unknown"),
        card_id=card_data.get("id", 0),
        zone=zone,
    )

    # Card identity features (available in all zones)
    card.is_creature = card_data.get("is_creature", False)
    card.is_land = card_data.get("is_land", False)
    card.is_artifact = card_data.get("is_artifact", False)
    card.is_enchantment = card_data.get("is_enchantment", False)
    card.is_planeswalker = card_data.get("is_planeswalker", False)
    card.is_instant = card_data.get("is_instant", False)
    card.is_sorcery = card_data.get("is_sorcery", False)
    card.power = card_data.get("power", 0) or 0
    card.toughness = card_data.get("toughness", 0) or 0
    card.cmc = card_data.get("cmc", 0) or 0

    # Battlefield-specific fields
    if zone == Zone.BATTLEFIELD:
        card.is_tapped = card_data.get("tapped", False)
        card.is_attacking = card_data.get("attacking", False)
        card.is_blocking = card_data.get("blocking", False)
        card.is_blocked = card_data.get("blocked", False)
        card.summoning_sickness = card_data.get("summoningSickness", True)
        card.damage_marked = card_data.get("damage", 0)
        card.power_modifier = card_data.get("powerMod", 0)
        card.toughness_modifier = card_data.get("toughnessMod", 0)
        card.attached_to_id = card_data.get("attachedTo")
        card.attached_ids = card_data.get("attachedCards", [])
        card.turns_on_battlefield = card_data.get("turnsOnBattlefield", 0)

        # Counters
        for counter_data in card_data.get("counters", []):
            ct = counter_data.get("type", "unknown").lower()
            count = counter_data.get("count", 0)
            card.counters[ct] = count

    # Control info
    card.controller_is_self = card_data.get("controllerIsSelf", True)
    card.owner_is_self = card_data.get("ownerIsSelf", True)
    card.cast_this_turn = card_data.get("castThisTurn", False)

    return card


# =============================================================================
# NEURAL NETWORK COMPONENTS
# =============================================================================

class CardEmbeddingMLP(nn.Module):
    """Shared 2-layer MLP for projecting raw card features to d_model.

    Replaces the per-zone single linear projection with a shared nonlinear
    embedding that can learn mechanic interactions (e.g., FLYING + DEATHTOUCH).
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class ZoneEncoder(nn.Module):
    """
    Encodes cards in a specific zone using self-attention.

    Takes variable number of cards, outputs fixed-size zone embedding.
    """

    def __init__(self, config: GameStateConfig, card_embedding: CardEmbeddingMLP):
        super().__init__()
        self.config = config
        n_zone_layers = 2

        # Shared card embedding projection (mechanics + params + state -> d_model)
        self.card_embedding = card_embedding

        # Stacked self-attention + feed-forward layers for card interactions
        self.layers = nn.ModuleList()
        for _ in range(n_zone_layers):
            self.layers.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(
                    config.d_model, config.n_heads,
                    dropout=config.dropout, batch_first=True,
                ),
                'attn_norm': nn.LayerNorm(config.d_model),
                'ff': nn.Sequential(
                    nn.Linear(config.d_model, config.d_ff),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.d_ff, config.d_model),
                    nn.Dropout(config.dropout),
                ),
                'ff_norm': nn.LayerNorm(config.d_model),
            }))

        # Pooling to zone embedding
        self.pool_proj = nn.Sequential(
            nn.Linear(config.d_model, config.zone_embedding_dim),
            nn.LayerNorm(config.zone_embedding_dim),
        )

        # CLS token for pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model))

    def forward(
        self,
        card_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode cards in a zone.

        Args:
            card_features: [batch, max_cards, feature_dim]
            mask: [batch, max_cards] - 1 for valid, 0 for padding

        Returns:
            card_embeddings: [batch, max_cards, d_model]
            zone_embedding: [batch, zone_embedding_dim]
        """
        batch_size = card_features.shape[0]

        # Project cards through shared embedding
        x = self.card_embedding(card_features)

        # Add CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Extend mask for CLS token
        if mask is not None:
            cls_mask = torch.ones(batch_size, 1, device=mask.device)
            attn_mask = torch.cat([cls_mask, mask], dim=1)
            key_padding_mask = (attn_mask == 0)
        else:
            key_padding_mask = None

        # Apply stacked self-attention + feed-forward layers
        for layer in self.layers:
            attn_out, _ = layer['attn'](x, x, x, key_padding_mask=key_padding_mask)
            x = layer['attn_norm'](x + attn_out)

            ff_out = layer['ff'](x)
            x = layer['ff_norm'](x + ff_out)

        # Extract CLS for zone embedding, rest are card embeddings
        zone_emb = self.pool_proj(x[:, 0])
        card_emb = x[:, 1:]

        return card_emb, zone_emb


class GlobalEncoder(nn.Module):
    """
    Encodes global game state (life totals, turn, phase, mana, board awareness).
    """

    # Number of extra global features (per-player + relative + game flow)
    EXTRA_FEATURES_DIM = 46

    def __init__(self, config: GameStateConfig):
        super().__init__()
        self.config = config

        # Life total encoding (binary for each player)
        self.life_embed = nn.Sequential(
            nn.Linear(config.life_bits * config.max_players, 96),
            nn.GELU(),
        )

        # Mana encoding (per player, per color)
        mana_dim = config.mana_colors * config.max_players
        self.mana_embed = nn.Sequential(
            nn.Linear(mana_dim, 96),
            nn.GELU(),
        )

        # Turn/phase encoding
        self.turn_embed = nn.Sequential(
            nn.Linear(1 + 14, 48),  # turn number + phase one-hot
            nn.GELU(),
        )

        # Priority/active player encoding
        self.priority_embed = nn.Sequential(
            nn.Linear(config.max_players * 2, 48),  # active + priority
            nn.GELU(),
        )

        # Board-awareness / extra features encoding
        self.extra_embed = nn.Sequential(
            nn.Linear(self.EXTRA_FEATURES_DIM, 96),
            nn.GELU(),
        )

        # Combine all global features
        self.combine = nn.Sequential(
            nn.Linear(96 + 96 + 48 + 48 + 96, config.global_embedding_dim),
            nn.LayerNorm(config.global_embedding_dim),
            nn.GELU(),
        )

    def forward(
        self,
        life_totals: torch.Tensor,      # [batch, max_players]
        mana_pools: torch.Tensor,        # [batch, max_players, mana_colors]
        turn_number: torch.Tensor,       # [batch, 1]
        phase: torch.Tensor,             # [batch, 14] one-hot
        active_player: torch.Tensor,     # [batch, max_players] one-hot
        priority_player: torch.Tensor,   # [batch, max_players] one-hot
        extra_features: Optional[torch.Tensor] = None,  # [batch, 46]
    ) -> torch.Tensor:
        """
        Encode global game state.

        Returns:
            [batch, global_embedding_dim]
        """
        batch_size = life_totals.shape[0]

        # Encode life totals (binary representation)
        life_binary = self._to_binary(life_totals, self.config.life_bits)
        life_emb = self.life_embed(life_binary.view(batch_size, -1))

        # Encode mana
        mana_emb = self.mana_embed(mana_pools.view(batch_size, -1))

        # Encode turn/phase
        turn_phase = torch.cat([turn_number / 20.0, phase], dim=-1)  # Normalize turn
        turn_emb = self.turn_embed(turn_phase)

        # Encode priority
        priority = torch.cat([active_player, priority_player], dim=-1)
        priority_emb = self.priority_embed(priority)

        # Encode extra features (board awareness)
        if extra_features is None:
            extra_features = torch.zeros(
                batch_size, self.EXTRA_FEATURES_DIM,
                device=life_totals.device
            )
        extra_emb = self.extra_embed(extra_features)

        # Combine
        combined = torch.cat([life_emb, mana_emb, turn_emb, priority_emb, extra_emb], dim=-1)
        return self.combine(combined)

    def _to_binary(self, x: torch.Tensor, num_bits: int) -> torch.Tensor:
        """Convert integers to binary representation."""
        x = x.long().clamp(0, 2**num_bits - 1)
        binary = torch.zeros(*x.shape, num_bits, device=x.device)
        for i in range(num_bits):
            binary[..., i] = (x >> i) & 1
        return binary


class StackEncoder(nn.Module):
    """
    Encodes the stack (spells and abilities waiting to resolve).

    The stack has LIFO order which is important for decision making.
    Uses positional encoding to maintain order.
    """

    def __init__(self, config: GameStateConfig, card_embedding: CardEmbeddingMLP):
        super().__init__()
        self.config = config

        # Shared card embedding projection
        self.card_embedding = card_embedding

        # Position projection: card embedding (d_model) + position (max_stack) -> d_model
        self.item_proj = nn.Sequential(
            nn.Linear(config.d_model + config.max_stack, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
        )

        # Self-attention for stack interactions
        self.self_attn = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(config.d_model)

        # Pool to stack embedding
        self.pool = nn.Sequential(
            nn.Linear(config.d_model, config.zone_embedding_dim),
            nn.LayerNorm(config.zone_embedding_dim),
        )

        # Positional encoding for stack order
        self.pos_encoding = nn.Parameter(
            torch.randn(1, config.max_stack, config.max_stack)
        )

    def forward(
        self,
        stack_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode the stack.

        Args:
            stack_features: [batch, max_stack, feature_dim]
            mask: [batch, max_stack] - 1 for valid, 0 for padding

        Returns:
            stack_embedding: [batch, zone_embedding_dim]
        """
        batch_size = stack_features.shape[0]

        # Handle empty stack (all zeros in mask)
        if mask is not None:
            has_items = mask.sum(dim=1) > 0
            if not has_items.any():
                # All stacks are empty, return zero embedding
                return torch.zeros(
                    batch_size, self.config.zone_embedding_dim,
                    device=stack_features.device
                )

        # Project card features through shared embedding first
        x = self.card_embedding(stack_features)

        # Add positional encoding
        pos = self.pos_encoding.expand(batch_size, -1, -1)
        x = torch.cat([x, pos], dim=-1)

        # Project combined embedding + position
        x = self.item_proj(x)

        # Self-attention (skip if all masked to avoid NaN)
        if mask is not None:
            # Only apply attention if there are items
            valid_mask = mask.sum(dim=1) > 0
            if valid_mask.all():
                key_padding_mask = (mask == 0)
                attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
                x = self.norm(x + attn_out)
            else:
                # Mixed: some batches have items, some don't
                # Just skip attention and use projections only
                x = self.norm(x)
        else:
            attn_out, _ = self.self_attn(x, x, x)
            x = self.norm(x + attn_out)

        # Pool (mean over valid items)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            x = x * mask_expanded
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
            x = x.sum(dim=1) / denom
        else:
            x = x.mean(dim=1)

        return self.pool(x)


class CrossZoneAttention(nn.Module):
    """
    Models interactions between zones with multi-layer attention.

    Multiple layers enable multi-step cross-zone reasoning, e.g.:
    - Layer 1: "reanimation target in graveyard"
    - Layer 2: "reanimation spell in hand + target identified"
    - Layer 3: "mana available to cast reanimation spell"
    """

    def __init__(self, config: GameStateConfig):
        super().__init__()
        self.config = config
        n_cross_layers = 3

        # Input/output projections
        self.input_proj = nn.Linear(config.zone_embedding_dim, config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.zone_embedding_dim)

        # Stack of cross-attention + feed-forward layers
        self.layers = nn.ModuleList()
        for _ in range(n_cross_layers):
            self.layers.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(
                    config.d_model, config.n_heads,
                    dropout=config.dropout, batch_first=True,
                ),
                'attn_norm': nn.LayerNorm(config.d_model),
                'ff': nn.Sequential(
                    nn.Linear(config.d_model, config.d_ff),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.d_ff, config.d_model),
                    nn.Dropout(config.dropout),
                ),
                'ff_norm': nn.LayerNorm(config.d_model),
            }))

    def forward(self, zone_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-layer cross-zone attention.

        Args:
            zone_embeddings: [batch, num_zones, zone_embedding_dim]

        Returns:
            updated_embeddings: [batch, num_zones, zone_embedding_dim]
        """
        # Project to common space
        x = self.input_proj(zone_embeddings)

        # Apply each cross-attention + feed-forward layer
        for layer in self.layers:
            # Self-attention across zones
            attn_out, _ = layer['attn'](x, x, x)
            x = layer['attn_norm'](x + attn_out)

            # Feed-forward
            ff_out = layer['ff'](x)
            x = layer['ff_norm'](x + ff_out)

        # Project back to zone embedding dim
        return self.output_proj(x)


# =============================================================================
# MAIN ENCODER
# =============================================================================

class ForgeGameStateEncoder(nn.Module):
    """
    Complete game state encoder for Forge MTG engine.

    Takes Forge JSON state, outputs tensor for policy/value network.

    Architecture:
    1. Load card mechanics from HDF5 (pre-computed)
    2. Encode each zone separately (hand, battlefield, etc.)
    3. Encode global game state (life, mana, turn, phase)
    4. Encode stack (spells/abilities waiting to resolve)
    5. Cross-zone attention for inter-zone relationships
    6. Combine into final state embedding
    """

    def __init__(self, config: Optional[GameStateConfig] = None):
        super().__init__()
        self.config = config or GameStateConfig()

        # Load mechanics cache
        self.mechanics_cache = MechanicsCache(
            self.config.mechanics_h5_path,
            vocab_size=self.config.vocab_size,
            max_params=self.config.max_params,
        )

        # Shared card embedding MLP (used by all zone encoders and stack encoder)
        # 42 = state features (zone 8 + bools 7 + counters 10 + combat 4 + mods 2 + temporal 1 + types 7 + P/T 2 + CMC 1)
        input_dim = self.config.vocab_size + self.config.max_params + 42
        self.card_embedding = CardEmbeddingMLP(
            input_dim, 1024, self.config.d_model, self.config.dropout
        )

        # Zone encoders (one per zone type, sharing card embedding)
        self.zone_encoders = nn.ModuleDict({
            "hand": ZoneEncoder(self.config, self.card_embedding),
            "battlefield": ZoneEncoder(self.config, self.card_embedding),
            "graveyard": ZoneEncoder(self.config, self.card_embedding),
            "exile": ZoneEncoder(self.config, self.card_embedding),
        })

        # Global encoder
        self.global_encoder = GlobalEncoder(self.config)

        # Stack encoder (shares card embedding)
        self.stack_encoder = StackEncoder(self.config, self.card_embedding)

        # Cross-zone attention
        self.cross_zone_attn = CrossZoneAttention(self.config)

        # Combine all embeddings
        # 4 zone embeddings (cards from all players combined per zone)
        # + 1 stack embedding + global embedding
        num_zones = 4  # hand, battlefield, graveyard, exile
        zone_dim = self.config.zone_embedding_dim

        combine_dim = (num_zones * zone_dim +                 # Zone embeddings
                      zone_dim +                              # Stack embedding
                      self.config.global_embedding_dim)       # Global embedding

        self.combine = nn.Sequential(
            nn.Linear(combine_dim, self.config.d_ff),
            nn.LayerNorm(self.config.d_ff),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_ff, self.config.output_dim),
            nn.LayerNorm(self.config.output_dim),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def encode_json(self, json_state: Dict[str, Any]) -> torch.Tensor:
        """
        Encode a Forge JSON game state.

        This is the main entry point for encoding game states.

        Args:
            json_state: Forge JSON game state dict

        Returns:
            state_embedding: [1, output_dim] tensor
        """
        # Parse JSON to structured data
        parsed = parse_forge_json(json_state)

        # Convert to tensors
        tensors = self._prepare_tensors(parsed)

        # Run through network
        return self.forward(**tensors)

    def _prepare_tensors(
        self,
        parsed_state: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Convert parsed state to tensors for the network.
        """

        # Encode cards by zone for each player
        zone_cards = {}
        zone_masks = {}

        for zone_name, max_cards in [
            ("hand", self.config.max_hand_size),
            ("battlefield", self.config.max_battlefield),
            ("graveyard", self.config.max_graveyard),
            ("exile", self.config.max_exile),
        ]:
            zone = Zone[zone_name.upper()]

            # Collect cards from all players
            all_cards = []
            for player_idx, player in enumerate(parsed_state["players"]):
                is_self = (player_idx == 0)  # First player is "self"
                cards = player["cards"].get(zone, [])
                for card in cards:
                    card.controller_is_self = is_self
                    all_cards.append(card)

            # Encode cards
            card_features = np.zeros(
                (max_cards, self.config.vocab_size + self.config.max_params + 42),
                dtype=np.float32
            )
            mask = np.zeros(max_cards, dtype=np.float32)

            for i, card in enumerate(all_cards[:max_cards]):
                card_features[i] = encode_card_state(
                    card, self.mechanics_cache, self.config
                )
                mask[i] = 1.0

            zone_cards[zone_name] = torch.tensor(card_features).unsqueeze(0)
            zone_masks[zone_name] = torch.tensor(mask).unsqueeze(0)

        # Encode stack
        stack_features = np.zeros(
            (self.config.max_stack, self.config.vocab_size + self.config.max_params + 42),
            dtype=np.float32
        )
        stack_mask = np.zeros(self.config.max_stack, dtype=np.float32)

        for i, card in enumerate(parsed_state["stack"][:self.config.max_stack]):
            stack_features[i] = encode_card_state(
                card, self.mechanics_cache, self.config
            )
            stack_mask[i] = 1.0

        stack_features = torch.tensor(stack_features).unsqueeze(0)
        stack_mask = torch.tensor(stack_mask).unsqueeze(0)

        # Global state

        life_totals = np.zeros((1, self.config.max_players), dtype=np.float32)
        mana_pools = np.zeros(
            (1, self.config.max_players, self.config.mana_colors),
            dtype=np.float32
        )

        for i, player in enumerate(parsed_state["players"][:self.config.max_players]):
            life_totals[0, i] = player["life"]
            mana = player["mana"]
            for j, color in enumerate(["W", "U", "B", "R", "G", "C"]):
                mana_pools[0, i, j] = min(mana.get(color, 0), self.config.max_mana)

        # Normalize mana
        mana_pools = mana_pools / self.config.max_mana

        # Turn/phase
        turn_number = np.array([[parsed_state["turn"]]], dtype=np.float32)
        phase = np.zeros((1, 14), dtype=np.float32)
        phase[0, min(parsed_state["phase"], 13)] = 1.0

        # Active/priority player
        active_player = np.zeros((1, self.config.max_players), dtype=np.float32)
        active_player[0, parsed_state["active_player"] % self.config.max_players] = 1.0

        priority_player = np.zeros((1, self.config.max_players), dtype=np.float32)
        priority_player[0, parsed_state["priority_player"] % self.config.max_players] = 1.0

        # Extra global features (46 dims) for board-awareness
        extra_features = self._compute_extra_features(parsed_state)

        return {
            "zone_cards": zone_cards,
            "zone_masks": zone_masks,
            "stack_features": stack_features,
            "stack_mask": stack_mask,
            "life_totals": torch.tensor(life_totals),
            "mana_pools": torch.tensor(mana_pools),
            "turn_number": torch.tensor(turn_number),
            "phase": torch.tensor(phase),
            "active_player": torch.tensor(active_player),
            "priority_player": torch.tensor(priority_player),
            "extra_features": torch.tensor(extra_features),
        }

    def _compute_extra_features(
        self,
        parsed_state: Dict[str, Any],
    ) -> np.ndarray:
        """
        Compute 46-dim extra global features for board awareness.

        Layout (46 dims):
        - Per-player (4 players × 9 features = 36): poison, library_size,
          hand_size, lands_played, max_lands, has_lost, creature_count,
          total_power, untapped_lands
        - Relative (6): life_diff, card_advantage, board_advantage,
          library_ratio, poison_threat, lethal_on_board
        - Game flow (4): stack_depth, is_combat, is_main, self_graveyard_count
        """
        extra = np.zeros((1, GlobalEncoder.EXTRA_FEATURES_DIM), dtype=np.float32)
        players = parsed_state["players"]

        # Per-player features (9 per player, 4 players = 36 dims)
        per_player_stats = []
        for pidx in range(self.config.max_players):
            if pidx < len(players):
                p = players[pidx]
                battlefield_cards = p["cards"].get(Zone.BATTLEFIELD, [])
                creature_count = sum(1 for c in battlefield_cards if c.is_creature)
                total_power = sum(c.power for c in battlefield_cards if c.is_creature)
                untapped_lands = sum(
                    1 for c in battlefield_cards
                    if c.is_land and not c.is_tapped
                )

                stats = [
                    min(p.get("poison", 0), 10) / 10.0,
                    min(p.get("library_size", 0), 60) / 60.0,
                    min(p.get("hand_size", 0), 15) / 15.0,
                    min(p.get("lands_played_this_turn", 0), 3) / 3.0,
                    min(p.get("max_land_plays", 1), 3) / 3.0,
                    float(p.get("has_lost", False)),
                    min(creature_count, 20) / 20.0,
                    min(total_power, 40) / 40.0,
                    min(untapped_lands, 10) / 10.0,
                ]
            else:
                stats = [0.0] * 9
            per_player_stats.append(stats)

        # Flatten per-player → 36 dims
        for pidx in range(self.config.max_players):
            offset = pidx * 9
            for j, val in enumerate(per_player_stats[pidx]):
                extra[0, offset + j] = val

        # Relative features (6 dims) — self (player 0) vs best opponent
        if len(players) >= 2:
            p0 = players[0]
            p1 = players[1]  # Primary opponent
            bf0 = p0["cards"].get(Zone.BATTLEFIELD, [])
            bf1 = p1["cards"].get(Zone.BATTLEFIELD, [])

            life_diff = np.clip((p0["life"] - p1["life"]) / 40.0, -1.0, 1.0)
            card_adv = np.clip(
                (p0.get("hand_size", 0) - p1.get("hand_size", 0)) / 7.0,
                -1.0, 1.0,
            )
            board_adv = np.clip(
                (len(bf0) - len(bf1)) / 10.0,
                -1.0, 1.0,
            )
            lib0 = max(p0.get("library_size", 1), 1)
            lib1 = max(p1.get("library_size", 1), 1)
            library_ratio = np.clip(lib0 / (lib0 + lib1), 0.0, 1.0)
            poison_threat = min(p0.get("poison", 0), 10) / 10.0
            opp_total_power = sum(c.power for c in bf1 if c.is_creature)
            lethal_on_board = float(opp_total_power >= p0["life"])
        else:
            life_diff = 0.0
            card_adv = 0.0
            board_adv = 0.0
            library_ratio = 0.5
            poison_threat = 0.0
            lethal_on_board = 0.0

        extra[0, 36] = life_diff
        extra[0, 37] = card_adv
        extra[0, 38] = board_adv
        extra[0, 39] = library_ratio
        extra[0, 40] = poison_threat
        extra[0, 41] = lethal_on_board

        # Game flow features (4 dims)
        phase_val = parsed_state["phase"]
        stack_depth = min(len(parsed_state["stack"]), 10) / 10.0
        is_combat = float(Phase.COMBAT_BEGIN <= phase_val <= Phase.COMBAT_END)
        is_main = float(phase_val in (Phase.MAIN1, Phase.MAIN2))
        gy_count = 0
        if players:
            gy_count = len(players[0]["cards"].get(Zone.GRAVEYARD, []))

        extra[0, 42] = stack_depth
        extra[0, 43] = is_combat
        extra[0, 44] = is_main
        extra[0, 45] = min(gy_count, 30) / 30.0

        return extra

    def forward(
        self,
        zone_cards: Dict[str, torch.Tensor],
        zone_masks: Dict[str, torch.Tensor],
        stack_features: torch.Tensor,
        stack_mask: torch.Tensor,
        life_totals: torch.Tensor,
        mana_pools: torch.Tensor,
        turn_number: torch.Tensor,
        phase: torch.Tensor,
        active_player: torch.Tensor,
        priority_player: torch.Tensor,
        extra_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Returns:
            state_embedding: [batch, output_dim]
        """
        batch_size = life_totals.shape[0]

        # Encode each zone
        zone_embeddings = []
        for zone_name in ["hand", "battlefield", "graveyard", "exile"]:
            _, zone_emb = self.zone_encoders[zone_name](
                zone_cards[zone_name],
                zone_masks[zone_name]
            )
            zone_embeddings.append(zone_emb)

        # Stack embeddings [batch, num_zones, zone_dim]
        zone_tensor = torch.stack(zone_embeddings, dim=1)

        # Cross-zone attention
        zone_tensor = self.cross_zone_attn(zone_tensor)

        # Flatten zone embeddings
        zone_flat = zone_tensor.view(batch_size, -1)

        # Encode stack
        stack_emb = self.stack_encoder(stack_features, stack_mask)

        # Encode global state
        global_emb = self.global_encoder(
            life_totals, mana_pools, turn_number, phase,
            active_player, priority_player, extra_features
        )

        # Combine all
        combined = torch.cat([zone_flat, stack_emb, global_emb], dim=-1)

        return self.combine(combined)

    def save(self, path: str):
        """Save encoder to file."""
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'ForgeGameStateEncoder':
        """Load encoder from file."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        encoder = cls(checkpoint['config'])
        encoder.load_state_dict(checkpoint['state_dict'])
        return encoder
