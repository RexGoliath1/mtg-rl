#!/usr/bin/env python3
"""
Hierarchical Action Space for MTG

Implements AlphaStar-style hierarchical action decomposition for handling
the massive combinatorial action space in Magic: The Gathering.

Action Space Complexity:
- ~30,000 unique cards
- Each card can have multiple modes, targets, X values
- Combat involves selecting attackers/blockers from variable-size sets
- Stack interactions require choosing when to respond

Hierarchical Decomposition (inspired by AlphaStar):
Level 1: Action Type     → {pass, play_spell, play_land, activate, attack, block, respond}
Level 2: Card Selection  → Which card from available set
Level 3: Mode Selection  → Which mode/kicker/X value
Level 4: Target Selection → What to target (can be multiple)

This reduces the action space from O(|cards| × |targets| × |modes|) to
O(|types|) + O(|cards|) + O(|modes|) + O(|targets|)

References:
- AlphaStar auto-regressive policy: https://deepmind.google/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii/
- Structured RL for combinatorial actions: https://openreview.net/forum?id=GS9o7u5njS
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from policy_network import TransformerConfig, PositionalEncoding, MultiHeadAttention


# =============================================================================
# ACTION TYPE DEFINITIONS
# =============================================================================

class ActionCategory(Enum):
    """High-level action categories."""
    PASS = 0
    PLAY_LAND = 1
    CAST_SPELL = 2
    ACTIVATE_ABILITY = 3
    DECLARE_ATTACKERS = 4
    DECLARE_BLOCKERS = 5
    RESPOND_TO_SPELL = 6
    SPECIAL_ACTION = 7  # Face-down, morph, etc.


class TargetType(Enum):
    """Types of targets in MTG."""
    NONE = 0
    PLAYER = 1
    CREATURE = 2
    PLANESWALKER = 3
    PERMANENT = 4
    SPELL_ON_STACK = 5
    CARD_IN_GRAVEYARD = 6
    CARD_IN_HAND = 7
    ANY = 8


@dataclass
class ActionSpec:
    """Specification for an available action."""
    category: ActionCategory
    card_index: int = -1  # Index in available cards
    card_name: str = ""
    requires_target: bool = False
    target_types: List[TargetType] = field(default_factory=list)
    requires_mode: bool = False
    available_modes: List[str] = field(default_factory=list)
    requires_x_value: bool = False
    max_x_value: int = 0
    mana_cost: str = ""
    raw_data: Dict = field(default_factory=dict)


@dataclass
class HierarchicalAction:
    """A fully specified hierarchical action."""
    category: ActionCategory
    card_index: int = -1
    mode_index: int = 0
    x_value: int = 0
    target_indices: List[int] = field(default_factory=list)

    def to_flat_index(self, action_specs: List[ActionSpec]) -> int:
        """Convert to flat action index for the environment."""
        if self.category == ActionCategory.PASS:
            return -1

        # Find matching action spec
        for i, spec in enumerate(action_specs):
            if (spec.category == self.category and
                spec.card_index == self.card_index):
                return spec.raw_data.get('index', i)

        return -1


# =============================================================================
# ACTION PARSER
# =============================================================================

class ActionParser:
    """
    Parses available actions from the environment into hierarchical structure.

    Maps the flat action list from Forge into a structured hierarchy.
    """

    def __init__(self):
        # Patterns for identifying action types
        self.land_patterns = ['play land', 'put land']
        self.spell_patterns = ['cast', 'play']
        self.ability_patterns = ['activate', 'tap:', '{t}:', 'untap:']
        self.attack_patterns = ['attack', 'declare attacker']
        self.block_patterns = ['block', 'declare blocker']

    def parse_actions(self, raw_actions: List[Dict]) -> List[ActionSpec]:
        """
        Parse raw action list into structured ActionSpec objects.

        Args:
            raw_actions: List of action dictionaries from environment

        Returns:
            List of ActionSpec objects
        """
        specs = []

        for i, action in enumerate(raw_actions):
            spec = self._parse_single_action(action, i)
            specs.append(spec)

        return specs

    def _parse_single_action(self, action: Dict, index: int) -> ActionSpec:
        """Parse a single action dictionary."""
        description = action.get('description', '').lower()
        card_name = action.get('card_name', '')

        # Determine category
        category = self._identify_category(description, action)

        # Check for targeting requirements
        requires_target = 'target' in description
        target_types = self._extract_target_types(description)

        # Check for mode/kicker
        requires_mode = any(x in description for x in ['choose', 'mode', 'kicker'])
        available_modes = action.get('modes', [])

        # Check for X value
        requires_x = 'x' in action.get('mana_cost', '').lower()
        max_x = action.get('max_x', 0)

        return ActionSpec(
            category=category,
            card_index=index,
            card_name=card_name,
            requires_target=requires_target,
            target_types=target_types,
            requires_mode=requires_mode,
            available_modes=available_modes,
            requires_x_value=requires_x,
            max_x_value=max_x,
            mana_cost=action.get('mana_cost', ''),
            raw_data=action,
        )

    def _identify_category(self, description: str, action: Dict) -> ActionCategory:
        """Identify the action category from description."""
        if action.get('is_land'):
            return ActionCategory.PLAY_LAND

        if any(p in description for p in self.attack_patterns):
            return ActionCategory.DECLARE_ATTACKERS

        if any(p in description for p in self.block_patterns):
            return ActionCategory.DECLARE_BLOCKERS

        if any(p in description for p in self.ability_patterns):
            return ActionCategory.ACTIVATE_ABILITY

        if any(p in description for p in self.spell_patterns):
            return ActionCategory.CAST_SPELL

        if action.get('is_special'):
            return ActionCategory.SPECIAL_ACTION

        return ActionCategory.PASS

    def _extract_target_types(self, description: str) -> List[TargetType]:
        """Extract required target types from description."""
        types = []

        if 'target creature' in description:
            types.append(TargetType.CREATURE)
        if 'target player' in description:
            types.append(TargetType.PLAYER)
        if 'target planeswalker' in description:
            types.append(TargetType.PLANESWALKER)
        if 'target permanent' in description:
            types.append(TargetType.PERMANENT)
        if 'any target' in description:
            types.append(TargetType.ANY)

        return types if types else [TargetType.NONE]

    def group_by_category(self, specs: List[ActionSpec]) -> Dict[ActionCategory, List[ActionSpec]]:
        """Group action specs by category."""
        groups = {cat: [] for cat in ActionCategory}
        for spec in specs:
            groups[spec.category].append(spec)
        return groups


# =============================================================================
# POINTER NETWORK FOR SELECTION
# =============================================================================

class PointerNetwork(nn.Module):
    """
    Pointer Network for selecting from variable-size sets.

    Used for card selection and target selection in the hierarchical policy.
    Uses attention to "point" to elements in the input sequence.

    Reference: Vinyals et al., "Pointer Networks" (2015)
    """

    def __init__(self, hidden_dim: int, n_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Query projection (from context)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)

        # Key projection (from candidates)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)

        # Multi-head attention for pointer
        self.attention = MultiHeadAttention(hidden_dim, n_heads)

        # Final pointer scores
        self.pointer_score = nn.Linear(hidden_dim, 1)

    def forward(self,
                context: torch.Tensor,
                candidates: torch.Tensor,
                candidate_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute pointer scores over candidates.

        Args:
            context: (batch, hidden_dim) - Query context
            candidates: (batch, n_candidates, hidden_dim) - Candidate embeddings
            candidate_mask: (batch, n_candidates) - Valid candidates mask

        Returns:
            pointer_logits: (batch, n_candidates) - Logits for each candidate
        """
        batch_size, n_candidates, _ = candidates.shape

        # Expand context for attention
        query = self.query_proj(context).unsqueeze(1)  # (batch, 1, hidden)
        keys = self.key_proj(candidates)  # (batch, n_candidates, hidden)

        # Attention scores
        attn_output = self.attention(query, keys, keys, candidate_mask)

        # Combine with candidates for final scoring
        combined = attn_output.expand(-1, n_candidates, -1) + candidates
        pointer_logits = self.pointer_score(combined).squeeze(-1)  # (batch, n_candidates)

        # Apply mask
        if candidate_mask is not None:
            pointer_logits = pointer_logits.masked_fill(candidate_mask == 0, float('-inf'))

        return pointer_logits


# =============================================================================
# HIERARCHICAL POLICY NETWORK
# =============================================================================

class HierarchicalPolicyNetwork(nn.Module):
    """
    Hierarchical policy network for MTG action selection.

    Architecture:
    1. State Encoder: Encodes game state using Transformer
    2. Action Type Head: Selects high-level action category
    3. Card Selector: Pointer network for card selection
    4. Mode Selector: Selects mode/kicker if applicable
    5. Target Selector: Pointer network for target selection
    6. Value Head: State value estimation

    Auto-regressive: Each selection conditions on previous selections.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # State encoding (shared with base policy)
        self.card_projection = nn.Linear(config.card_embedding_dim, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model, max_len=200)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model))

        # Global features projection
        self.global_proj = nn.Linear(config.global_feature_dim, config.d_model)

        # Level 1: Action Type Head
        self.action_type_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, len(ActionCategory)),
        )

        # Level 2: Card Selection (Pointer Network)
        self.card_selector = PointerNetwork(config.d_model, config.n_heads // 2)

        # Level 3: Mode Selection
        self.mode_selector = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_ff),  # context + selected card
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, 10),  # Max 10 modes
        )

        # X value prediction (for X spells)
        self.x_value_head = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_ff // 2),
            nn.GELU(),
            nn.Linear(config.d_ff // 2, 1),
            nn.Softplus(),  # Ensure positive
        )

        # Level 4: Target Selection (Pointer Network)
        self.target_selector = PointerNetwork(config.d_model, config.n_heads // 2)

        # Value Head
        self.value_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, 1),
        )

        # Conditioning embeddings for auto-regressive
        self.action_type_embedding = nn.Embedding(len(ActionCategory), config.d_model)

    def encode_state(self,
                     card_embeddings: torch.Tensor,
                     card_mask: torch.Tensor,
                     global_features: torch.Tensor) -> torch.Tensor:
        """
        Encode the game state.

        Returns:
            state_encoding: (batch, d_model) - Aggregated state representation
        """
        batch_size = card_embeddings.size(0)

        # Project cards
        x = self.card_projection(card_embeddings)
        x = self.pos_encoding(x)

        # Add CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Add global features to CLS
        global_emb = self.global_proj(global_features)
        x[:, 0, :] = x[:, 0, :] + global_emb

        # Extend mask for CLS
        cls_mask = torch.ones(batch_size, 1, device=card_mask.device)
        full_mask = torch.cat([cls_mask, card_mask], dim=1)

        # Create attention mask (True = masked out)
        attn_mask = (full_mask == 0)

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # Return CLS output
        return x[:, 0, :]

    def forward(self,
                card_embeddings: torch.Tensor,
                card_mask: torch.Tensor,
                global_features: torch.Tensor,
                action_type_mask: torch.Tensor,
                card_selection_mask: torch.Tensor,
                mode_mask: Optional[torch.Tensor] = None,
                target_mask: Optional[torch.Tensor] = None,
                target_embeddings: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through hierarchical policy.

        Args:
            card_embeddings: (batch, max_cards, card_dim)
            card_mask: (batch, max_cards)
            global_features: (batch, global_dim)
            action_type_mask: (batch, n_action_types)
            card_selection_mask: (batch, max_cards)
            mode_mask: Optional (batch, max_modes)
            target_mask: Optional (batch, max_targets)
            target_embeddings: Optional (batch, max_targets, card_dim)

        Returns:
            Dict with logits for each level
        """
        # Encode state
        state = self.encode_state(card_embeddings, card_mask, global_features)

        outputs = {}

        # Level 1: Action Type
        action_type_logits = self.action_type_head(state)
        action_type_logits = action_type_logits.masked_fill(action_type_mask == 0, float('-inf'))
        outputs['action_type_logits'] = action_type_logits

        # Level 2: Card Selection
        card_hidden = self.card_projection(card_embeddings)
        card_logits = self.card_selector(state, card_hidden, card_selection_mask)
        outputs['card_logits'] = card_logits

        # Level 3: Mode Selection (conditioned on selected card)
        # For simplicity, use state directly; full implementation would condition on card
        mode_logits = self.mode_selector(
            torch.cat([state, state], dim=-1)  # Placeholder for card-conditioned
        )
        if mode_mask is not None:
            mode_logits = mode_logits.masked_fill(mode_mask == 0, float('-inf'))
        outputs['mode_logits'] = mode_logits

        # X value prediction
        x_value = self.x_value_head(torch.cat([state, state], dim=-1))
        outputs['x_value'] = x_value

        # Level 4: Target Selection
        if target_embeddings is not None and target_mask is not None:
            target_hidden = self.card_projection(target_embeddings)
            target_logits = self.target_selector(state, target_hidden, target_mask)
            outputs['target_logits'] = target_logits

        # Value
        outputs['value'] = self.value_head(state)

        return outputs

    def select_action(self,
                      card_embeddings: torch.Tensor,
                      card_mask: torch.Tensor,
                      global_features: torch.Tensor,
                      action_specs: List[ActionSpec],
                      deterministic: bool = False) -> Tuple[HierarchicalAction, Dict]:
        """
        Select a complete hierarchical action.

        Args:
            card_embeddings, card_mask, global_features: State tensors
            action_specs: Parsed available actions
            deterministic: If True, select argmax; else sample

        Returns:
            action: HierarchicalAction
            info: Dict with log_probs, value, etc.
        """
        self.eval()

        with torch.no_grad():
            # Build masks from action specs
            action_type_mask = self._build_action_type_mask(action_specs)
            card_mask_refined = self._build_card_mask(action_specs, card_mask)

            # Encode state
            state = self.encode_state(card_embeddings, card_mask, global_features)

            # Level 1: Select action type
            action_type_logits = self.action_type_head(state)
            action_type_logits = action_type_logits.masked_fill(
                action_type_mask == 0, float('-inf')
            )
            action_type_probs = F.softmax(action_type_logits, dim=-1)

            if deterministic:
                action_type_idx = action_type_probs.argmax(dim=-1)
            else:
                action_type_idx = torch.multinomial(action_type_probs, 1).squeeze(-1)

            selected_category = ActionCategory(action_type_idx.item())

            # If PASS, we're done
            if selected_category == ActionCategory.PASS:
                return HierarchicalAction(category=ActionCategory.PASS), {
                    'value': self.value_head(state),
                    'log_prob': torch.log(action_type_probs[0, action_type_idx] + 1e-8),
                }

            # Level 2: Select card
            # Filter to cards of selected category
            category_card_mask = self._build_category_card_mask(
                action_specs, selected_category, card_mask_refined
            )

            card_hidden = self.card_projection(card_embeddings)
            card_logits = self.card_selector(state, card_hidden, category_card_mask)
            card_probs = F.softmax(card_logits, dim=-1)

            if deterministic:
                card_idx = card_probs.argmax(dim=-1)
            else:
                card_idx = torch.multinomial(card_probs, 1).squeeze(-1)

            # Get selected action spec
            selected_spec = None
            for spec in action_specs:
                if spec.card_index == card_idx.item() and spec.category == selected_category:
                    selected_spec = spec
                    break

            # Level 3: Mode selection (if needed)
            mode_idx = 0
            if selected_spec and selected_spec.requires_mode:
                # Simplified: just pick first mode
                mode_idx = 0

            # Level 4: X value (if needed)
            x_value = 0
            if selected_spec and selected_spec.requires_x_value:
                x_pred = self.x_value_head(torch.cat([state, state], dim=-1))
                x_value = min(int(x_pred.item()), selected_spec.max_x_value)

            # Level 5: Target selection (if needed)
            target_indices = []
            # Simplified: target selection handled by environment

            action = HierarchicalAction(
                category=selected_category,
                card_index=card_idx.item(),
                mode_index=mode_idx,
                x_value=x_value,
                target_indices=target_indices,
            )

            # Compute log probability
            log_prob = torch.log(action_type_probs[0, action_type_idx] + 1e-8)
            log_prob = log_prob + torch.log(card_probs[0, card_idx] + 1e-8)

            return action, {
                'value': self.value_head(state),
                'log_prob': log_prob,
                'action_type_probs': action_type_probs,
                'card_probs': card_probs,
            }

    def _build_action_type_mask(self, action_specs: List[ActionSpec]) -> torch.Tensor:
        """Build mask for valid action types."""
        mask = torch.zeros(1, len(ActionCategory))
        mask[0, ActionCategory.PASS.value] = 1  # Pass always valid

        for spec in action_specs:
            mask[0, spec.category.value] = 1

        return mask

    def _build_card_mask(self, action_specs: List[ActionSpec],
                         base_mask: torch.Tensor) -> torch.Tensor:
        """Build mask for valid card selections."""
        mask = torch.zeros_like(base_mask)

        for spec in action_specs:
            if spec.card_index >= 0 and spec.card_index < mask.size(1):
                mask[0, spec.card_index] = 1

        return mask

    def _build_category_card_mask(self, action_specs: List[ActionSpec],
                                  category: ActionCategory,
                                  base_mask: torch.Tensor) -> torch.Tensor:
        """Build mask for cards of a specific category."""
        mask = torch.zeros_like(base_mask)

        for spec in action_specs:
            if spec.category == category and spec.card_index >= 0:
                if spec.card_index < mask.size(1):
                    mask[0, spec.card_index] = 1

        return mask


# =============================================================================
# COMPLEX CARD MECHANICS HANDLER
# =============================================================================

class ComplexMechanicsEncoder:
    """
    Handles encoding of complex card mechanics for 2024-2025 sets.

    New mechanics to handle:
    - Prototype (different stats for reduced cost)
    - Bargain (sacrifice as additional cost)
    - Craft (exile cards to transform)
    - Plot (exile and cast later for free)
    - Disguise/Cloak (face-down variants)
    - Offspring (create token copies)
    - Saddle (tap creatures to attack)
    - Impending (countdown counters)
    """

    COMPLEX_KEYWORDS_2024_2025 = {
        # Duskmourn: House of Horror (2024)
        'impending': {'type': 'countdown', 'params': ['counters']},
        'eerie': {'type': 'trigger', 'params': []},
        'survival': {'type': 'trigger', 'params': []},

        # Bloomburrow (2024)
        'offspring': {'type': 'cost', 'params': ['mana_cost']},
        'valiant': {'type': 'trigger', 'params': []},
        'expend': {'type': 'threshold', 'params': ['amount']},
        'forage': {'type': 'action', 'params': []},
        'gift': {'type': 'decision', 'params': ['gift_type']},

        # Outlaws of Thunder Junction (2024)
        'plot': {'type': 'alt_cost', 'params': ['plot_cost']},
        'spree': {'type': 'modal', 'params': ['mode_costs']},
        'saddle': {'type': 'cost', 'params': ['saddle_n']},
        'crime': {'type': 'trigger', 'params': []},

        # Murders at Karlov Manor (2024)
        'disguise': {'type': 'face_down', 'params': ['disguise_cost']},
        'cloak': {'type': 'face_down', 'params': []},
        'collect_evidence': {'type': 'cost', 'params': ['evidence_n']},
        'suspect': {'type': 'status', 'params': []},

        # Lost Caverns of Ixalan (2023)
        'craft': {'type': 'transform', 'params': ['craft_cost', 'exile_req']},
        'descend': {'type': 'threshold', 'params': ['descend_n']},
        'discover': {'type': 'cascade_like', 'params': ['discover_n']},
        'explore': {'type': 'action', 'params': []},

        # Wilds of Eldraine (2023)
        'bargain': {'type': 'cost', 'params': []},
        'celebration': {'type': 'trigger', 'params': []},
        'role': {'type': 'token', 'params': ['role_type']},

        # Aetherdrift (2025)
        'start_your_engines': {'type': 'vehicle_trigger', 'params': []},
        'exhaust': {'type': 'cost', 'params': []},
    }

    def __init__(self):
        # Build keyword to index mapping
        self.keyword_to_idx = {kw: i for i, kw in enumerate(self.COMPLEX_KEYWORDS_2024_2025.keys())}

    def encode_complex_mechanics(self, card_data: Dict) -> np.ndarray:
        """
        Encode complex mechanics from card data.

        Returns additional feature vector for complex mechanics.
        """
        features = np.zeros(len(self.COMPLEX_KEYWORDS_2024_2025) + 10, dtype=np.float32)

        oracle_text = card_data.get('oracle_text', '').lower()
        keywords = card_data.get('keywords', [])

        # Check for complex keywords
        for kw, spec in self.COMPLEX_KEYWORDS_2024_2025.items():
            if kw in oracle_text or any(kw in k.lower() for k in keywords):
                idx = self.keyword_to_idx[kw]
                features[idx] = 1.0

                # Extract numeric parameters if present
                if spec['type'] == 'countdown':
                    # Extract counter number
                    import re
                    match = re.search(rf'{kw}\s+(\d+)', oracle_text)
                    if match:
                        features[len(self.keyword_to_idx) + 0] = int(match.group(1)) / 10.0

                elif spec['type'] == 'threshold':
                    match = re.search(rf'{kw}\s+(\d+)', oracle_text)
                    if match:
                        features[len(self.keyword_to_idx) + 1] = int(match.group(1)) / 10.0

        return features

    def get_mechanic_dimension(self) -> int:
        """Get total dimension of complex mechanics encoding."""
        return len(self.COMPLEX_KEYWORDS_2024_2025) + 10


# =============================================================================
# TESTS
# =============================================================================

def test_hierarchical_policy():
    """Test the hierarchical policy network."""
    print("Testing Hierarchical Policy Network")
    print("=" * 60)

    config = TransformerConfig(d_model=128, n_heads=4, n_layers=2)
    policy = HierarchicalPolicyNetwork(config)

    # Count parameters
    params = sum(p.numel() for p in policy.parameters())
    print(f"Total parameters: {params:,}")

    # Create dummy inputs
    batch_size = 2
    max_cards = 20

    card_embeddings = torch.randn(batch_size, max_cards, config.card_embedding_dim)
    card_mask = torch.ones(batch_size, max_cards)
    card_mask[:, 10:] = 0  # Only 10 cards
    global_features = torch.randn(batch_size, config.global_feature_dim)
    action_type_mask = torch.ones(batch_size, len(ActionCategory))
    card_selection_mask = card_mask.clone()

    # Forward pass
    outputs = policy(
        card_embeddings, card_mask, global_features,
        action_type_mask, card_selection_mask
    )

    print("\nOutput shapes:")
    for k, v in outputs.items():
        print(f"  {k}: {v.shape}")

    # Test action selection
    print("\nTesting action selection...")
    action_specs = [
        ActionSpec(category=ActionCategory.PLAY_LAND, card_index=0, card_name="Mountain"),
        ActionSpec(category=ActionCategory.CAST_SPELL, card_index=1, card_name="Lightning Bolt"),
        ActionSpec(category=ActionCategory.CAST_SPELL, card_index=2, card_name="Goblin Guide"),
    ]

    action, info = policy.select_action(
        card_embeddings[:1], card_mask[:1], global_features[:1],
        action_specs, deterministic=True
    )

    print(f"  Selected action: {action.category.name}, card_idx={action.card_index}")
    print(f"  Value: {info['value'].item():.4f}")
    print(f"  Log prob: {info['log_prob'].item():.4f}")

    print("\n" + "=" * 60)
    print("Hierarchical policy test completed!")


def test_complex_mechanics():
    """Test complex mechanics encoder."""
    print("\nTesting Complex Mechanics Encoder")
    print("=" * 60)

    encoder = ComplexMechanicsEncoder()
    print(f"Supported mechanics: {len(encoder.COMPLEX_KEYWORDS_2024_2025)}")
    print(f"Feature dimension: {encoder.get_mechanic_dimension()}")

    # Test with sample cards
    test_cards = [
        {
            'name': 'Test Card with Plot',
            'oracle_text': 'Plot {2}{R} (You may pay {2}{R} and exile this card. You may cast it later without paying its mana cost.)',
            'keywords': ['Plot'],
        },
        {
            'name': 'Test Card with Impending',
            'oracle_text': 'Impending 4 (If you cast this for its impending cost, it enters with 4 time counters.)',
            'keywords': [],
        },
        {
            'name': 'Test Card with Offspring',
            'oracle_text': 'Offspring {2} (You may pay an additional {2} as you cast this spell. If you do, create a 1/1 copy.)',
            'keywords': ['Offspring'],
        },
    ]

    for card in test_cards:
        features = encoder.encode_complex_mechanics(card)
        nonzero = np.nonzero(features)[0]
        print(f"\n{card['name']}:")
        print(f"  Non-zero features: {len(nonzero)}")


if __name__ == "__main__":
    test_hierarchical_policy()
    test_complex_mechanics()
