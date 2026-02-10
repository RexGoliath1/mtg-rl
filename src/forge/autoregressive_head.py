"""
Auto-Regressive Action Head (AlphaStar-style structured action decomposition).

Replaces the flat 203-action policy with a structured hierarchy:

Step 1: Action Type (8 classes)
    PASS, CAST_SPELL, ACTIVATE_ABILITY, DECLARE_ATTACKER,
    DECLARE_BLOCKER, PAY_COST, SELECT_TARGET, SELECT_MODE

Step 2: Card Selection (pointer over cards in relevant zone)
    Only for CAST_SPELL, ACTIVATE_ABILITY, DECLARE_ATTACKER, DECLARE_BLOCKER

Step 3: Target/Mode Selection (pointer over legal targets or mode index)
    Only for SELECT_TARGET, SELECT_MODE

Each step conditions on previous selections (auto-regressive).
Legal action masking at each level.

Usage:
    head = AutoRegressiveActionHead(state_dim=768, card_dim=512, max_cards=50)

    # During training (teacher forcing with flat action decomposed):
    log_prob = head.log_prob(state, card_embeddings, flat_action, action_mask)

    # During inference (auto-regressive sampling):
    action_type, card_idx, target_idx = head.sample(state, card_embeddings, legal_masks)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# ACTION TYPE ENUM (matches policy_value_heads.ActionType but simplified)
# =============================================================================

NUM_ACTION_TYPES = 8

ACTION_PASS = 0
ACTION_CAST_SPELL = 1
ACTION_ACTIVATE = 2
ACTION_ATTACK = 3
ACTION_BLOCK = 4
ACTION_PAY_COST = 5
ACTION_TARGET = 6
ACTION_MODE = 7

# Action types that require card selection
CARD_SELECTION_TYPES = {ACTION_CAST_SPELL, ACTION_ACTIVATE, ACTION_ATTACK, ACTION_BLOCK}

# Action types that require target/mode selection
TARGET_SELECTION_TYPES = {ACTION_TARGET, ACTION_MODE}


@dataclass
class AutoRegressiveConfig:
    """Configuration for the auto-regressive action head."""
    state_dim: int = 768       # Input state dimension
    card_dim: int = 512        # Per-card embedding dimension
    max_cards: int = 50        # Maximum cards for pointer network
    max_targets: int = 20      # Maximum targets for pointer network
    max_modes: int = 5         # Maximum modes to choose from
    hidden_dim: int = 256      # Hidden dim for sub-heads
    n_action_types: int = NUM_ACTION_TYPES
    dropout: float = 0.1


# =============================================================================
# POINTER NETWORK
# =============================================================================

class PointerNetwork(nn.Module):
    """
    Attention-based pointer network for selecting from variable-size sets.

    Uses the state as a query to attend over a set of key embeddings
    (e.g., card embeddings), producing selection logits.
    """

    def __init__(self, query_dim: int, key_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute pointer logits.

        Args:
            query: [batch, query_dim] - context vector
            keys: [batch, num_items, key_dim] - item embeddings
            mask: [batch, num_items] - 1 for valid, 0 for invalid

        Returns:
            logits: [batch, num_items] - unnormalized selection scores
        """
        # Project query and keys to shared space
        q = self.query_proj(query).unsqueeze(1)  # [batch, 1, hidden]
        k = self.key_proj(keys)                   # [batch, num_items, hidden]

        # Additive attention
        scores = self.v(torch.tanh(q + k)).squeeze(-1)  # [batch, num_items]

        # Mask invalid items
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        return scores


# =============================================================================
# AUTO-REGRESSIVE ACTION HEAD
# =============================================================================

class AutoRegressiveActionHead(nn.Module):
    """
    Structured action decomposition with auto-regressive conditioning.

    Level 1: type_head -> action type logits
    Level 2: card_pointer -> card selection (conditioned on type)
    Level 3: target_pointer -> target selection (conditioned on type + card)
    """

    def __init__(self, config: Optional[AutoRegressiveConfig] = None):
        super().__init__()
        self.config = config or AutoRegressiveConfig()
        c = self.config

        # Level 1: Action type classification
        self.type_head = nn.Sequential(
            nn.Linear(c.state_dim, c.hidden_dim),
            nn.LayerNorm(c.hidden_dim),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden_dim, c.n_action_types),
        )

        # Type embedding (for conditioning subsequent levels)
        self.type_embedding = nn.Embedding(c.n_action_types, c.hidden_dim)

        # Level 2: Card selection pointer
        # Query: state + type_embedding
        self.card_pointer = PointerNetwork(
            query_dim=c.state_dim + c.hidden_dim,
            key_dim=c.card_dim,
            hidden_dim=c.hidden_dim,
        )

        # Level 3: Target/mode selection
        # Query: state + type_embedding + selected_card_embedding
        self.target_pointer = PointerNetwork(
            query_dim=c.state_dim + c.hidden_dim + c.card_dim,
            key_dim=c.card_dim,  # Targets are also card embeddings
            hidden_dim=c.hidden_dim,
        )

        # Mode selection (simpler: just a linear head)
        self.mode_head = nn.Sequential(
            nn.Linear(c.state_dim + c.hidden_dim, c.hidden_dim),
            nn.GELU(),
            nn.Linear(c.hidden_dim, c.max_modes),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward_type(
        self,
        state: torch.Tensor,
        type_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Level 1: Compute action type logits.

        Args:
            state: [batch, state_dim]
            type_mask: [batch, n_action_types] - legal action types

        Returns:
            type_logits: [batch, n_action_types]
        """
        logits = self.type_head(state)
        if type_mask is not None:
            logits = logits.masked_fill(type_mask == 0, float('-inf'))
        return logits

    def forward_card(
        self,
        state: torch.Tensor,
        action_type: torch.Tensor,
        card_embeddings: torch.Tensor,
        card_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Level 2: Compute card selection logits (conditioned on action type).

        Args:
            state: [batch, state_dim]
            action_type: [batch] - selected action type indices
            card_embeddings: [batch, max_cards, card_dim]
            card_mask: [batch, max_cards] - valid cards

        Returns:
            card_logits: [batch, max_cards]
        """
        type_emb = self.type_embedding(action_type)  # [batch, hidden_dim]
        query = torch.cat([state, type_emb], dim=-1)
        return self.card_pointer(query, card_embeddings, card_mask)

    def forward_target(
        self,
        state: torch.Tensor,
        action_type: torch.Tensor,
        selected_card_emb: torch.Tensor,
        target_embeddings: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Level 3: Compute target selection logits.

        Args:
            state: [batch, state_dim]
            action_type: [batch] - action type
            selected_card_emb: [batch, card_dim] - embedding of selected card
            target_embeddings: [batch, max_targets, card_dim]
            target_mask: [batch, max_targets] - valid targets

        Returns:
            target_logits: [batch, max_targets]
        """
        type_emb = self.type_embedding(action_type)
        query = torch.cat([state, type_emb, selected_card_emb], dim=-1)
        return self.target_pointer(query, target_embeddings, target_mask)

    def forward_mode(
        self,
        state: torch.Tensor,
        action_type: torch.Tensor,
        mode_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Level 3 (alternative): Compute mode selection logits.

        Args:
            state: [batch, state_dim]
            action_type: [batch]
            mode_mask: [batch, max_modes] - valid modes

        Returns:
            mode_logits: [batch, max_modes]
        """
        type_emb = self.type_embedding(action_type)
        query = torch.cat([state, type_emb], dim=-1)
        logits = self.mode_head(query)
        if mode_mask is not None:
            logits = logits.masked_fill(mode_mask == 0, float('-inf'))
        return logits

    def sample(
        self,
        state: torch.Tensor,
        card_embeddings: torch.Tensor,
        type_mask: Optional[torch.Tensor] = None,
        card_mask: Optional[torch.Tensor] = None,
        target_embeddings: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        mode_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Auto-regressive sampling of full action.

        Returns:
            action_type: [batch] - selected type
            card_idx: [batch] - selected card (-1 if N/A)
            target_idx: [batch] - selected target (-1 if N/A)
            total_log_prob: [batch] - sum of log probs across levels
        """
        batch_size = state.shape[0]
        device = state.device

        # Level 1: Sample action type
        type_logits = self.forward_type(state, type_mask) / temperature
        type_probs = F.softmax(type_logits, dim=-1)
        type_dist = torch.distributions.Categorical(type_probs)
        action_type = type_dist.sample()
        type_log_prob = type_dist.log_prob(action_type)

        total_log_prob = type_log_prob
        card_idx = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        target_idx = torch.full((batch_size,), -1, dtype=torch.long, device=device)

        # Level 2: Sample card (for card-selecting action types)
        needs_card = torch.tensor(
            [t.item() in CARD_SELECTION_TYPES for t in action_type],
            dtype=torch.bool, device=device,
        )
        if needs_card.any():
            card_logits = self.forward_card(
                state, action_type, card_embeddings, card_mask
            ) / temperature
            card_probs = F.softmax(card_logits, dim=-1)
            card_dist = torch.distributions.Categorical(card_probs)
            sampled_cards = card_dist.sample()
            card_log_prob = card_dist.log_prob(sampled_cards)

            card_idx = torch.where(needs_card, sampled_cards, card_idx)
            total_log_prob = total_log_prob + torch.where(
                needs_card, card_log_prob, torch.zeros_like(card_log_prob)
            )

        # Level 3: Sample target (for target-selecting action types)
        needs_target = (action_type == ACTION_TARGET)
        if needs_target.any() and target_embeddings is not None:
            # Get selected card embeddings for conditioning
            safe_card_idx = card_idx.clamp(min=0)
            selected_card_emb = card_embeddings[
                torch.arange(batch_size, device=device), safe_card_idx
            ]

            target_logits = self.forward_target(
                state, action_type, selected_card_emb,
                target_embeddings, target_mask
            ) / temperature
            target_probs = F.softmax(target_logits, dim=-1)
            target_dist = torch.distributions.Categorical(target_probs)
            sampled_targets = target_dist.sample()
            target_log_prob = target_dist.log_prob(sampled_targets)

            target_idx = torch.where(needs_target, sampled_targets, target_idx)
            total_log_prob = total_log_prob + torch.where(
                needs_target, target_log_prob, torch.zeros_like(target_log_prob)
            )

        # Level 3 (alt): Mode selection
        needs_mode = (action_type == ACTION_MODE)
        if needs_mode.any():
            mode_logits = self.forward_mode(
                state, action_type, mode_mask
            ) / temperature
            mode_probs = F.softmax(mode_logits, dim=-1)
            mode_dist = torch.distributions.Categorical(mode_probs)
            sampled_modes = mode_dist.sample()
            mode_log_prob = mode_dist.log_prob(sampled_modes)

            target_idx = torch.where(needs_mode, sampled_modes, target_idx)
            total_log_prob = total_log_prob + torch.where(
                needs_mode, mode_log_prob, torch.zeros_like(mode_log_prob)
            )

        return action_type, card_idx, target_idx, total_log_prob
