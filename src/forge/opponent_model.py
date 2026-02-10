"""
Opponent Belief Model - Predicts distribution over possible opponent hands.

Used for informed PIMC (Perfect Information Monte Carlo) determinization
during MCTS search. Instead of uniformly sampling possible opponent hands,
uses a learned model to generate plausible opponent hands.

The model observes:
- Our own hand and board
- Both graveyards and exile zones
- Visible information (hand size, known cards)
- Game trajectory (what opponent has played)

And predicts:
- Per-card probability of being in opponent's hand

Training signal: At game end, the full history is revealed. Train belief
model on actual opponent hands as supervised auxiliary loss.

Usage:
    belief = OpponentBeliefModel(config)

    # During MCTS determinization:
    card_probs = belief(observable_state)
    sampled_hand = informed_determinize(card_probs, deck_remaining, hand_size)

    # During training (auxiliary loss):
    pred_probs = belief(observable_state)
    loss = belief_loss(pred_probs, actual_opponent_hand)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OpponentModelConfig:
    """Configuration for the opponent belief model."""
    # Input dimensions
    state_dim: int = 768       # Observable game state from encoder
    card_vocab_size: int = 50  # Max unique cards to track (in a game context)

    # Architecture
    hidden_dim: int = 256
    n_attention_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1

    # Output
    max_hand_size: int = 15    # Max opponent hand size to model

    # Training
    belief_loss_weight: float = 0.1  # Weight for auxiliary belief loss


class CardAttentionBlock(nn.Module):
    """
    Self-attention over known cards to build context for belief prediction.

    Attends over cards we've seen (our hand, both boards, graveyards)
    to infer what the opponent might have.
    """

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, num_cards, dim] - card embeddings
            mask: [batch, num_cards] - 1 for valid cards, 0 for padding

        Returns:
            [batch, num_cards, dim] - contextualized card embeddings
        """
        # Convert mask for MultiheadAttention (True = ignore)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask == 0)

        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)

        # FFN with residual
        x = self.norm2(x + self.ffn(x))
        return x


class OpponentBeliefModel(nn.Module):
    """
    Predicts what cards the opponent likely has in hand.

    Architecture:
    1. Project game state to card-level queries
    2. Self-attend over known cards for context
    3. Cross-attend from state to known cards
    4. Predict per-card hand probability

    For PIMC determinization: output probabilities guide sampling
    of plausible opponent hands instead of uniform random.
    """

    def __init__(self, config: Optional[OpponentModelConfig] = None):
        super().__init__()
        self.config = config or OpponentModelConfig()
        c = self.config

        # State projection: game state -> belief query
        self.state_proj = nn.Sequential(
            nn.Linear(c.state_dim, c.hidden_dim),
            nn.LayerNorm(c.hidden_dim),
            nn.GELU(),
            nn.Dropout(c.dropout),
        )

        # Card context: self-attention over known cards
        self.card_attention_layers = nn.ModuleList([
            CardAttentionBlock(c.hidden_dim, c.n_attention_heads, c.dropout)
            for _ in range(c.n_layers)
        ])

        # Cross-attention: state queries attend to known cards
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=c.hidden_dim,
            num_heads=c.n_attention_heads,
            dropout=c.dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(c.hidden_dim)

        # Hand size predictor (auxiliary: how many cards does opponent have?)
        self.hand_size_head = nn.Sequential(
            nn.Linear(c.hidden_dim, c.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(c.hidden_dim // 2, c.max_hand_size + 1),  # 0-15
        )

        # Card probability predictor
        # Input: contextualized state + card pool context
        self.card_prob_head = nn.Sequential(
            nn.Linear(c.hidden_dim * 2, c.hidden_dim),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        state: torch.Tensor,
        card_pool_embs: torch.Tensor,
        card_pool_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict opponent hand composition.

        Args:
            state: [batch, state_dim] - observable game state
            card_pool_embs: [batch, num_cards, state_dim] - embeddings of
                cards that could be in opponent's hand (remaining deck pool)
            card_pool_mask: [batch, num_cards] - 1 for valid, 0 for padding

        Returns:
            card_probs: [batch, num_cards] - probability each card is in hand
            hand_size_logits: [batch, max_hand_size+1] - predicted hand size
        """
        batch_size = state.shape[0]
        num_cards = card_pool_embs.shape[1]

        # Project state
        state_proj = self.state_proj(state)  # [batch, hidden]

        # Project card pool to hidden dim
        # Reuse state_proj's linear for card embeddings (they share state_dim)
        card_hidden = self.state_proj(
            card_pool_embs.reshape(-1, self.config.state_dim)
        ).reshape(batch_size, num_cards, self.config.hidden_dim)

        # Self-attention over card pool
        for layer in self.card_attention_layers:
            card_hidden = layer(card_hidden, card_pool_mask)

        # Cross-attention: state queries into card context
        state_query = state_proj.unsqueeze(1)  # [batch, 1, hidden]
        key_padding_mask = None
        if card_pool_mask is not None:
            key_padding_mask = (card_pool_mask == 0)

        cross_out, _ = self.cross_attn(
            state_query, card_hidden, card_hidden,
            key_padding_mask=key_padding_mask,
        )
        context = self.cross_norm(state_query + cross_out).squeeze(1)  # [batch, hidden]

        # Hand size prediction
        hand_size_logits = self.hand_size_head(context)

        # Per-card probability
        # Expand context to match each card
        context_expanded = context.unsqueeze(1).expand(-1, num_cards, -1)
        combined = torch.cat([context_expanded, card_hidden], dim=-1)
        card_logits = self.card_prob_head(combined).squeeze(-1)  # [batch, num_cards]

        # Mask invalid cards
        if card_pool_mask is not None:
            card_logits = card_logits.masked_fill(card_pool_mask == 0, float('-inf'))

        # Sigmoid for independent per-card probabilities
        card_probs = torch.sigmoid(card_logits)

        # Zero out masked positions (sigmoid(-inf) ≈ 0 but let's be explicit)
        if card_pool_mask is not None:
            card_probs = card_probs * card_pool_mask

        return card_probs, hand_size_logits

    def predict_hand(
        self,
        state: torch.Tensor,
        card_pool_embs: torch.Tensor,
        card_pool_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convenience method: return just card probabilities (no hand size).

        Used during MCTS determinization.
        """
        card_probs, _ = self.forward(state, card_pool_embs, card_pool_mask)
        return card_probs


def compute_belief_loss(
    card_probs: torch.Tensor,
    hand_size_logits: torch.Tensor,
    actual_hand_mask: torch.Tensor,
    actual_hand_size: torch.Tensor,
    card_pool_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute supervised belief loss from revealed game history.

    Args:
        card_probs: [batch, num_cards] - predicted per-card probabilities
        hand_size_logits: [batch, max_hand_size+1] - predicted hand size distribution
        actual_hand_mask: [batch, num_cards] - 1 if card was actually in hand
        actual_hand_size: [batch] - actual number of cards in hand (long)
        card_pool_mask: [batch, num_cards] - 1 for valid cards

    Returns:
        total_loss: scalar loss
        metrics: dict with component losses
    """
    # Binary cross-entropy for per-card predictions
    if card_pool_mask is not None:
        # Only compute loss on valid card positions
        valid = card_pool_mask.bool()
        bce_loss = F.binary_cross_entropy(
            card_probs[valid],
            actual_hand_mask[valid].float(),
        )
    else:
        bce_loss = F.binary_cross_entropy(
            card_probs,
            actual_hand_mask.float(),
        )

    # Cross-entropy for hand size prediction
    hand_size_loss = F.cross_entropy(hand_size_logits, actual_hand_size)

    total_loss = bce_loss + 0.5 * hand_size_loss

    metrics = {
        'card_bce_loss': bce_loss.item(),
        'hand_size_loss': hand_size_loss.item(),
        'total_belief_loss': total_loss.item(),
    }

    return total_loss, metrics


def informed_determinize(
    card_probs: torch.Tensor,
    hand_size: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Sample a plausible opponent hand using belief model predictions.

    Instead of uniform random sampling over possible hands (standard PIMC),
    uses the belief model's per-card probabilities to generate more
    realistic opponent hands.

    Args:
        card_probs: [num_cards] - probability each card is in opponent's hand
        hand_size: number of cards to sample for the hand
        temperature: >1 more uniform, <1 more peaked, 1 = raw probs

    Returns:
        hand_indices: [hand_size] - indices of sampled cards
    """
    if hand_size <= 0 or card_probs.sum() == 0:
        return torch.tensor([], dtype=torch.long)

    # Apply temperature
    if temperature != 1.0:
        # Work in log space for numerical stability
        log_probs = torch.log(card_probs.clamp(min=1e-8))
        log_probs = log_probs / temperature
        probs = torch.softmax(log_probs, dim=0)
    else:
        # Normalize to distribution
        total = card_probs.sum()
        if total > 0:
            probs = card_probs / total
        else:
            probs = torch.ones_like(card_probs) / len(card_probs)

    # Sample without replacement
    num_to_sample = min(hand_size, (probs > 0).sum().item())
    if num_to_sample == 0:
        return torch.tensor([], dtype=torch.long)

    hand_indices = torch.multinomial(probs, num_to_sample, replacement=False)
    return hand_indices
