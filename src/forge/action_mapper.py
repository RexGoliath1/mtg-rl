"""
Action Mapper - converts between flat 203-action space and structured actions.

Maps flat action indices to (type, card_idx, target_idx) tuples and back.
Used for training the auto-regressive head with flat action labels.

The flat action space layout (from policy_value_heads.ActionConfig):
    [0]       pass
    [1]       mulligan
    [2]       concede
    [3-17]    cast spell (15 hand slots)
    [18-67]   activate ability (50 battlefield slots)
    [68-117]  attack (50 battlefield slots)
    [118-167] block (50 battlefield slots)
    [168-187] choose target (20 target slots)
    [188-192] choose mode (5 mode slots)
    [193-202] pay cost (10 cost slots)

Usage:
    mapper = ActionMapper()
    action_type, card_idx, target_idx = mapper.flat_to_structured(42)
    flat_idx = mapper.structured_to_flat(ACTION_CAST_SPELL, card_idx=3)
"""

from dataclasses import dataclass
from typing import Tuple

import torch

from src.forge.autoregressive_head import (
    ACTION_ACTIVATE,
    ACTION_ATTACK,
    ACTION_BLOCK,
    ACTION_CAST_SPELL,
    ACTION_MODE,
    ACTION_PASS,
    ACTION_PAY_COST,
    ACTION_TARGET,
)


@dataclass
class FlatActionLayout:
    """Defines the flat action space boundaries."""
    # From ActionConfig defaults
    num_special: int = 3
    max_hand: int = 15
    max_battlefield: int = 50
    max_targets: int = 20
    max_modes: int = 5
    max_costs: int = 10

    @property
    def total(self) -> int:
        return (self.num_special + self.max_hand +
                self.max_battlefield * 3 +
                self.max_targets + self.max_modes + self.max_costs)

    # Offset boundaries
    @property
    def cast_start(self) -> int:
        return self.num_special

    @property
    def activate_start(self) -> int:
        return self.cast_start + self.max_hand

    @property
    def attack_start(self) -> int:
        return self.activate_start + self.max_battlefield

    @property
    def block_start(self) -> int:
        return self.attack_start + self.max_battlefield

    @property
    def target_start(self) -> int:
        return self.block_start + self.max_battlefield

    @property
    def mode_start(self) -> int:
        return self.target_start + self.max_targets

    @property
    def cost_start(self) -> int:
        return self.mode_start + self.max_modes


class ActionMapper:
    """Maps between flat action indices and structured (type, card, target) tuples."""

    def __init__(self, layout: FlatActionLayout = None):
        self.layout = layout or FlatActionLayout()

    def flat_to_structured(self, flat_idx: int) -> Tuple[int, int, int]:
        """
        Convert flat action index to (action_type, card_idx, target_idx).

        card_idx and target_idx are -1 when not applicable.
        """
        L = self.layout

        if flat_idx == 0:
            return (ACTION_PASS, -1, -1)
        elif flat_idx == 1:
            return (ACTION_PASS, -1, -1)  # mulligan mapped to pass type
        elif flat_idx == 2:
            return (ACTION_PASS, -1, -1)  # concede mapped to pass type

        elif L.cast_start <= flat_idx < L.activate_start:
            card_idx = flat_idx - L.cast_start
            return (ACTION_CAST_SPELL, card_idx, -1)

        elif L.activate_start <= flat_idx < L.attack_start:
            card_idx = flat_idx - L.activate_start
            return (ACTION_ACTIVATE, card_idx, -1)

        elif L.attack_start <= flat_idx < L.block_start:
            card_idx = flat_idx - L.attack_start
            return (ACTION_ATTACK, card_idx, -1)

        elif L.block_start <= flat_idx < L.target_start:
            card_idx = flat_idx - L.block_start
            return (ACTION_BLOCK, card_idx, -1)

        elif L.target_start <= flat_idx < L.mode_start:
            target_idx = flat_idx - L.target_start
            return (ACTION_TARGET, -1, target_idx)

        elif L.mode_start <= flat_idx < L.cost_start:
            mode_idx = flat_idx - L.mode_start
            return (ACTION_MODE, -1, mode_idx)

        elif L.cost_start <= flat_idx < self.layout_end:
            cost_idx = flat_idx - L.cost_start
            return (ACTION_PAY_COST, cost_idx, -1)

        else:
            return (ACTION_PASS, -1, -1)

    @property
    def layout_end(self) -> int:
        return self.layout.cost_start + self.layout.max_costs

    def structured_to_flat(
        self, action_type: int, card_idx: int = -1, target_idx: int = -1
    ) -> int:
        """Convert structured action back to flat index."""
        L = self.layout

        if action_type == ACTION_PASS:
            return 0
        elif action_type == ACTION_CAST_SPELL:
            return L.cast_start + max(card_idx, 0)
        elif action_type == ACTION_ACTIVATE:
            return L.activate_start + max(card_idx, 0)
        elif action_type == ACTION_ATTACK:
            return L.attack_start + max(card_idx, 0)
        elif action_type == ACTION_BLOCK:
            return L.block_start + max(card_idx, 0)
        elif action_type == ACTION_TARGET:
            return L.target_start + max(target_idx, 0)
        elif action_type == ACTION_MODE:
            return L.mode_start + max(target_idx, 0)
        elif action_type == ACTION_PAY_COST:
            return L.cost_start + max(card_idx, 0)
        else:
            return 0

    def batch_flat_to_structured(
        self, flat_actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vectorized conversion of flat actions to structured.

        Args:
            flat_actions: [batch] flat action indices

        Returns:
            action_types: [batch] action type indices
            card_indices: [batch] card indices (-1 if N/A)
            target_indices: [batch] target indices (-1 if N/A)
        """
        batch_size = flat_actions.shape[0]
        device = flat_actions.device
        L = self.layout

        action_types = torch.zeros(batch_size, dtype=torch.long, device=device)
        card_indices = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        target_indices = torch.full((batch_size,), -1, dtype=torch.long, device=device)

        # Pass/mulligan/concede
        special_mask = flat_actions < L.num_special
        action_types[special_mask] = ACTION_PASS

        # Cast spell
        cast_mask = (flat_actions >= L.cast_start) & (flat_actions < L.activate_start)
        action_types[cast_mask] = ACTION_CAST_SPELL
        card_indices[cast_mask] = flat_actions[cast_mask] - L.cast_start

        # Activate ability
        act_mask = (flat_actions >= L.activate_start) & (flat_actions < L.attack_start)
        action_types[act_mask] = ACTION_ACTIVATE
        card_indices[act_mask] = flat_actions[act_mask] - L.activate_start

        # Attack
        atk_mask = (flat_actions >= L.attack_start) & (flat_actions < L.block_start)
        action_types[atk_mask] = ACTION_ATTACK
        card_indices[atk_mask] = flat_actions[atk_mask] - L.attack_start

        # Block
        blk_mask = (flat_actions >= L.block_start) & (flat_actions < L.target_start)
        action_types[blk_mask] = ACTION_BLOCK
        card_indices[blk_mask] = flat_actions[blk_mask] - L.block_start

        # Target
        tgt_mask = (flat_actions >= L.target_start) & (flat_actions < L.mode_start)
        action_types[tgt_mask] = ACTION_TARGET
        target_indices[tgt_mask] = flat_actions[tgt_mask] - L.target_start

        # Mode
        mode_mask = (flat_actions >= L.mode_start) & (flat_actions < L.cost_start)
        action_types[mode_mask] = ACTION_MODE
        target_indices[mode_mask] = flat_actions[mode_mask] - L.mode_start

        # Pay cost
        cost_end = L.cost_start + L.max_costs
        cost_mask = (flat_actions >= L.cost_start) & (flat_actions < cost_end)
        action_types[cost_mask] = ACTION_PAY_COST
        card_indices[cost_mask] = flat_actions[cost_mask] - L.cost_start

        return action_types, card_indices, target_indices
