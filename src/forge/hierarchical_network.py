"""
Hierarchical AlphaZero Network - wraps encoder + strategic core + turn planner.

Combines:
1. ForgeGameStateEncoder -> 768-dim state embedding
2. StrategicCoreGRU -> 256-dim game trajectory context (updated per turn)
3. TurnPlannerMLP -> 128-dim phase-level tactical plan
4. Context projection: cat(state, z_game, z_turn) -> 768-dim enriched context
5. Existing PolicyHead + ValueHead (unchanged)

Backward compatible: flat AlphaZero checkpoints load cleanly (new modules
get default init). The hierarchical modules are additive, not replacing.

Usage:
    network = HierarchicalAlphaZeroNetwork()

    # Game loop:
    z_game = network.initial_game_state(batch_size=1)
    prev_turn = torch.zeros(1)
    prev_phase = torch.zeros(1)

    for decision in game:
        policy, value, z_game = network(
            encoder_inputs, z_game, current_turn, prev_turn,
            current_phase, prev_phase, action_mask
        )
        prev_turn = current_turn
        prev_phase = current_phase
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.forge.game_state_encoder import ForgeGameStateEncoder, GameStateConfig
from src.forge.policy_value_heads import PolicyHead, PolicyValueConfig, ValueHead
from src.forge.strategic_core import StrategicCoreGRU
from src.forge.turn_planner import TurnPlannerMLP


@dataclass
class HierarchicalConfig:
    """Configuration for the hierarchical network."""
    # Encoder output (input to hierarchy)
    state_dim: int = 768

    # Strategic Core GRU
    strategic_hidden_dim: int = 256

    # Turn Planner MLP
    tactical_output_dim: int = 128
    num_phases: int = 14

    # Context projection output (must match state_dim for head compatibility)
    context_dim: int = 768


class HierarchicalAlphaZeroNetwork(nn.Module):
    """
    Full hierarchical network with strategic reasoning.

    Adds game-trajectory awareness (GRU) and phase-level planning (MLP)
    on top of the flat AlphaZero architecture. The enriched context is
    projected back to 768-dim so existing policy/value heads work unchanged.
    """

    def __init__(
        self,
        encoder_config: Optional[GameStateConfig] = None,
        head_config: Optional[PolicyValueConfig] = None,
        hier_config: Optional[HierarchicalConfig] = None,
        num_players: int = 2,
    ):
        super().__init__()

        # Configs
        self.encoder_config = encoder_config or GameStateConfig()
        self.hier_config = hier_config or HierarchicalConfig()
        head_config = head_config or PolicyValueConfig()
        head_config.state_dim = self.hier_config.context_dim

        # Core encoder
        self.encoder = ForgeGameStateEncoder(self.encoder_config)

        # Hierarchical modules
        self.strategic_core = StrategicCoreGRU(
            state_dim=self.encoder_config.output_dim,
            hidden_dim=self.hier_config.strategic_hidden_dim,
        )
        self.turn_planner = TurnPlannerMLP(
            strategic_dim=self.hier_config.strategic_hidden_dim,
            state_dim=self.encoder_config.output_dim,
            num_phases=self.hier_config.num_phases,
            output_dim=self.hier_config.tactical_output_dim,
        )

        # Context projection: cat(state, z_game, z_turn) -> context_dim
        proj_input = (
            self.encoder_config.output_dim +
            self.hier_config.strategic_hidden_dim +
            self.hier_config.tactical_output_dim
        )
        self.context_proj = nn.Sequential(
            nn.Linear(proj_input, self.hier_config.context_dim),
            nn.LayerNorm(self.hier_config.context_dim),
            nn.GELU(),
        )

        # Standard heads (same architecture as flat AlphaZero)
        self.policy_head = PolicyHead(head_config)
        self.value_head = ValueHead(head_config, num_players)

        self._init_context_proj()

    def _init_context_proj(self):
        for module in self.context_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        prev_z_game: torch.Tensor,
        current_turn: torch.Tensor,
        prev_turn: torch.Tensor,
        current_phase: torch.Tensor,
        prev_phase: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        **encoder_inputs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full hierarchical forward pass.

        Args:
            prev_z_game: [batch, strategic_hidden_dim] - previous GRU state
            current_turn: [batch] - current turn number
            prev_turn: [batch] - previous turn number
            current_phase: [batch] - current phase index
            prev_phase: [batch] - previous phase index
            action_mask: [batch, action_dim] - optional legal action mask
            **encoder_inputs: inputs for ForgeGameStateEncoder

        Returns:
            policy: [batch, action_dim] - action probabilities
            value: [batch, 1] or [batch, num_players] - state value
            z_game: [batch, strategic_hidden_dim] - updated GRU state
        """
        # 1. Encode game state
        state = self.encoder(**encoder_inputs)  # [batch, 768]

        # 2. Update strategic context (only at turn boundaries)
        z_game = self.strategic_core(state, prev_z_game, current_turn, prev_turn)

        # 3. Compute tactical plan (phase-level)
        z_turn = self.turn_planner(z_game, state, current_phase, prev_phase)

        # 4. Project enriched context
        context = self.context_proj(
            torch.cat([state, z_game, z_turn], dim=-1)
        )  # [batch, 768]

        # 5. Policy and value from enriched context
        policy = self.policy_head(context, action_mask)
        value = self.value_head(context)

        return policy, value, z_game

    def forward_flat(
        self,
        action_mask: Optional[torch.Tensor] = None,
        **encoder_inputs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Flat forward pass (no hierarchical context).

        For backward compatibility and ablation studies. Uses zero
        strategic/tactical context, equivalent to the flat AlphaZero.
        """
        state = self.encoder(**encoder_inputs)
        batch_size = state.shape[0]
        device = state.device

        # Zero context
        z_game = torch.zeros(batch_size, self.hier_config.strategic_hidden_dim, device=device)
        z_turn = torch.zeros(batch_size, self.hier_config.tactical_output_dim, device=device)

        context = self.context_proj(
            torch.cat([state, z_game, z_turn], dim=-1)
        )

        policy = self.policy_head(context, action_mask)
        value = self.value_head(context)

        return policy, value

    def initial_game_state(
        self,
        batch_size: int,
        device: torch.device = None,
    ) -> torch.Tensor:
        """Create zero GRU state for the start of a game."""
        return self.strategic_core.initial_state(batch_size, device)

    def param_count_by_component(self) -> dict:
        """Report parameter count per component."""
        def _count(module):
            return sum(p.numel() for p in module.parameters())

        return {
            'encoder': _count(self.encoder),
            'strategic_core': _count(self.strategic_core),
            'turn_planner': _count(self.turn_planner),
            'context_proj': _count(self.context_proj),
            'policy_head': _count(self.policy_head),
            'value_head': _count(self.value_head),
            'total': _count(self),
        }

    def load_flat_checkpoint(self, path: str, device: torch.device = None):
        """
        Load a flat AlphaZero checkpoint into this hierarchical network.

        Maps encoder/policy_head/value_head weights from the flat model.
        Hierarchical modules (strategic_core, turn_planner, context_proj)
        keep their random initialization.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)

        # Load matching keys, skip missing (hierarchical modules)
        own_state = self.state_dict()
        loaded = 0
        skipped = 0
        for name, param in state_dict.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded += 1
            else:
                skipped += 1

        return {'loaded': loaded, 'skipped': skipped}
