"""
Turn Planner MLP - Phase-level tactical planning.

Produces a 128-dim tactical plan vector updated at each phase change.
Encodes phase-level intent like "in main phase 1, deploy threats" vs
"in combat, attack with evasion creatures".

Takes the strategic context (z_game, 256-dim) + current game state
(768-dim) + phase one-hot (14-dim) as input.

Usage:
    planner = TurnPlannerMLP()
    z_turn = planner(z_game, state, current_phase, prev_phase)
"""

import torch
import torch.nn as nn


# Number of distinct phases (matches game_state_encoder.Phase enum)
NUM_PHASES = 14


class TurnPlannerMLP(nn.Module):
    """
    MLP that produces phase-level tactical plans.

    Updated at phase boundaries within a turn. The output z_turn
    conditions the policy and value heads to make phase-appropriate
    decisions.

    Architecture:
        Input: z_game (256) + state (768) + phase_onehot (14) = 1038
        Hidden: 512 -> 256 -> 128
        Output: z_turn (128) - tactical plan vector
    """

    def __init__(
        self,
        strategic_dim: int = 256,
        state_dim: int = 768,
        num_phases: int = NUM_PHASES,
        output_dim: int = 128,
    ):
        super().__init__()
        self.strategic_dim = strategic_dim
        self.state_dim = state_dim
        self.num_phases = num_phases
        self.output_dim = output_dim

        input_dim = strategic_dim + state_dim + num_phases

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
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
        z_game: torch.Tensor,
        state: torch.Tensor,
        current_phase: torch.Tensor,
        prev_phase: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute tactical plan, updating only at phase boundaries.

        Args:
            z_game: [batch, strategic_dim] - strategic context from GRU
            state: [batch, state_dim] - current game state embedding
            current_phase: [batch] - current phase index (0-13)
            prev_phase: [batch] - previous phase index

        Returns:
            z_turn: [batch, output_dim] - tactical plan vector
        """
        # Phase one-hot encoding
        batch_size = state.shape[0]
        device = state.device
        phase_onehot = torch.zeros(batch_size, self.num_phases, device=device)
        phase_idx = current_phase.long().clamp(0, self.num_phases - 1)
        phase_onehot.scatter_(1, phase_idx.unsqueeze(1), 1.0)

        # Concatenate inputs
        x = torch.cat([z_game, state, phase_onehot], dim=-1)

        # MLP forward
        z_turn = self.mlp(x)

        return z_turn
