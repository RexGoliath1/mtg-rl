"""
Strategic Core GRU - Game-level trajectory encoding.

Maintains a 256-dim hidden state that summarizes the game trajectory,
updated once per turn (not per decision). This captures long-range
strategic context like "I'm ahead on board", "opponent is mana-screwed",
or "I'm in topdeck mode".

The GRU input combines the current game state embedding (768-dim from
ForgeGameStateEncoder) with the previous hidden state, enabling the
network to learn temporal patterns across turns.

Usage:
    core = StrategicCoreGRU()
    z_game = torch.zeros(batch, 256)  # reset at game start

    # Each turn:
    z_game = core(state_embedding, z_game, current_turn, prev_turn)
"""

import torch
import torch.nn as nn


class StrategicCoreGRU(nn.Module):
    """
    GRU cell that maintains game-level strategic context.

    Updated once per turn boundary, not per decision. This prevents
    the hidden state from being overwhelmed by per-decision noise
    (e.g., choosing targets, paying costs) and focuses on turn-level
    strategic shifts.

    Architecture:
        Input: game_state (768) -> projection (256) -> GRU(256, 256)
        Output: z_game (256) - strategic context vector
    """

    def __init__(
        self,
        state_dim: int = 768,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Project state down to hidden dim before GRU
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Single GRU cell (not a full RNN - we manually manage timesteps)
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

        # Post-GRU normalization for training stability
        self.output_norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GRUCell):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(
        self,
        state: torch.Tensor,
        prev_z_game: torch.Tensor,
        current_turn: torch.Tensor,
        prev_turn: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update strategic context, but only at turn boundaries.

        Args:
            state: [batch, state_dim] - current game state embedding
            prev_z_game: [batch, hidden_dim] - previous strategic context
            current_turn: [batch] - current turn number
            prev_turn: [batch] - previous turn number

        Returns:
            z_game: [batch, hidden_dim] - updated strategic context
        """
        # Project state to hidden dim
        projected = self.input_proj(state)

        # GRU update
        new_z = self.gru_cell(projected, prev_z_game)
        new_z = self.output_norm(new_z)

        # Only update at turn boundaries
        turn_changed = (current_turn != prev_turn).float().unsqueeze(-1)
        z_game = turn_changed * new_z + (1.0 - turn_changed) * prev_z_game

        return z_game

    def initial_state(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """Create zero initial hidden state for game start."""
        return torch.zeros(batch_size, self.hidden_dim, device=device)
