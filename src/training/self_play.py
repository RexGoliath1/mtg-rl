"""
Self-Play Training for AlphaZero-style MTG Agent

This module implements the complete training loop:
1. ACTOR: Plays games using MCTS + network, generates training data
2. REPLAY BUFFER: Stores game trajectories
3. LEARNER: Trains network on collected data

The key insight of AlphaZero:
- MCTS provides "expert" move probabilities through search
- Network learns to predict MCTS output (policy) and game outcome (value)
- Better network → better MCTS → better training data → better network
- This bootstrapping loop improves without human data

Training Data Format:
- state: Game state (encoded as tensor)
- policy: MCTS visit count distribution (what MCTS thinks is best)
- value: Game outcome from this state (+1 win, -1 loss, 0 draw)

Usage:
    trainer = SelfPlayTrainer(config)
    trainer.train(num_iterations=100)
"""

import json
import os
import time
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import save_file as safetensors_save, load_file as safetensors_load

from src.forge.game_state_encoder import ForgeGameStateEncoder, GameStateConfig
from src.forge.policy_value_heads import (
    PolicyHead, ValueHead, PolicyValueConfig, ActionConfig,
    create_action_mask, decode_action
)
from src.forge.mcts import MCTS, MCTSConfig, SimulatedForgeClient, ForgeClientInterface


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SelfPlayConfig:
    """Configuration for self-play training."""

    # Game settings
    num_players: int = 2  # 1v1 for now, Commander = 4

    # Self-play settings
    num_actors: int = 1  # Parallel game threads (increase for distributed)
    games_per_iteration: int = 10  # Games to play before training
    max_game_length: int = 500  # Max moves per game

    # MCTS settings
    mcts_simulations: int = 100  # Simulations per move (800 for production)
    mcts_c_puct: float = 1.5
    temperature_moves: int = 30  # Moves to use temperature=1, then 0

    # Training settings
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs_per_iteration: int = 5
    replay_buffer_size: int = 100000
    min_buffer_size: int = 1000  # Min samples before training

    # Loss weights
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0

    # Checkpointing
    checkpoint_dir: str = "checkpoints/selfplay"
    checkpoint_interval: int = 10  # Iterations between checkpoints
    eval_interval: int = 20  # Iterations between evaluation

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# REPLAY BUFFER
# =============================================================================

@dataclass
class TrainingSample:
    """A single training sample from self-play."""
    state: np.ndarray  # Encoded game state
    policy: np.ndarray  # MCTS policy (visit count distribution)
    value: float  # Game outcome from player's perspective


class ReplayBuffer:
    """
    Fixed-size replay buffer for training samples.

    Stores game trajectories and samples random batches for training.
    """

    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.buffer: Deque[TrainingSample] = deque(maxlen=max_size)

    def add(self, sample: TrainingSample):
        """Add a sample to the buffer."""
        self.buffer.append(sample)

    def add_game(self, samples: List[TrainingSample]):
        """Add all samples from a game."""
        for sample in samples:
            self.add(sample)

    def sample(self, batch_size: int) -> List[TrainingSample]:
        """Sample a random batch."""
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


class ReplayDataset(Dataset):
    """PyTorch Dataset wrapper for replay buffer."""

    def __init__(self, samples: List[TrainingSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        return (
            torch.tensor(sample.state, dtype=torch.float32),
            torch.tensor(sample.policy, dtype=torch.float32),
            torch.tensor([sample.value], dtype=torch.float32),
        )


# =============================================================================
# ACTOR (Self-Play Game Player)
# =============================================================================

class SelfPlayActor:
    """
    Plays games using MCTS and collects training data.

    The actor:
    1. Runs games using current network + MCTS
    2. Records (state, MCTS_policy, outcome) for each move
    3. Returns completed games as training data
    """

    def __init__(
        self,
        network: nn.Module,
        encoder: ForgeGameStateEncoder,
        config: SelfPlayConfig,
    ):
        self.network = network
        self.encoder = encoder
        self.config = config
        self.action_config = ActionConfig()

        # Create MCTS
        mcts_config = MCTSConfig(
            num_simulations=config.mcts_simulations,
            c_puct=config.mcts_c_puct,
            temperature=1.0,
            temperature_drop_move=config.temperature_moves,
        )

        self.mcts = MCTS(
            policy_value_fn=self._policy_value_fn,
            encode_state_fn=self._encode_state,
            config=mcts_config,
        )

    def _policy_value_fn(self, state_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get policy and value from network."""
        self.network.eval()
        with torch.no_grad():
            # Ensure batch dimension
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)

            # Get policy logits and value
            policy_logits = self.network.policy_head(state_tensor, return_logits=True)
            value = self.network.value_head(state_tensor)

            # Softmax for policy
            policy = torch.softmax(policy_logits, dim=-1)

        return policy, value

    def _encode_state(self, game_state: Dict[str, Any]) -> torch.Tensor:
        """Encode game state to tensor."""
        return self.encoder.encode_json(game_state)

    def play_game(
        self,
        forge_client: Optional[ForgeClientInterface] = None
    ) -> List[TrainingSample]:
        """
        Play one complete game and return training samples.

        Returns:
            List of (state, policy, value) samples
        """
        # Use simulated client if none provided
        if forge_client is None:
            forge_client = SimulatedForgeClient()

        # Reset MCTS tree
        self.mcts.reset()

        # Game trajectory: (state_tensor, mcts_policy, player_to_move)
        trajectory = []

        move_count = 0
        while not forge_client.is_game_over() and move_count < self.config.max_game_length:
            # Get current state
            game_state = forge_client.get_game_state()
            current_player = game_state.get("priorityPlayer", 0)

            # Encode state
            state_tensor = self._encode_state(game_state)

            # Get legal actions
            legal_actions = forge_client.get_legal_actions()
            action_mask = create_action_mask(legal_actions, self.action_config)

            # Run MCTS
            temperature = 1.0 if move_count < self.config.temperature_moves else 0.0
            mcts_policy = self.mcts.search(
                forge_client,
                action_mask,
                num_simulations=self.config.mcts_simulations,
            )

            # Pad policy to action_dim
            padded_policy = np.zeros(self.action_config.total_actions, dtype=np.float32)
            padded_policy[:len(mcts_policy)] = mcts_policy

            # Store in trajectory
            trajectory.append((
                state_tensor.detach().cpu().numpy().flatten(),
                padded_policy,
                current_player,
            ))

            # Select action
            if temperature > 0:
                # Sample from policy
                action = np.random.choice(len(mcts_policy), p=mcts_policy)
            else:
                # Greedy
                action = np.argmax(mcts_policy)

            # Apply action
            action_dict = decode_action(action, self.action_config)
            forge_client.apply_action(action_dict)

            # Update MCTS tree
            self.mcts.update_with_action(action)

            move_count += 1

        # Game over - assign values
        samples = []
        winner_player = forge_client.get_game_state().get("winner")

        for state, policy, player in trajectory:
            # Value from this player's perspective
            if winner_player is None:
                value = 0.0  # Draw
            elif winner_player == player:
                value = 1.0  # Win
            else:
                value = -1.0  # Loss

            samples.append(TrainingSample(
                state=state,
                policy=policy,
                value=value,
            ))

        return samples


# =============================================================================
# LEARNER (Network Trainer)
# =============================================================================

class Learner:
    """
    Trains the network on self-play data.

    Loss = policy_loss + value_loss
    - policy_loss: Cross-entropy between network policy and MCTS policy
    - value_loss: MSE between network value and game outcome
    """

    def __init__(
        self,
        network: nn.Module,
        config: SelfPlayConfig,
    ):
        self.network = network
        self.config = config

        self.optimizer = optim.AdamW(
            network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()

    def train_on_batch(
        self,
        states: torch.Tensor,
        target_policies: torch.Tensor,
        target_values: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Train on a single batch.

        Returns:
            Dict with loss values
        """
        self.network.train()

        states = states.to(self.config.device)
        target_policies = target_policies.to(self.config.device)
        target_values = target_values.to(self.config.device)

        # Forward pass
        policy_logits = self.network.policy_head(states, return_logits=True)
        values = self.network.value_head(states)

        # Compute losses
        policy_loss = self.policy_loss_fn(policy_logits, target_policies)
        value_loss = self.value_loss_fn(values, target_values)

        total_loss = (
            self.config.policy_loss_weight * policy_loss +
            self.config.value_loss_weight * value_loss
        )

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item(),
        }

    def train_epoch(self, replay_buffer: ReplayBuffer) -> Dict[str, float]:
        """
        Train for one epoch on replay buffer.

        Returns:
            Average losses over epoch
        """
        if len(replay_buffer) < self.config.min_buffer_size:
            return {"skipped": True}

        # Sample from buffer
        samples = replay_buffer.sample(
            min(len(replay_buffer), self.config.batch_size * 10)
        )
        dataset = ReplayDataset(samples)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        epoch_losses = {"policy_loss": 0, "value_loss": 0, "total_loss": 0}
        num_batches = 0

        for states, policies, values in dataloader:
            losses = self.train_on_batch(states, policies, values)
            for k, v in losses.items():
                epoch_losses[k] += v
            num_batches += 1

        # Average
        for k in epoch_losses:
            epoch_losses[k] /= max(num_batches, 1)

        epoch_losses["num_batches"] = num_batches
        return epoch_losses


# =============================================================================
# COMBINED NETWORK (for training)
# =============================================================================

class AlphaZeroNetwork(nn.Module):
    """
    Combined network with shared encoder and separate heads.

    This is the full network used for training and inference.
    """

    def __init__(
        self,
        encoder_config: Optional[GameStateConfig] = None,
        head_config: Optional[PolicyValueConfig] = None,
        num_players: int = 2,
    ):
        super().__init__()

        self.encoder = ForgeGameStateEncoder(encoder_config)
        self.encoder_config = encoder_config or GameStateConfig()

        # Heads take encoder output
        head_config = head_config or PolicyValueConfig()
        head_config.state_dim = self.encoder_config.output_dim

        self.policy_head = PolicyHead(head_config)
        self.value_head = ValueHead(head_config, num_players)

    def forward(self, **encoder_inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward through encoder and heads."""
        state = self.encoder(**encoder_inputs)
        policy = self.policy_head(state, return_logits=True)
        value = self.value_head(state)
        return policy, value

    def save(self, path: str):
        """Save network in SafeTensors format with JSON metadata.

        Writes:
          - ``<stem>.safetensors`` -- model weights (safe, fast, portable)
          - ``<stem>.json``        -- encoder config and other metadata
          - ``<stem>.pt``          -- legacy torch checkpoint for backward compat
        """
        stem = path.replace(".pt", "")

        # SafeTensors weights
        safetensors_save(self.state_dict(), f"{stem}.safetensors")

        # JSON metadata (encoder config)
        meta = {
            "encoder_config": {
                k: getattr(self.encoder_config, k)
                for k in self.encoder_config.__dataclass_fields__
            },
        }
        with open(f"{stem}.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        # Legacy .pt for backward compatibility
        torch.save({
            "encoder_config": self.encoder_config,
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "AlphaZeroNetwork":
        """Load network, preferring SafeTensors when available.

        Looks for ``<stem>.safetensors`` + ``<stem>.json`` first.
        Falls back to legacy ``.pt`` checkpoint for old saves.
        """
        stem = path.replace(".pt", "")
        st_path = f"{stem}.safetensors"
        json_path = f"{stem}.json"

        if os.path.exists(st_path) and os.path.exists(json_path):
            # --- SafeTensors path ---
            with open(json_path) as f:
                meta = json.load(f)

            encoder_cfg_dict = meta.get("encoder_config", {})
            encoder_config = GameStateConfig(**encoder_cfg_dict) if encoder_cfg_dict else None
            network = cls(encoder_config=encoder_config)

            state_dict = safetensors_load(st_path, device=device or "cpu")
            network.load_state_dict(state_dict)
            return network

        # --- Legacy .pt fallback ---
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        network = cls(encoder_config=checkpoint["encoder_config"])
        network.load_state_dict(checkpoint["state_dict"])
        return network


# =============================================================================
# MAIN TRAINER
# =============================================================================

class SelfPlayTrainer:
    """
    Main training orchestrator.

    Coordinates actors and learner in the training loop.
    """

    def __init__(self, config: Optional[SelfPlayConfig] = None):
        self.config = config or SelfPlayConfig()

        # Create network
        self.network = AlphaZeroNetwork(num_players=self.config.num_players)
        self.network.to(self.config.device)

        # Create replay buffer
        self.replay_buffer = ReplayBuffer(self.config.replay_buffer_size)

        # Create actor and learner
        self.actor = SelfPlayActor(
            network=self.network,
            encoder=self.network.encoder,
            config=self.config,
        )
        self.learner = Learner(self.network, self.config)

        # Stats
        self.iteration = 0
        self.total_games = 0
        self.total_samples = 0

    def run_iteration(self) -> Dict[str, Any]:
        """
        Run one training iteration.

        1. Play games and collect data
        2. Train network on collected data
        3. Return stats
        """
        iteration_start = time.time()
        stats = {"iteration": self.iteration}

        # 1. Self-play phase
        print(f"\n[Iteration {self.iteration}] Self-play phase...")
        games_samples = []
        for game_idx in range(self.config.games_per_iteration):
            samples = self.actor.play_game()
            games_samples.extend(samples)
            self.replay_buffer.add_game(samples)
            print(f"  Game {game_idx + 1}/{self.config.games_per_iteration}: "
                  f"{len(samples)} moves")

        stats["games_played"] = self.config.games_per_iteration
        stats["samples_collected"] = len(games_samples)
        stats["buffer_size"] = len(self.replay_buffer)
        self.total_games += self.config.games_per_iteration
        self.total_samples += len(games_samples)

        # 2. Training phase
        print(f"[Iteration {self.iteration}] Training phase...")
        epoch_stats = []
        for epoch in range(self.config.epochs_per_iteration):
            losses = self.learner.train_epoch(self.replay_buffer)
            epoch_stats.append(losses)
            if "skipped" not in losses:
                print(f"  Epoch {epoch + 1}: policy_loss={losses['policy_loss']:.4f}, "
                      f"value_loss={losses['value_loss']:.4f}")

        if epoch_stats and "skipped" not in epoch_stats[-1]:
            stats["policy_loss"] = np.mean([e["policy_loss"] for e in epoch_stats])
            stats["value_loss"] = np.mean([e["value_loss"] for e in epoch_stats])

        # 3. Checkpointing
        if self.iteration % self.config.checkpoint_interval == 0:
            self._save_checkpoint()

        stats["iteration_time"] = time.time() - iteration_start
        self.iteration += 1

        return stats

    def train(self, num_iterations: int):
        """
        Run full training loop.

        Args:
            num_iterations: Number of iterations to run
        """
        print("=" * 70)
        print("AlphaZero Self-Play Training")
        print("=" * 70)
        print(f"Config: {self.config}")
        print(f"Device: {self.config.device}")
        print(f"Network parameters: {sum(p.numel() for p in self.network.parameters()):,}")
        print()

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        for i in range(num_iterations):
            stats = self.run_iteration()
            print(f"\n[Iteration {stats['iteration']}] Summary:")
            print(f"  Games: {stats['games_played']}, Samples: {stats['samples_collected']}")
            print(f"  Buffer size: {stats['buffer_size']}")
            if "policy_loss" in stats:
                print(f"  Policy loss: {stats['policy_loss']:.4f}, "
                      f"Value loss: {stats['value_loss']:.4f}")
            print(f"  Time: {stats['iteration_time']:.1f}s")

        print("\n" + "=" * 70)
        print(f"Training complete! Total games: {self.total_games}, "
              f"Total samples: {self.total_samples}")
        print("=" * 70)

        # Final checkpoint
        self._save_checkpoint()

    def _save_checkpoint(self):
        """Save training checkpoint."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_iter{self.iteration}.pt"
        )
        torch.save({
            "iteration": self.iteration,
            "total_games": self.total_games,
            "total_samples": self.total_samples,
            "network_state": self.network.state_dict(),
            "config": self.config,
        }, path)
        print(f"Saved checkpoint: {path}")


# =============================================================================
# TESTING
# =============================================================================

def test_self_play():
    """Test self-play training."""
    print("=" * 70)
    print("Testing Self-Play Training")
    print("=" * 70)

    # Small config for testing
    config = SelfPlayConfig(
        games_per_iteration=2,
        mcts_simulations=10,
        epochs_per_iteration=2,
        batch_size=32,
        min_buffer_size=10,
        checkpoint_dir="/tmp/selfplay_test",
    )

    trainer = SelfPlayTrainer(config)

    print(f"\nNetwork parameters: {sum(p.numel() for p in trainer.network.parameters()):,}")

    # Run a few iterations
    for i in range(2):
        stats = trainer.run_iteration()
        print(f"\nIteration {i}: {stats}")

    print("\n" + "=" * 70)
    print("Self-play test completed!")
    print("=" * 70)


if __name__ == "__main__":
    test_self_play()
