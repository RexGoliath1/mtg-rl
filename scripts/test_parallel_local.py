#!/usr/bin/env python3
"""
Local Parallel Self-Play Test

Simplified test that doesn't use threading to verify the core training works.
Uses a simple test network instead of the full AlphaZero architecture.
"""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

from src.training.profiler import TrainingProfiler, compare_configurations


class SimpleTestNetwork(nn.Module):
    """
    Simple network for testing training pipeline.
    Takes raw state vectors instead of structured game states.
    """

    def __init__(self, state_dim: int = 512, num_actions: int = 153, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor, action_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: [batch, state_dim]
            action_mask: [batch, num_actions] - 1 for valid, 0 for invalid

        Returns:
            policy: [batch, num_actions] - probability distribution
            value: [batch, 1] - win probability
        """
        # Encode state
        h = self.encoder(state)

        # Policy with masking
        logits = self.policy_head(h)
        logits = logits.masked_fill(action_mask == 0, float('-inf'))
        policy = torch.softmax(logits, dim=-1)

        # Value
        value = self.value_head(h)

        return policy, value


@dataclass
class SimpleConfig:
    """Simplified config for testing."""
    num_actors: int = 4
    games_per_iteration: int = 8
    mcts_simulations: int = 10
    batch_size: int = 32
    state_dim: int = 512
    num_actions: int = 153
    buffer_size: int = 10000
    min_buffer_size: int = 50
    device: str = "auto"

    def __post_init__(self):
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"


def simulate_game(network, config: SimpleConfig, device: torch.device, profiler: TrainingProfiler):
    """Simulate a single game without threading."""
    samples = []
    game_length = np.random.randint(30, 50)

    for move in range(game_length):
        # Generate random state
        state = torch.randn(config.state_dim)
        action_mask = torch.ones(config.num_actions)

        # Randomly mask some actions
        num_legal = np.random.randint(5, 30)
        invalid = np.random.choice(config.num_actions, config.num_actions - num_legal, replace=False)
        action_mask[invalid] = 0

        # Run simplified MCTS (just get network policy)
        with profiler.measure("forward_pass"):
            with torch.no_grad():
                state_batch = state.unsqueeze(0).to(device)
                mask_batch = action_mask.unsqueeze(0).to(device)
                policy, value = network(state_batch, mask_batch)
                policy = policy.squeeze(0).cpu()

        # Add Dirichlet noise
        noise = torch.from_numpy(np.random.dirichlet([0.3] * config.num_actions)).float()
        noisy_policy = 0.75 * policy + 0.25 * noise
        noisy_policy = noisy_policy * action_mask
        noisy_policy = noisy_policy / (noisy_policy.sum() + 1e-8)

        samples.append({
            "state": state,
            "policy": noisy_policy,
            "action_mask": action_mask,
        })

    # Assign outcomes
    winner = np.random.choice([0, 1])
    for i, sample in enumerate(samples):
        player = i % 2
        sample["value"] = torch.tensor([1.0 if player == winner else -1.0])

    profiler.increment("games", 1)
    profiler.increment("samples", len(samples))

    return samples


def train_step(network, optimizer, replay_buffer, config: SimpleConfig, device, profiler):
    """Single training step."""
    if len(replay_buffer) < config.min_buffer_size:
        return 0.0

    # Sample batch
    indices = np.random.choice(len(replay_buffer), min(config.batch_size, len(replay_buffer)), replace=False)

    states = torch.stack([replay_buffer[i]["state"] for i in indices]).to(device)
    policies = torch.stack([replay_buffer[i]["policy"] for i in indices]).to(device)
    values = torch.stack([replay_buffer[i]["value"] for i in indices]).to(device)
    masks = torch.stack([replay_buffer[i]["action_mask"] for i in indices]).to(device)

    network.train()

    with profiler.measure("forward_train"):
        pred_policy, pred_value = network(states, masks)

    with profiler.measure("loss"):
        policy_loss = -torch.sum(policies * torch.log(pred_policy + 1e-8), dim=-1).mean()
        value_loss = torch.nn.functional.mse_loss(pred_value, values)
        loss = policy_loss + value_loss

    with profiler.measure("backward"):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def main():
    print("=" * 70)
    print("PARALLEL SELF-PLAY TEST (Simplified)")
    print("=" * 70)

    # Show configuration comparison first
    print("\nTraining Time Estimates:")
    print(compare_configurations())
    print()

    # Config
    config = SimpleConfig(
        num_actors=4,
        games_per_iteration=8,
        mcts_simulations=10,
    )

    print(f"Config:")
    print(f"  Actors: {config.num_actors}")
    print(f"  Games/iteration: {config.games_per_iteration}")
    print(f"  MCTS sims: {config.mcts_simulations}")
    print(f"  Device: {config.device}")
    print()

    # Setup
    device = torch.device(config.device)
    profiler = TrainingProfiler()

    print("Creating network...")
    network = SimpleTestNetwork(
        state_dim=config.state_dim,
        num_actions=config.num_actions,
    ).to(device)
    print(f"Network: {sum(p.numel() for p in network.parameters()):,} params")

    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4)
    replay_buffer = deque(maxlen=config.buffer_size)

    # Warmup
    print("\nWarmup...")
    with torch.no_grad():
        dummy = torch.randn(1, config.state_dim).to(device)
        mask = torch.ones(1, config.num_actions).to(device)
        for _ in range(5):
            _ = network(dummy, mask)

    # Training
    print("\nTraining...")
    start_time = time.time()
    num_iterations = 3
    total_games = 0
    total_samples = 0

    for iteration in range(1, num_iterations + 1):
        iter_start = time.time()

        # Self-play
        with profiler.measure("selfplay_total"):
            for actor in range(config.num_actors):
                for _ in range(config.games_per_iteration // config.num_actors):
                    samples = simulate_game(network, config, device, profiler)
                    for s in samples:
                        replay_buffer.append(s)
                    total_games += 1
                    total_samples += len(samples)

        # Training
        with profiler.measure("training_total"):
            for _ in range(2):  # 2 epochs
                loss = train_step(network, optimizer, replay_buffer, config, device, profiler)

        iter_time = time.time() - iter_start
        print(f"[Iter {iteration}] Games: {config.games_per_iteration} | "
              f"Samples: {total_samples} | Buffer: {len(replay_buffer)} | "
              f"Loss: {loss:.4f} | Time: {iter_time:.1f}s")

    elapsed = time.time() - start_time

    # Results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total time: {elapsed:.1f}s")
    print(f"Total games: {total_games}")
    print(f"Total samples: {total_samples}")
    print(f"Games/second: {total_games/elapsed:.2f}")
    print(f"Samples/second: {total_samples/elapsed:.1f}")
    print()

    # Extrapolate to 1M samples
    samples_per_hour = total_samples / elapsed * 3600
    hours_to_1m = 1_000_000 / samples_per_hour
    print(f"Projected samples/hour: {samples_per_hour:,.0f}")
    print(f"Projected time to 1M samples: {hours_to_1m:.1f} hours ({hours_to_1m/24:.1f} days)")
    print()

    # Profiling report
    print(profiler.report())

    # Memory info
    if device.type == "cuda":
        print(f"\nGPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1e6:.1f} MB")
        print(f"  Max allocated: {torch.cuda.max_memory_allocated()/1e6:.1f} MB")
        print(f"  Reserved: {torch.cuda.memory_reserved()/1e6:.1f} MB")


if __name__ == "__main__":
    main()
