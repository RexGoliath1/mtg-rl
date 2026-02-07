"""
Parallel Self-Play Training for MTG RL

Scales training through:
1. Multiple actors playing games in parallel
2. Batched neural network inference across actors
3. Asynchronous game execution
4. Distributed training across multiple GPUs/machines

Architecture:
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │ Actor 1  │     │ Actor 2  │     │ Actor N  │
    │  (Game)  │     │  (Game)  │     │  (Game)  │
    └────┬─────┘     └────┬─────┘     └────┬─────┘
         │                │                │
         └───────────┬────┴────────────────┘
                     │
              ┌──────▼──────┐
              │   Batcher   │  ← Collects states from all actors
              │  (GPU)      │  ← Single batched forward pass
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │   Learner   │  ← Trains on replay buffer
              │  (GPU)      │
              └─────────────┘

Usage:
    trainer = ParallelSelfPlayTrainer(config)
    trainer.train(num_iterations=100)
"""

import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

import torch
import torch.nn as nn

from src.training.profiler import TrainingProfiler


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ParallelConfig:
    """Configuration for parallel self-play training."""

    # Parallelism
    num_actors: int = 8
    num_inference_threads: int = 2
    batch_timeout_ms: float = 10.0  # Max wait to fill batch

    # MCTS
    mcts_simulations: int = 100
    mcts_cpuct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    # Training
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs_per_iteration: int = 4
    games_per_iteration: int = 100

    # Replay buffer
    buffer_size: int = 500_000
    min_buffer_size: int = 10_000

    # Network
    state_dim: int = 512
    num_actions: int = 203

    # Checkpointing
    checkpoint_dir: str = "checkpoints/parallel"
    checkpoint_interval: int = 10

    # Hardware
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"
    mixed_precision: bool = True
    pin_memory: bool = True

    # Profiling
    profile_enabled: bool = True

    def __post_init__(self):
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"


# =============================================================================
# BATCHED INFERENCE SERVER
# =============================================================================

class InferenceRequest:
    """A request for neural network inference."""

    def __init__(self, state: torch.Tensor, action_mask: torch.Tensor):
        self.state = state
        self.action_mask = action_mask
        self.event = threading.Event()
        self.policy: Optional[torch.Tensor] = None
        self.value: Optional[torch.Tensor] = None


class BatchedInferenceServer:
    """
    Batches inference requests from multiple actors for efficient GPU usage.

    Instead of each actor doing individual forward passes, requests are
    collected and processed together, significantly improving throughput.
    """

    def __init__(
        self,
        network: nn.Module,
        config: ParallelConfig,
        profiler: Optional[TrainingProfiler] = None,
    ):
        self.network = network
        self.config = config
        self.profiler = profiler or TrainingProfiler(enabled=False)
        self.device = torch.device(config.device)

        # Request queue
        self.request_queue: queue.Queue = queue.Queue()
        self.pending_requests: List[InferenceRequest] = []

        # Control
        self.running = False
        self.inference_threads: List[threading.Thread] = []

        # Stats
        self.total_inferences = 0
        self.total_batches = 0
        self.batch_sizes: List[int] = []

    def start(self):
        """Start inference server threads."""
        self.running = True
        for i in range(self.config.num_inference_threads):
            t = threading.Thread(target=self._inference_loop, name=f"Inference-{i}")
            t.daemon = True
            t.start()
            self.inference_threads.append(t)

    def stop(self):
        """Stop inference server."""
        self.running = False
        for t in self.inference_threads:
            t.join(timeout=1.0)
        self.inference_threads.clear()

    def request_inference(
        self,
        state: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Submit an inference request and wait for result.

        Args:
            state: State tensor [state_dim]
            action_mask: Valid actions mask [num_actions]

        Returns:
            (policy, value) tensors
        """
        request = InferenceRequest(state, action_mask)
        self.request_queue.put(request)

        # Wait for result
        request.event.wait()

        return request.policy, request.value

    def _inference_loop(self):
        """Main inference loop - collects and processes batches."""
        while self.running:
            try:
                batch = self._collect_batch()
                if batch:
                    self._process_batch(batch)
            except Exception as e:
                print(f"Inference error: {e}")

    def _collect_batch(self) -> List[InferenceRequest]:
        """Collect requests into a batch."""
        batch = []
        timeout = self.config.batch_timeout_ms / 1000.0
        deadline = time.time() + timeout

        # Get at least one request
        try:
            first = self.request_queue.get(timeout=timeout)
            batch.append(first)
        except queue.Empty:
            return []

        # Collect more requests until batch is full or timeout
        max_batch = self.config.batch_size
        while len(batch) < max_batch and time.time() < deadline:
            try:
                req = self.request_queue.get_nowait()
                batch.append(req)
            except queue.Empty:
                break

        return batch

    def _process_batch(self, batch: List[InferenceRequest]):
        """Process a batch of inference requests."""
        if not batch:
            return

        batch_size = len(batch)
        self.batch_sizes.append(batch_size)
        self.total_batches += 1

        with self.profiler.measure("batch_assembly"):
            # Stack inputs
            states = torch.stack([r.state for r in batch]).to(self.device)
            masks = torch.stack([r.action_mask for r in batch]).to(self.device)

        with self.profiler.measure("forward_pass"):
            with torch.no_grad():
                if self.config.mixed_precision and self.device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        policies, values = self.network(states, masks)
                else:
                    policies, values = self.network(states, masks)

        with self.profiler.measure("result_distribution"):
            # Move to CPU and distribute results
            policies = policies.cpu()
            values = values.cpu()

            for i, request in enumerate(batch):
                request.policy = policies[i]
                request.value = values[i]
                request.event.set()

        self.total_inferences += batch_size
        self.profiler.increment("inferences", batch_size)

    def get_stats(self) -> Dict[str, Any]:
        """Get inference server statistics."""
        avg_batch = np.mean(self.batch_sizes) if self.batch_sizes else 0
        return {
            "total_inferences": self.total_inferences,
            "total_batches": self.total_batches,
            "avg_batch_size": avg_batch,
            "batch_efficiency": avg_batch / self.config.batch_size if self.config.batch_size > 0 else 0,
        }


# =============================================================================
# PARALLEL ACTOR
# =============================================================================

class ParallelActor:
    """
    Actor that plays games using MCTS with batched inference.

    Each actor runs in its own thread, playing games and submitting
    inference requests to the shared BatchedInferenceServer.
    """

    def __init__(
        self,
        actor_id: int,
        inference_server: BatchedInferenceServer,
        config: ParallelConfig,
        profiler: Optional[TrainingProfiler] = None,
    ):
        self.actor_id = actor_id
        self.inference_server = inference_server
        self.config = config
        self.profiler = profiler or TrainingProfiler(enabled=False)

        # Game state (simulated for now)
        self.current_game = None

        # Stats
        self.games_played = 0
        self.samples_collected = 0

    def play_game(self) -> List[Dict]:
        """
        Play a complete game using MCTS.

        Returns:
            List of training samples [(state, policy, value), ...]
        """
        samples = []

        # Initialize game (simulated)
        game_length = np.random.randint(30, 50)  # Typical MTG game length

        for move in range(game_length):
            with self.profiler.measure("mcts_search"):
                # Get state and mask (simulated)
                state = torch.randn(self.config.state_dim)
                action_mask = torch.ones(self.config.num_actions)
                # Randomly mask some actions
                num_legal = np.random.randint(5, 30)
                invalid_actions = np.random.choice(
                    self.config.num_actions,
                    size=self.config.num_actions - num_legal,
                    replace=False
                )
                action_mask[invalid_actions] = 0

                # Run MCTS
                mcts_policy = self._run_mcts(state, action_mask)

                # Record sample (will be updated with outcome at game end)
                samples.append({
                    "state": state.clone(),
                    "policy": mcts_policy.clone(),
                    "action_mask": action_mask.clone(),
                    "move_num": move,
                })

                # Select and play action
                _ = torch.multinomial(mcts_policy, 1).item()

            with self.profiler.measure("forge_step"):
                # Simulate Forge communication
                time.sleep(0.01)  # ~10ms simulated latency

        # Determine winner (simulated)
        winner = np.random.choice([0, 1])

        # Add outcomes to samples
        for i, sample in enumerate(samples):
            # Perspective of player who moved
            player = i % 2
            sample["value"] = torch.tensor([1.0 if player == winner else -1.0])

        self.games_played += 1
        self.samples_collected += len(samples)
        self.profiler.increment("games_played")
        self.profiler.increment("samples_collected", len(samples))

        return samples

    def _run_mcts(
        self,
        root_state: torch.Tensor,
        root_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run MCTS search using batched inference.

        Simplified MCTS that uses the inference server for evaluations.
        """
        # Get initial policy and value from network
        policy, value = self.inference_server.request_inference(root_state, root_mask)

        # Apply Dirichlet noise for exploration
        noise = torch.from_numpy(
            np.random.dirichlet([self.config.dirichlet_alpha] * self.config.num_actions)
        ).float()

        noisy_policy = (
            (1 - self.config.dirichlet_epsilon) * policy +
            self.config.dirichlet_epsilon * noise
        )

        # Apply mask
        noisy_policy = noisy_policy * root_mask
        noisy_policy = noisy_policy / (noisy_policy.sum() + 1e-8)

        # Simplified: just return the policy (full MCTS would do tree search)
        # In production, this would build a search tree with multiple simulations
        visit_counts = torch.zeros(self.config.num_actions)
        for _ in range(self.config.mcts_simulations):
            # Select action using UCB
            action = self._select_action(noisy_policy, visit_counts, root_mask)

            # Simulate (in real MCTS, would expand tree)
            # For now, just get network evaluation
            _, leaf_value = self.inference_server.request_inference(root_state, root_mask)

            # Update visit counts
            visit_counts[action] += 1

        # Convert visits to policy
        mcts_policy = visit_counts / (visit_counts.sum() + 1e-8)
        return mcts_policy

    def _select_action(
        self,
        prior: torch.Tensor,
        visits: torch.Tensor,
        mask: torch.Tensor,
    ) -> int:
        """Select action using PUCT formula."""
        total_visits = visits.sum() + 1

        # UCB score
        exploration = self.config.mcts_cpuct * prior * np.sqrt(total_visits) / (1 + visits)
        exploitation = visits / (total_visits + 1e-8)

        ucb = exploitation + exploration

        # Mask invalid actions
        ucb = ucb * mask
        ucb[mask == 0] = float('-inf')

        return ucb.argmax().item()


# =============================================================================
# PARALLEL TRAINER
# =============================================================================

class ParallelSelfPlayTrainer:
    """
    Orchestrates parallel self-play training.

    Manages:
    - Multiple actor threads
    - Batched inference server
    - Replay buffer
    - Training loop
    """

    def __init__(self, config: Optional[ParallelConfig] = None):
        self.config = config or ParallelConfig()
        self.device = torch.device(self.config.device)

        # Profiler
        self.profiler = TrainingProfiler(enabled=self.config.profile_enabled)

        # Network (import here to avoid circular imports)
        from src.training.self_play import AlphaZeroNetwork
        from src.forge.game_state_encoder import GameStateConfig
        from src.forge.policy_value_heads import PolicyValueConfig, ActionConfig

        # Configure network dimensions
        encoder_config = GameStateConfig(output_dim=self.config.state_dim)
        action_config = ActionConfig()  # Default 203 actions
        head_config = PolicyValueConfig(
            state_dim=self.config.state_dim,
            action_config=action_config,
        )
        # Update num_actions to match actual action space
        self.config.num_actions = action_config.total_actions

        self.network = AlphaZeroNetwork(
            encoder_config=encoder_config,
            head_config=head_config,
        ).to(self.device)

        # Inference server
        self.inference_server = BatchedInferenceServer(
            self.network,
            self.config,
            self.profiler,
        )

        # Actors
        self.actors: List[ParallelActor] = []
        for i in range(self.config.num_actors):
            actor = ParallelActor(i, self.inference_server, self.config, self.profiler)
            self.actors.append(actor)

        # Replay buffer
        self.replay_buffer: deque = deque(maxlen=self.config.buffer_size)

        # Training
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if (
            self.config.mixed_precision and self.device.type == "cuda"
        ) else None

        # Stats
        self.iteration = 0
        self.total_games = 0
        self.total_samples = 0

    def train(self, num_iterations: int):
        """
        Run parallel self-play training.

        Args:
            num_iterations: Number of training iterations
        """
        print("=" * 70)
        print("PARALLEL SELF-PLAY TRAINING")
        print("=" * 70)
        print(f"Actors: {self.config.num_actors}")
        print(f"MCTS simulations: {self.config.mcts_simulations}")
        print(f"Games per iteration: {self.config.games_per_iteration}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.config.mixed_precision}")
        print()

        # Start inference server
        self.inference_server.start()

        try:
            for i in range(num_iterations):
                self.iteration = i + 1
                self._run_iteration()

                if i > 0 and i % self.config.checkpoint_interval == 0:
                    self._save_checkpoint()

        finally:
            self.inference_server.stop()

        # Final checkpoint
        self._save_checkpoint()

        # Print profiling report
        print()
        print(self.profiler.report())

    def _run_iteration(self):
        """Run a single training iteration."""
        iter_start = time.time()

        # === Self-Play Phase ===
        print(f"\n[Iter {self.iteration}] Self-play phase...")

        # Run actors in parallel (using threads for now)
        with self.profiler.measure("selfplay_phase"):
            games_to_play = self.config.games_per_iteration // self.config.num_actors
            all_samples = []

            # Simple parallel execution
            threads = []
            results = [[] for _ in self.actors]

            def actor_work(actor_idx, num_games):
                for _ in range(num_games):
                    samples = self.actors[actor_idx].play_game()
                    results[actor_idx].extend(samples)

            for i, actor in enumerate(self.actors):
                t = threading.Thread(target=actor_work, args=(i, games_to_play))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            for r in results:
                all_samples.extend(r)

            # Add to replay buffer
            for sample in all_samples:
                self.replay_buffer.append(sample)

            self.total_games += games_to_play * self.config.num_actors
            self.total_samples += len(all_samples)

        games_played = games_to_play * self.config.num_actors
        samples_collected = len(all_samples)

        # === Training Phase ===
        if len(self.replay_buffer) >= self.config.min_buffer_size:
            print(f"[Iter {self.iteration}] Training phase...")

            with self.profiler.measure("training_phase"):
                total_loss = 0.0
                num_batches = 0

                for epoch in range(self.config.epochs_per_iteration):
                    epoch_loss = self._train_epoch()
                    total_loss += epoch_loss
                    num_batches += 1

                avg_loss = total_loss / num_batches if num_batches > 0 else 0

        else:
            avg_loss = 0.0
            print(f"[Iter {self.iteration}] Filling buffer: {len(self.replay_buffer)}/{self.config.min_buffer_size}")

        # === Report ===
        iter_time = time.time() - iter_start
        inference_stats = self.inference_server.get_stats()

        print(
            f"[Iter {self.iteration}] "
            f"Games: {games_played} | "
            f"Samples: {samples_collected} | "
            f"Buffer: {len(self.replay_buffer)} | "
            f"Loss: {avg_loss:.4f} | "
            f"Batch eff: {inference_stats['batch_efficiency']:.1%} | "
            f"Time: {iter_time:.1f}s"
        )

    def _train_epoch(self) -> float:
        """Train for one epoch on replay buffer."""
        self.network.train()

        # Sample batch
        indices = np.random.choice(
            len(self.replay_buffer),
            size=min(self.config.batch_size, len(self.replay_buffer)),
            replace=False
        )

        batch_states = []
        batch_policies = []
        batch_values = []
        batch_masks = []

        for idx in indices:
            sample = self.replay_buffer[idx]
            batch_states.append(sample["state"])
            batch_policies.append(sample["policy"])
            batch_values.append(sample["value"])
            batch_masks.append(sample["action_mask"])

        states = torch.stack(batch_states).to(self.device)
        target_policies = torch.stack(batch_policies).to(self.device)
        target_values = torch.stack(batch_values).to(self.device)
        masks = torch.stack(batch_masks).to(self.device)

        # Forward pass
        with self.profiler.measure("forward_train"):
            if self.scaler:
                with torch.cuda.amp.autocast():
                    pred_policies, pred_values = self.network(states, masks)
                    policy_loss = self._policy_loss(pred_policies, target_policies)
                    value_loss = self._value_loss(pred_values, target_values)
                    loss = policy_loss + value_loss
            else:
                pred_policies, pred_values = self.network(states, masks)
                policy_loss = self._policy_loss(pred_policies, target_policies)
                value_loss = self._value_loss(pred_values, target_values)
                loss = policy_loss + value_loss

        # Backward pass
        with self.profiler.measure("backward"):
            self.optimizer.zero_grad()
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

        return loss.item()

    def _policy_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss for policy."""
        return -torch.sum(target * torch.log(pred + 1e-8), dim=-1).mean()

    def _value_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """MSE loss for value."""
        return torch.nn.functional.mse_loss(pred, target)

    def _save_checkpoint(self):
        """Save training checkpoint."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        path = os.path.join(
            self.config.checkpoint_dir,
            f"parallel_iter{self.iteration}.pt"
        )

        torch.save({
            "iteration": self.iteration,
            "network_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "total_games": self.total_games,
            "total_samples": self.total_samples,
            "config": self.config,
        }, path)

        print(f"[Checkpoint] Saved to {path}")


# =============================================================================
# BENCHMARKING
# =============================================================================

def benchmark_configurations(num_iterations: int = 5):
    """Benchmark different parallel configurations."""
    from src.training.profiler import compare_configurations

    print(compare_configurations())
    print()

    # Test actual performance with different actor counts
    configs_to_test = [
        ("1 actor", ParallelConfig(num_actors=1, games_per_iteration=10, mcts_simulations=20)),
        ("4 actors", ParallelConfig(num_actors=4, games_per_iteration=20, mcts_simulations=20)),
        ("8 actors", ParallelConfig(num_actors=8, games_per_iteration=40, mcts_simulations=20)),
    ]

    print("=" * 70)
    print("ACTUAL BENCHMARK RESULTS")
    print("=" * 70)

    for name, config in configs_to_test:
        print(f"\nTesting: {name}")
        config.profile_enabled = True
        config.checkpoint_dir = f"/tmp/benchmark_{name.replace(' ', '_')}"

        trainer = ParallelSelfPlayTrainer(config)

        start = time.time()
        trainer.train(num_iterations=num_iterations)
        elapsed = time.time() - start

        games = trainer.total_games
        samples = trainer.total_samples
        inference_stats = trainer.inference_server.get_stats()

        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"  Games: {games} ({games/elapsed:.1f}/s)")
        print(f"  Samples: {samples} ({samples/elapsed:.1f}/s)")
        print(f"  Batch efficiency: {inference_stats['batch_efficiency']:.1%}")
        print(f"  Avg batch size: {inference_stats['avg_batch_size']:.1f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parallel self-play training")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--actors", type=int, default=4, help="Number of actors")
    parser.add_argument("--iterations", type=int, default=10, help="Training iterations")
    parser.add_argument("--games", type=int, default=20, help="Games per iteration")
    parser.add_argument("--mcts-sims", type=int, default=50, help="MCTS simulations")
    args = parser.parse_args()

    if args.benchmark:
        benchmark_configurations()
    else:
        config = ParallelConfig(
            num_actors=args.actors,
            games_per_iteration=args.games,
            mcts_simulations=args.mcts_sims,
        )

        trainer = ParallelSelfPlayTrainer(config)
        trainer.train(num_iterations=args.iterations)
