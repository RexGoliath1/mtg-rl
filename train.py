#!/usr/bin/env python3
"""
DEPRECATED: Use training_pipeline.py for RL training instead.

This script has been superseded by training_pipeline.py which provides:
- Unified BC + RL training
- Better integration with the current architecture
- More flexible configuration

Migration:
    # Old
    python train.py --mode daemon --episodes 10000

    # New
    python training_pipeline.py --mode rl --episodes 10000

---

MTG RL Training Script (LEGACY)

Trains a PPO agent against the Forge daemon with:
- Curriculum learning (progressive deck difficulty)
- S3 checkpointing with spot instance handling
- TensorBoard and WandB logging
- Parallel environment support

Usage:
    # Local training (daemon must be running)
    python train.py --mode daemon --episodes 10000

    # Test with mock data (no daemon needed)
    python train.py --mode test

    # Resume from checkpoint
    python train.py --mode daemon --resume
"""

import warnings
warnings.warn(
    "train.py is deprecated. Use training_pipeline.py instead.",
    DeprecationWarning,
    stacklevel=2
)

import os
import time
import json
import random
import argparse
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Tuple, Dict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Local imports
from daemon_environment import DaemonMTGEnvironment, check_daemon_status
from checkpoint_manager import CheckpointManager, TrainingState


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Environment settings
    daemon_host: str = "localhost"
    daemon_port: int = 17171
    n_envs: int = 1  # Number of parallel environments

    # Training settings
    total_episodes: int = 100_000
    max_decisions_per_game: int = 1000

    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    n_epochs: int = 4
    batch_size: int = 64
    n_steps: int = 128  # Steps before PPO update
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Network settings
    hidden_dim: int = 256
    n_layers: int = 3

    # Curriculum learning
    curriculum_enabled: bool = True
    curriculum_stages: List[Dict] = field(default_factory=lambda: [
        # Stage 1: Mirror matches - learn basic play patterns
        {
            "name": "mirror_matches",
            "episodes": 5000,
            "matchups": [
                ("decks/competitive/mono_red_aggro.dck", "decks/competitive/mono_red_aggro.dck"),
                ("decks/competitive/boros_aggro.dck", "decks/competitive/boros_aggro.dck"),
            ]
        },
        # Stage 2: Similar archetypes - aggro vs aggro
        {
            "name": "aggro_matchups",
            "episodes": 10000,
            "matchups": [
                ("decks/competitive/mono_red_aggro.dck", "decks/competitive/boros_aggro.dck"),
                ("decks/competitive/boros_aggro.dck", "decks/competitive/mono_red_aggro.dck"),
            ]
        },
        # Stage 3: Different archetypes - learn adaptation
        {
            "name": "mixed_matchups",
            "episodes": 30000,
            "matchups": [
                ("decks/competitive/mono_red_aggro.dck", "decks/competitive/dimir_midrange.dck"),
                ("decks/competitive/mono_red_aggro.dck", "decks/competitive/jeskai_control.dck"),
                ("decks/competitive/boros_aggro.dck", "decks/competitive/dimir_midrange.dck"),
                ("decks/competitive/boros_aggro.dck", "decks/competitive/jeskai_control.dck"),
            ]
        },
        # Stage 4: Full meta - all matchups
        {
            "name": "full_meta",
            "episodes": -1,  # Run indefinitely
            "matchups": [
                ("decks/competitive/mono_red_aggro.dck", "decks/competitive/mono_red_aggro.dck"),
                ("decks/competitive/mono_red_aggro.dck", "decks/competitive/boros_aggro.dck"),
                ("decks/competitive/mono_red_aggro.dck", "decks/competitive/dimir_midrange.dck"),
                ("decks/competitive/mono_red_aggro.dck", "decks/competitive/jeskai_control.dck"),
                ("decks/competitive/boros_aggro.dck", "decks/competitive/boros_aggro.dck"),
                ("decks/competitive/boros_aggro.dck", "decks/competitive/dimir_midrange.dck"),
                ("decks/competitive/boros_aggro.dck", "decks/competitive/jeskai_control.dck"),
                ("decks/competitive/dimir_midrange.dck", "decks/competitive/dimir_midrange.dck"),
                ("decks/competitive/dimir_midrange.dck", "decks/competitive/jeskai_control.dck"),
                ("decks/competitive/jeskai_control.dck", "decks/competitive/jeskai_control.dck"),
            ]
        },
    ])

    # Logging and saving
    log_interval: int = 100  # Log every N episodes
    save_interval_episodes: int = 1000
    save_interval_seconds: int = 1800  # 30 minutes
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # S3 settings
    s3_bucket: Optional[str] = None
    s3_prefix: str = "mtg-rl/checkpoints"

    # WandB settings
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d


class SimplePolicyNetwork(nn.Module):
    """
    Simple MLP policy network for MTG.

    Takes game state observation and outputs:
    - Action logits (for each possible action)
    - State value estimate
    """

    def __init__(
        self,
        obs_dim: int = 38,
        max_actions: int = 50,
        hidden_dim: int = 256,
        n_layers: int = 3,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.max_actions = max_actions

        # Shared feature extractor
        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim

        self.features = nn.Sequential(*layers)

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_actions),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: [batch, obs_dim] observation tensor
            action_mask: [batch, max_actions] mask (1=valid, 0=invalid)

        Returns:
            action_logits: [batch, max_actions]
            value: [batch, 1]
        """
        features = self.features(obs)
        action_logits = self.policy_head(features)
        value = self.value_head(features)

        # Apply action mask
        if action_mask is not None:
            # Set logits of invalid actions to very negative value
            action_logits = action_logits.masked_fill(action_mask == 0, -1e9)

        return action_logits, value


class RolloutBuffer:
    """Stores rollout data for PPO updates."""

    def __init__(self, n_steps: int, obs_dim: int, device: torch.device):
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.device = device
        self.reset()

    def reset(self):
        self.observations = []
        self.actions = []
        self.action_masks = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.pos = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        action_mask: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ):
        self.observations.append(obs)
        self.actions.append(action)
        self.action_masks.append(action_mask)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.pos += 1

    def compute_returns(self, last_value: float, gamma: float, gae_lambda: float):
        """Compute GAE returns and advantages."""
        n = len(self.rewards)
        self.advantages = np.zeros(n, dtype=np.float32)
        self.returns = np.zeros(n, dtype=np.float32)

        gae = 0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_done = 0
            else:
                next_value = self.values[t + 1]
                next_done = self.dones[t + 1]

            delta = self.rewards[t] + gamma * next_value * (1 - next_done) - self.values[t]
            gae = delta + gamma * gae_lambda * (1 - next_done) * gae
            self.advantages[t] = gae

        self.returns = self.advantages + np.array(self.values)

    def get_batches(self, batch_size: int):
        """Yield batches for training."""
        n = len(self.observations)
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]

            yield {
                'obs': torch.tensor(np.array([self.observations[i] for i in batch_indices]), dtype=torch.float32, device=self.device),
                'actions': torch.tensor([self.actions[i] for i in batch_indices], dtype=torch.long, device=self.device),
                'action_masks': torch.tensor(np.array([self.action_masks[i] for i in batch_indices]), dtype=torch.float32, device=self.device),
                'old_log_probs': torch.tensor([self.log_probs[i] for i in batch_indices], dtype=torch.float32, device=self.device),
                'advantages': torch.tensor([self.advantages[i] for i in batch_indices], dtype=torch.float32, device=self.device),
                'returns': torch.tensor([self.returns[i] for i in batch_indices], dtype=torch.float32, device=self.device),
            }


class PPOTrainer:
    """PPO training algorithm."""

    def __init__(
        self,
        config: TrainingConfig,
        device: torch.device,
    ):
        self.config = config
        self.device = device

        # Create network
        self.policy = SimplePolicyNetwork(
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
        ).to(device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate,
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(config.n_steps, 38, device)

        # Stats
        self.total_steps = 0
        self.num_updates = 0

    def get_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
    ) -> Tuple[int, float, float]:
        """
        Get action from policy.

        Returns:
            action, log_prob, value
        """
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_t = torch.tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

            logits, value = self.policy(obs_t, mask_t)
            probs = F.softmax(logits, dim=-1)

            # Sample action
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            return action.item(), log_prob.item(), value.squeeze().item()

    def train_step(self, last_value: float) -> Dict[str, float]:
        """Perform PPO update on collected rollout."""
        # Compute returns
        self.buffer.compute_returns(last_value, self.config.gamma, self.config.gae_lambda)

        # Normalize advantages
        advantages = np.array(self.buffer.advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages = advantages.tolist()

        # PPO epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_batches = 0

        for _ in range(self.config.n_epochs):
            for batch in self.buffer.get_batches(self.config.batch_size):
                # Forward pass
                logits, values = self.policy(batch['obs'], batch['action_masks'])
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)

                # New log probs
                new_log_probs = dist.log_prob(batch['actions'])
                entropy = dist.entropy().mean()

                # Ratio
                ratio = torch.exp(new_log_probs - batch['old_log_probs'])

                # Clipped surrogate objective
                advantages = batch['advantages']
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch['returns'])

                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_batches += 1

        self.num_updates += 1
        self.buffer.reset()

        return {
            'policy_loss': total_policy_loss / max(n_batches, 1),
            'value_loss': total_value_loss / max(n_batches, 1),
            'entropy': total_entropy / max(n_batches, 1),
        }

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_updates': self.num_updates,
            'total_steps': self.total_steps,
        }, path)

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.num_updates = checkpoint.get('num_updates', 0)
        self.total_steps = checkpoint.get('total_steps', 0)


class CurriculumManager:
    """Manages curriculum learning stages."""

    def __init__(self, stages: List[Dict]):
        self.stages = stages
        self.current_stage = 0
        self.episodes_in_stage = 0

    def get_matchup(self) -> Tuple[str, str]:
        """Get a random matchup from current stage."""
        stage = self.stages[self.current_stage]
        matchups = stage['matchups']
        return random.choice(matchups)

    def advance_episode(self) -> bool:
        """
        Advance episode counter.
        Returns True if stage changed.
        """
        self.episodes_in_stage += 1
        stage = self.stages[self.current_stage]

        # Check if we should advance to next stage
        if stage['episodes'] > 0 and self.episodes_in_stage >= stage['episodes']:
            if self.current_stage < len(self.stages) - 1:
                self.current_stage += 1
                self.episodes_in_stage = 0
                return True

        return False

    def get_stage_name(self) -> str:
        return self.stages[self.current_stage]['name']

    def to_dict(self) -> Dict:
        return {
            'current_stage': self.current_stage,
            'episodes_in_stage': self.episodes_in_stage,
        }

    def from_dict(self, data: Dict):
        self.current_stage = data.get('current_stage', 0)
        self.episodes_in_stage = data.get('episodes_in_stage', 0)


def train_daemon(config: TrainingConfig, resume: bool = False):
    """
    Main training loop using daemon environment.

    Args:
        config: Training configuration
        resume: Whether to resume from latest checkpoint
    """
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config.log_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Save config
    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create trainer
    trainer = PPOTrainer(config, device)
    print(f"Policy network parameters: {sum(p.numel() for p in trainer.policy.parameters()):,}")

    # Create checkpoint manager
    checkpoint_mgr = CheckpointManager(
        local_dir=config.checkpoint_dir,
        s3_bucket=config.s3_bucket,
        s3_prefix=config.s3_prefix,
        save_interval_episodes=config.save_interval_episodes,
        save_interval_seconds=config.save_interval_seconds,
    )

    # Create curriculum manager
    curriculum = CurriculumManager(config.curriculum_stages)

    # Setup TensorBoard
    writer = SummaryWriter(run_dir)

    # Setup WandB if configured
    if config.wandb_project:
        try:
            import wandb
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name or f"mtg-rl-{timestamp}",
                config=config.to_dict(),
            )
        except ImportError:
            print("WandB not installed, skipping")
            config.wandb_project = None

    # Training state
    episode = 0
    total_games = 0
    wins = 0
    losses = 0
    episode_rewards = []
    win_rates = []
    best_win_rate = 0.0

    # Try to resume from checkpoint
    if resume:
        training_state = checkpoint_mgr.load_latest(trainer.policy, trainer.optimizer)
        if training_state:
            episode = training_state.episode
            total_games = training_state.total_games
            wins = training_state.wins
            losses = training_state.losses
            episode_rewards = training_state.episode_rewards
            win_rates = training_state.win_rates
            best_win_rate = training_state.best_win_rate

            # Restore curriculum state if saved
            if 'curriculum' in training_state.training_config:
                curriculum.from_dict(training_state.training_config['curriculum'])

            print(f"Resumed from episode {episode}")

    # Check daemon status
    status = check_daemon_status(config.daemon_host, config.daemon_port)
    if status['status'] != 'running':
        print(f"ERROR: Daemon not running at {config.daemon_host}:{config.daemon_port}")
        print("Start the daemon first: java -jar forge.jar daemon -p 17171")
        print("Or use docker-compose: docker-compose up daemon")
        return

    print(f"\nDaemon connected at {config.daemon_host}:{config.daemon_port}")
    print(f"Starting training for {config.total_episodes:,} episodes")
    print(f"Curriculum stage: {curriculum.get_stage_name()}")
    print("=" * 60)

    # Training loop
    start_time = time.time()
    steps_since_update = 0

    try:
        while episode < config.total_episodes:
            # Get matchup from curriculum
            deck1, deck2 = curriculum.get_matchup()

            # Create environment for this game
            env = DaemonMTGEnvironment(
                host=config.daemon_host,
                port=config.daemon_port,
                deck1=deck1,
                deck2=deck2,
            )

            try:
                obs, info = env.reset()
            except Exception as e:
                print(f"Failed to reset environment: {e}")
                time.sleep(1)
                continue

            episode_reward = 0.0
            episode_steps = 0
            done = False

            # Play episode
            while not done and episode_steps < config.max_decisions_per_game:
                action_mask = env.get_action_mask()
                action, log_prob, value = trainer.get_action(obs, action_mask)

                next_obs, reward, done, truncated, info = env.step(action)
                done = done or truncated

                # Store transition
                trainer.buffer.add(
                    obs=obs,
                    action=action,
                    action_mask=action_mask,
                    reward=reward,
                    done=done,
                    value=value,
                    log_prob=log_prob,
                )

                obs = next_obs
                episode_reward += reward
                episode_steps += 1
                trainer.total_steps += 1
                steps_since_update += 1

                # PPO update when buffer is full
                if steps_since_update >= config.n_steps:
                    # Get value of final state
                    with torch.no_grad():
                        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                        mask_t = torch.tensor(action_mask, dtype=torch.float32, device=device).unsqueeze(0)
                        _, last_value = trainer.policy(obs_t, mask_t)
                        last_value = last_value.squeeze().item()

                    metrics = trainer.train_step(last_value)
                    steps_since_update = 0

                    # Log to TensorBoard
                    writer.add_scalar('train/policy_loss', metrics['policy_loss'], trainer.num_updates)
                    writer.add_scalar('train/value_loss', metrics['value_loss'], trainer.num_updates)
                    writer.add_scalar('train/entropy', metrics['entropy'], trainer.num_updates)

                    if config.wandb_project:
                        import wandb
                        wandb.log({
                            'train/policy_loss': metrics['policy_loss'],
                            'train/value_loss': metrics['value_loss'],
                            'train/entropy': metrics['entropy'],
                        }, step=trainer.num_updates)

            # Episode complete
            env._cleanup()
            episode += 1
            total_games += 1

            if env.won:
                wins += 1
            else:
                losses += 1

            episode_rewards.append(episode_reward)

            # Calculate rolling win rate
            recent_wins = sum(1 for i in range(max(0, total_games - 100), total_games) if i < wins)
            recent_games = min(total_games, 100)
            current_win_rate = recent_wins / max(recent_games, 1)
            win_rates.append(current_win_rate)

            # Track best win rate
            if current_win_rate > best_win_rate and total_games >= 100:
                best_win_rate = current_win_rate

            # Advance curriculum
            stage_changed = curriculum.advance_episode()
            if stage_changed:
                print(f"\n{'='*60}")
                print(f"CURRICULUM: Advancing to stage '{curriculum.get_stage_name()}'")
                print(f"{'='*60}\n")

            # Logging
            if episode % config.log_interval == 0:
                elapsed = time.time() - start_time
                eps_per_hour = episode / (elapsed / 3600) if elapsed > 0 else 0

                avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0

                print(f"\nEpisode {episode:,} | Games: {total_games:,}")
                print(f"  Win rate (100): {current_win_rate*100:.1f}% | Best: {best_win_rate*100:.1f}%")
                print(f"  Avg reward (100): {avg_reward:.4f}")
                print(f"  Total steps: {trainer.total_steps:,} | Updates: {trainer.num_updates:,}")
                print(f"  Stage: {curriculum.get_stage_name()} | Eps/hr: {eps_per_hour:.1f}")
                print(f"  Matchup: {Path(deck1).stem} vs {Path(deck2).stem}")

                # TensorBoard
                writer.add_scalar('episode/reward', episode_reward, episode)
                writer.add_scalar('episode/win_rate_100', current_win_rate, episode)
                writer.add_scalar('episode/steps', episode_steps, episode)
                writer.add_scalar('episode/total_steps', trainer.total_steps, episode)

                if config.wandb_project:
                    import wandb
                    wandb.log({
                        'episode/reward': episode_reward,
                        'episode/win_rate_100': current_win_rate,
                        'episode/steps': episode_steps,
                        'curriculum/stage': curriculum.current_stage,
                    }, step=episode)

            # Checkpointing
            if checkpoint_mgr.should_save(episode):
                training_state = TrainingState(
                    episode=episode,
                    total_steps=trainer.total_steps,
                    total_games=total_games,
                    wins=wins,
                    losses=losses,
                    episode_rewards=episode_rewards[-1000:],  # Keep last 1000
                    win_rates=win_rates[-1000:],
                    best_win_rate=best_win_rate,
                    training_config={
                        **config.to_dict(),
                        'curriculum': curriculum.to_dict(),
                    },
                    timestamp=datetime.now().isoformat(),
                )

                is_best = current_win_rate >= best_win_rate and total_games >= 100
                checkpoint_mgr.save(
                    training_state,
                    trainer.policy,
                    trainer.optimizer,
                    is_best=is_best,
                )

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    finally:
        # Final save
        training_state = TrainingState(
            episode=episode,
            total_steps=trainer.total_steps,
            total_games=total_games,
            wins=wins,
            losses=losses,
            episode_rewards=episode_rewards[-1000:],
            win_rates=win_rates[-1000:],
            best_win_rate=best_win_rate,
            training_config={
                **config.to_dict(),
                'curriculum': curriculum.to_dict(),
            },
            timestamp=datetime.now().isoformat(),
        )

        checkpoint_mgr.save(
            training_state,
            trainer.policy,
            trainer.optimizer,
            checkpoint_name=f"final_ep{episode}.pt",
        )

        writer.close()

        if config.wandb_project:
            import wandb
            wandb.finish()

        print("\nTraining complete!")
        print(f"  Total episodes: {episode:,}")
        print(f"  Total games: {total_games:,}")
        print(f"  Final win rate: {current_win_rate*100:.1f}%")
        print(f"  Best win rate: {best_win_rate*100:.1f}%")


def test_training_mock():
    """Test training with mock data (no daemon needed)."""
    print("Testing training loop with mock data...")

    config = TrainingConfig(n_steps=16)
    device = torch.device("cpu")
    trainer = PPOTrainer(config, device)

    # Simulate training
    for update in range(5):
        print(f"\nUpdate {update + 1}")

        # Collect fake rollout
        for step in range(16):
            obs = np.random.randn(38).astype(np.float32)
            action_mask = np.ones(50, dtype=np.float32)
            action_mask[5:] = 0  # Only 5 valid actions

            action, log_prob, value = trainer.get_action(obs, action_mask)
            reward = np.random.randn() * 0.1
            done = np.random.random() < 0.1

            trainer.buffer.add(obs, action, action_mask, reward, done, value, log_prob)

        # Train
        metrics = trainer.train_step(last_value=0.0)
        print(f"  Policy loss: {metrics['policy_loss']:.6f}")
        print(f"  Value loss: {metrics['value_loss']:.6f}")
        print(f"  Entropy: {metrics['entropy']:.6f}")

    print("\nMock training test passed!")


def main():
    parser = argparse.ArgumentParser(description="MTG RL Training")

    parser.add_argument("--mode", choices=["daemon", "test"], default="test",
                        help="Training mode: daemon (real training) or test (mock)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")

    # Environment settings
    parser.add_argument("--host", type=str, default="localhost",
                        help="Daemon host")
    parser.add_argument("--port", type=int, default=17171,
                        help="Daemon port")
    parser.add_argument("--n-envs", type=int, default=1,
                        help="Number of parallel environments")

    # Training settings
    parser.add_argument("--episodes", type=int, default=100_000,
                        help="Total training episodes")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")

    # Curriculum
    parser.add_argument("--no-curriculum", action="store_true",
                        help="Disable curriculum learning")

    # Logging
    parser.add_argument("--log-interval", type=int, default=100,
                        help="Log every N episodes")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Log directory")

    # S3
    parser.add_argument("--s3-bucket", type=str, default=None,
                        help="S3 bucket for checkpoints")

    # WandB
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="WandB project name")

    args = parser.parse_args()

    if args.mode == "test":
        test_training_mock()
    else:
        config = TrainingConfig(
            daemon_host=args.host,
            daemon_port=args.port,
            n_envs=args.n_envs,
            total_episodes=args.episodes,
            learning_rate=args.lr,
            curriculum_enabled=not args.no_curriculum,
            log_interval=args.log_interval,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            s3_bucket=args.s3_bucket or os.environ.get('S3_BUCKET'),
            wandb_project=args.wandb_project or os.environ.get('WANDB_PROJECT'),
        )

        train_daemon(config, resume=args.resume)


if __name__ == "__main__":
    main()
