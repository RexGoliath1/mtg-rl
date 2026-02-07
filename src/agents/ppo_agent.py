#!/usr/bin/env python3
"""
PPO Agent for MTG

Implements Proximal Policy Optimization (PPO) for training the MTG agent.
Designed to work with the Transformer policy network and MTG environment.

Key Features:
- PPO-Clip objective for stable training
- Generalized Advantage Estimation (GAE)
- Action masking integration
- Parallel environment support
- Gradient clipping and learning rate scheduling
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import time
from collections import deque

from src.models.policy_network import MTGPolicyNetwork, TransformerConfig, StatePreprocessor


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99           # Discount factor
    gae_lambda: float = 0.95      # GAE lambda
    clip_epsilon: float = 0.2     # PPO clip range
    clip_value: float = 0.2       # Value function clip range
    entropy_coef: float = 0.01    # Entropy bonus coefficient
    value_coef: float = 0.5       # Value loss coefficient
    max_grad_norm: float = 0.5    # Gradient clipping

    # Training parameters
    n_steps: int = 128            # Steps per environment before update
    n_epochs: int = 4             # PPO epochs per update
    batch_size: int = 32          # Minibatch size
    n_envs: int = 1               # Number of parallel environments

    # Learning rate schedule
    lr_schedule: str = "linear"   # "constant" or "linear"
    total_timesteps: int = 1_000_000

    # Logging
    log_interval: int = 10        # Log every N updates
    save_interval: int = 100      # Save model every N updates


# =============================================================================
# ROLLOUT BUFFER
# =============================================================================

class RolloutBuffer:
    """Buffer for storing rollout data."""

    def __init__(self, config: PPOConfig, transformer_config: TransformerConfig, device: torch.device):
        self.config = config
        self.transformer_config = transformer_config
        self.device = device

        self.n_steps = config.n_steps
        self.n_envs = config.n_envs
        self.max_seq_len = transformer_config.max_sequence_length

        self.reset()

    def reset(self):
        """Reset the buffer."""
        self.card_embeddings = []
        self.zone_ids = []
        self.card_masks = []
        self.global_features = []
        self.action_masks = []

        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []

        self.advantages = None
        self.returns = None
        self.ptr = 0

    def add(
        self,
        card_embeddings: torch.Tensor,
        zone_ids: torch.Tensor,
        card_mask: torch.Tensor,
        global_features: torch.Tensor,
        action_mask: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        done: bool,
        value: torch.Tensor,
        log_prob: torch.Tensor
    ):
        """Add a transition to the buffer."""
        self.card_embeddings.append(card_embeddings.cpu())
        self.zone_ids.append(zone_ids.cpu())
        self.card_masks.append(card_mask.cpu())
        self.global_features.append(global_features.cpu())
        self.action_masks.append(action_mask.cpu())

        self.actions.append(action.cpu())
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value.cpu())
        self.log_probs.append(log_prob.cpu())

        self.ptr += 1

    def compute_returns_and_advantages(self, last_value: torch.Tensor):
        """Compute GAE advantages and returns."""
        n = len(self.rewards)

        # Convert to numpy for easier manipulation
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = torch.cat(self.values).numpy().squeeze()

        # Add last value for bootstrapping
        values = np.append(values, last_value.cpu().numpy().squeeze())

        # GAE computation
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0

        for t in reversed(range(n)):
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.config.gamma * values[t + 1] * next_non_terminal - values[t]
            last_gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values[:-1]

        self.advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        self.returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Normalize advantages
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_batches(self, batch_size: int):
        """Generate random minibatches for training."""
        n = len(self.rewards)
        indices = np.random.permutation(n)

        # Stack all data
        card_embeddings = torch.cat(self.card_embeddings, dim=0).to(self.device)
        zone_ids = torch.cat(self.zone_ids, dim=0).to(self.device)
        card_masks = torch.cat(self.card_masks, dim=0).to(self.device)
        global_features = torch.cat(self.global_features, dim=0).to(self.device)
        action_masks = torch.cat(self.action_masks, dim=0).to(self.device)
        actions = torch.cat(self.actions, dim=0).to(self.device)
        log_probs = torch.cat(self.log_probs, dim=0).to(self.device)
        values = torch.cat(self.values, dim=0).squeeze().to(self.device)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]

            yield {
                'card_embeddings': card_embeddings[batch_idx],
                'zone_ids': zone_ids[batch_idx],
                'card_masks': card_masks[batch_idx],
                'global_features': global_features[batch_idx],
                'action_masks': action_masks[batch_idx],
                'actions': actions[batch_idx],
                'old_log_probs': log_probs[batch_idx],
                'old_values': values[batch_idx],
                'advantages': self.advantages[batch_idx],
                'returns': self.returns[batch_idx],
            }


# =============================================================================
# PPO AGENT
# =============================================================================

class PPOAgent:
    """PPO Agent for MTG."""

    def __init__(
        self,
        ppo_config: Optional[PPOConfig] = None,
        transformer_config: Optional[TransformerConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.ppo_config = ppo_config or PPOConfig()
        self.transformer_config = transformer_config or TransformerConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create policy network
        self.policy = MTGPolicyNetwork(self.transformer_config).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.ppo_config.learning_rate,
            eps=1e-5
        )

        # State preprocessor
        self.preprocessor = StatePreprocessor(self.transformer_config)

        # Rollout buffer
        self.buffer = RolloutBuffer(
            self.ppo_config, self.transformer_config, self.device
        )

        # Training stats
        self.total_timesteps = 0
        self.num_updates = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

    def get_action(
        self,
        game_state: Dict,
        action_mask: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, Dict]:
        """
        Select an action given the game state.

        Args:
            game_state: Game state dictionary
            action_mask: Valid action mask
            deterministic: If True, select greedy action

        Returns:
            action: Selected action index
            info: Dictionary with value, log_prob, etc.
        """
        # Prepare state tensors
        card_emb, zone_ids, card_mask, global_feat, _ = self.preprocessor.prepare_state(
            game_state, self.device
        )

        # Convert action mask
        action_mask_tensor = torch.tensor(
            action_mask, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # Get action from policy
        with torch.no_grad():
            action, log_prob, probs, value = self.policy.get_action(
                card_emb, zone_ids, card_mask, global_feat, action_mask_tensor,
                deterministic=deterministic
            )

        return action.item(), {
            'log_prob': log_prob,
            'value': value,
            'action_probs': probs,
            'card_embeddings': card_emb,
            'zone_ids': zone_ids,
            'card_mask': card_mask,
            'global_features': global_feat,
            'action_mask': action_mask_tensor,
        }

    def get_action_from_tensors(
        self,
        state_tensors: Tuple[torch.Tensor, ...],
        action_mask: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, Dict]:
        """
        Select an action given preprocessed state tensors.

        Args:
            state_tensors: Tuple of (card_emb, zone_ids, card_mask, global_feat, _)
            action_mask: Valid action mask
            deterministic: If True, select greedy action

        Returns:
            action: Selected action index
            info: Dictionary with value, log_prob, etc.
        """
        card_emb, zone_ids, card_mask, global_feat, _ = state_tensors

        # Move tensors to device if needed
        card_emb = card_emb.to(self.device)
        zone_ids = zone_ids.to(self.device)
        card_mask = card_mask.to(self.device)
        global_feat = global_feat.to(self.device)

        # Convert action mask
        action_mask_tensor = torch.tensor(
            action_mask, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # Get action from policy
        with torch.no_grad():
            action, log_prob, probs, value = self.policy.get_action(
                card_emb, zone_ids, card_mask, global_feat, action_mask_tensor,
                deterministic=deterministic
            )

        return action.item(), {
            'log_prob': log_prob,
            'value': value,
            'action_probs': probs,
            'card_embeddings': card_emb,
            'zone_ids': zone_ids,
            'card_mask': card_mask,
            'global_features': global_feat,
            'action_mask': action_mask_tensor,
        }

    def store_transition(
        self,
        info: Dict,
        action: int,
        reward: float,
        done: bool
    ):
        """Store a transition in the buffer."""
        self.buffer.add(
            info['card_embeddings'],
            info['zone_ids'],
            info['card_mask'],
            info['global_features'],
            info['action_mask'],
            torch.tensor([action], device=self.device),
            reward,
            done,
            info['value'],
            info['log_prob']
        )

    def train_step(self, last_value: torch.Tensor) -> Dict[str, float]:
        """
        Perform PPO update.

        Args:
            last_value: Value estimate for the last state (for bootstrapping)

        Returns:
            Dictionary of training metrics
        """
        # Compute advantages
        self.buffer.compute_returns_and_advantages(last_value)

        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_approx_kl = 0
        n_batches = 0

        # Multiple epochs over the data
        for epoch in range(self.ppo_config.n_epochs):
            for batch in self.buffer.get_batches(self.ppo_config.batch_size):
                # Evaluate current policy on batch
                log_probs, values, entropy = self.policy.evaluate_actions(
                    batch['card_embeddings'],
                    batch['zone_ids'],
                    batch['card_masks'],
                    batch['global_features'],
                    batch['action_masks'],
                    batch['actions']
                )

                values = values.squeeze()

                # Compute ratio
                ratio = torch.exp(log_probs - batch['old_log_probs'])

                # Policy loss (PPO-Clip)
                surr1 = ratio * batch['advantages']
                surr2 = torch.clamp(
                    ratio,
                    1 - self.ppo_config.clip_epsilon,
                    1 + self.ppo_config.clip_epsilon
                ) * batch['advantages']
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                values_pred = values
                values_clipped = batch['old_values'] + torch.clamp(
                    values_pred - batch['old_values'],
                    -self.ppo_config.clip_value,
                    self.ppo_config.clip_value
                )
                value_loss1 = (values_pred - batch['returns']) ** 2
                value_loss2 = (values_clipped - batch['returns']) ** 2
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()

                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss +
                    self.ppo_config.value_coef * value_loss +
                    self.ppo_config.entropy_coef * entropy_loss
                )

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.ppo_config.max_grad_norm
                )
                self.optimizer.step()

                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
                    total_approx_kl += approx_kl.item()

                n_batches += 1

        # Reset buffer
        self.buffer.reset()
        self.num_updates += 1

        return {
            'policy_loss': total_policy_loss / n_batches,
            'value_loss': total_value_loss / n_batches,
            'entropy': total_entropy / n_batches,
            'approx_kl': total_approx_kl / n_batches,
        }

    def update_learning_rate(self, progress: float):
        """Update learning rate based on training progress."""
        if self.ppo_config.lr_schedule == "linear":
            lr = self.ppo_config.learning_rate * (1 - progress)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def save(self, path: str):
        """Save the agent."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_timesteps': self.total_timesteps,
            'num_updates': self.num_updates,
            'ppo_config': self.ppo_config,
            'transformer_config': self.transformer_config,
        }, path)
        print(f"Saved agent to {path}")

    def load(self, path: str):
        """Load the agent."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_timesteps = checkpoint['total_timesteps']
        self.num_updates = checkpoint['num_updates']
        print(f"Loaded agent from {path}")


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train(
    agent: PPOAgent,
    env,
    total_timesteps: int,
    log_interval: int = 10,
    save_interval: int = 100,
    save_path: str = "mtg_agent.pt"
):
    """
    Main training loop.

    Args:
        agent: PPO agent
        env: MTG environment
        total_timesteps: Total training steps
        log_interval: Print stats every N updates
        save_interval: Save model every N updates
        save_path: Path to save model
    """
    print(f"\nStarting training for {total_timesteps:,} timesteps")
    print(f"Device: {agent.device}")
    print("=" * 60)

    start_time = time.time()
    episode_reward = 0
    episode_length = 0

    # Reset environment
    obs, info = env.reset()

    while agent.total_timesteps < total_timesteps:
        # Collect rollouts
        for step in range(agent.ppo_config.n_steps):
            # Convert observation to game state format
            game_state = {
                'our_player': env.current_state.our_player.__dict__ if env.current_state else {},
                'opponent': env.current_state.opponent.__dict__ if env.current_state else {},
                'stack': [],
                'turn': info.get('turn', 1) if info else 1,
            }

            # Get action
            action_mask = env.get_action_mask()
            action, action_info = agent.get_action(game_state, action_mask)

            # Step environment
            next_obs, reward, done, truncated, info = env.step(action)

            # Store transition
            agent.store_transition(action_info, action, reward, done)

            episode_reward += reward
            episode_length += 1
            agent.total_timesteps += 1

            if done:
                # Episode finished
                agent.episode_rewards.append(episode_reward)
                agent.episode_lengths.append(episode_length)
                episode_reward = 0
                episode_length = 0
                obs, info = env.reset()

        # Compute last value for bootstrapping
        game_state = {
            'our_player': env.current_state.our_player.__dict__ if env.current_state else {},
            'opponent': env.current_state.opponent.__dict__ if env.current_state else {},
            'stack': [],
            'turn': info.get('turn', 1) if info else 1,
        }
        with torch.no_grad():
            card_emb, zone_ids, card_mask, global_feat, _ = agent.preprocessor.prepare_state(
                game_state, agent.device
            )
            _, _, _, last_value = agent.policy.get_action(
                card_emb, zone_ids, card_mask, global_feat,
                torch.ones(1, agent.transformer_config.max_actions, device=agent.device),
                deterministic=True
            )

        # PPO update
        train_metrics = agent.train_step(last_value)

        # Update learning rate
        progress = agent.total_timesteps / total_timesteps
        agent.update_learning_rate(progress)

        # Logging
        if agent.num_updates % log_interval == 0:
            elapsed = time.time() - start_time
            fps = agent.total_timesteps / elapsed

            mean_reward = np.mean(agent.episode_rewards) if agent.episode_rewards else 0
            mean_length = np.mean(agent.episode_lengths) if agent.episode_lengths else 0

            print(f"\n[Update {agent.num_updates}] Timesteps: {agent.total_timesteps:,}")
            print(f"  FPS: {fps:.0f}")
            print(f"  Mean reward: {mean_reward:.4f}")
            print(f"  Mean episode length: {mean_length:.0f}")
            print(f"  Policy loss: {train_metrics['policy_loss']:.4f}")
            print(f"  Value loss: {train_metrics['value_loss']:.4f}")
            print(f"  Entropy: {train_metrics['entropy']:.4f}")
            print(f"  Approx KL: {train_metrics['approx_kl']:.4f}")

        # Save checkpoint
        if agent.num_updates % save_interval == 0:
            agent.save(save_path)

    # Final save
    agent.save(save_path)
    print(f"\nTraining complete! Total time: {time.time() - start_time:.0f}s")


# =============================================================================
# TESTING
# =============================================================================

def test_ppo_agent():
    """Test the PPO agent with a mock environment."""
    print("Testing PPO Agent")
    print("=" * 60)

    # Create agent
    ppo_config = PPOConfig(n_steps=16, batch_size=8, n_epochs=2)
    transformer_config = TransformerConfig()
    agent = PPOAgent(ppo_config, transformer_config)

    print(f"\nAgent created on device: {agent.device}")
    print(f"Policy parameters: {sum(p.numel() for p in agent.policy.parameters()):,}")

    # Create mock game state
    game_state = {
        'our_player': {
            'hand': [
                {'name': 'Lightning Bolt', 'mana_cost': '{R}', 'types': 'Instant', 'cmc': 1},
                {'name': 'Mountain', 'mana_cost': '', 'types': 'Basic Land - Mountain', 'cmc': 0},
            ],
            'battlefield': [
                {'name': 'Goblin Guide', 'mana_cost': '{R}', 'types': 'Creature - Goblin Scout',
                 'power': 2, 'toughness': 2, 'is_creature': True, 'tapped': False},
            ],
            'graveyard': [],
            'life': 20,
        },
        'opponent': {
            'battlefield': [
                {'name': 'Tarmogoyf', 'mana_cost': '{1}{G}', 'types': 'Creature - Lhurgoyf',
                 'power': 3, 'toughness': 4, 'is_creature': True, 'tapped': False},
            ],
            'life': 18,
        },
        'stack': [],
        'turn': 3,
    }

    # Create action mask (5 valid actions)
    action_mask = np.zeros(50, dtype=np.float32)
    action_mask[:5] = 1.0

    print("\nTesting action selection...")
    action, info = agent.get_action(game_state, action_mask, deterministic=False)
    print(f"  Selected action: {action}")
    print(f"  Value estimate: {info['value'].item():.4f}")
    print(f"  Log prob: {info['log_prob'].item():.4f}")

    # Test storing transitions
    print("\nTesting transition storage...")
    for i in range(16):
        action, info = agent.get_action(game_state, action_mask)
        reward = np.random.randn() * 0.1
        done = i == 15
        agent.store_transition(info, action, reward, done)

    print(f"  Buffer size: {agent.buffer.ptr}")

    # Test training step
    print("\nTesting training step...")
    with torch.no_grad():
        card_emb, zone_ids, card_mask, global_feat, _ = agent.preprocessor.prepare_state(
            game_state, agent.device
        )
        _, _, _, last_value = agent.policy.get_action(
            card_emb, zone_ids, card_mask, global_feat,
            torch.ones(1, 50, device=agent.device)
        )

    metrics = agent.train_step(last_value)
    print(f"  Policy loss: {metrics['policy_loss']:.4f}")
    print(f"  Value loss: {metrics['value_loss']:.4f}")
    print(f"  Entropy: {metrics['entropy']:.4f}")

    # Test save/load
    print("\nTesting save/load...")
    agent.save("/tmp/test_agent.pt")
    agent.load("/tmp/test_agent.pt")
    print("  Save/load successful!")

    print("\n" + "=" * 60)
    print("PPO Agent test completed successfully!")


if __name__ == "__main__":
    test_ppo_agent()
