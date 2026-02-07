#!/usr/bin/env python3
"""
DEPRECATED: Use train_draft.py for BC training, training_pipeline.py for full pipeline.

This script uses the deprecated data_17lands.py module which assumes a different
CSV format than the actual 17lands public data.

Migration:
    # For behavioral cloning on 17lands data:
    python train_draft.py --sets FDN DSK BLB --epochs 10

    # For full training pipeline:
    python training_pipeline.py --mode full --sets FDN DSK BLB

---

Draft Training Pipeline for MTG (LEGACY)

Implements a multi-stage training pipeline:

1. BEHAVIORAL CLONING (BC) on 17lands data
   - Learn from human draft picks (millions of examples)
   - Supervised learning with cross-entropy loss
   - Fast convergence (hours)

2. DAGGER (Dataset Aggregation) for distribution shift
   - Mix BC policy with Forge drafts
   - Collect more data where policy is uncertain
   - Bridge the gap between 17lands and Forge

3. PPO FINE-TUNING on Forge simulation
   - RL with shaped rewards
   - Learn Forge-specific card pool
   - Optimize for actual win rate

4. TRANSFER TO GAMEPLAY
   - Freeze the card encoder
   - Train gameplay policy heads
   - Leverage learned card understanding

This pipeline is inspired by:
- AlphaStar's supervised learning pre-training
- OpenAI Five's curriculum approach
- Decision Transformer's offline RL
"""

import json
import logging
import os
import time
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_17lands import (
    SeventeenLandsConfig, DraftDataset,
    CardDatabase, create_data_splits, create_synthetic_picks
)
from draft_environment import DraftEnvironment
from draft_policy import DraftPolicyNetwork, DraftPolicyConfig
from shared_card_encoder import CardFeatureExtractor

warnings.warn(
    "draft_training.py is deprecated. Use train_draft.py or training_pipeline.py instead.",
    DeprecationWarning,
    stacklevel=2
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for the full training pipeline."""
    # Stage control
    run_bc: bool = True           # Behavioral cloning on 17lands
    run_dagger: bool = False      # DAgger for distribution shift
    run_rl: bool = True           # RL fine-tuning on Forge
    run_transfer: bool = False    # Transfer to gameplay

    # Behavioral Cloning
    bc_epochs: int = 10
    bc_learning_rate: float = 1e-4
    bc_batch_size: int = 64
    bc_weight_decay: float = 0.01
    bc_warmup_steps: int = 1000
    bc_grad_clip: float = 1.0

    # DAgger
    dagger_iterations: int = 5
    dagger_samples_per_iter: int = 10000
    dagger_beta_start: float = 1.0   # Start with 100% expert
    dagger_beta_end: float = 0.0     # End with 0% expert

    # RL (PPO)
    rl_total_drafts: int = 10000
    rl_learning_rate: float = 3e-5   # Lower than BC
    rl_batch_size: int = 32
    rl_gamma: float = 0.99
    rl_gae_lambda: float = 0.95
    rl_clip_epsilon: float = 0.2
    rl_entropy_coef: float = 0.01
    rl_value_coef: float = 0.5
    rl_n_epochs: int = 4

    # Reward shaping for RL
    reward_card_quality: float = 0.3
    reward_synergy: float = 0.3
    reward_curve: float = 0.2
    reward_color: float = 0.2

    # Logging and checkpointing
    log_dir: str = "logs/draft_training"
    checkpoint_dir: str = "checkpoints/draft"
    log_interval: int = 100
    save_interval: int = 1000
    eval_interval: int = 500

    # Data
    seventeen_lands_dir: str = "data/17lands"
    set_code: Optional[str] = None  # Specific set or all

    def to_dict(self) -> Dict:
        return asdict(self)


class BehavioralCloning:
    """
    Stage 1: Behavioral Cloning on 17lands data.

    Learns to imitate human draft picks using supervised learning.
    This provides a strong initialization for RL fine-tuning.
    """

    def __init__(
        self,
        policy: DraftPolicyNetwork,
        config: TrainingConfig,
        device: torch.device,
    ):
        self.policy = policy
        self.config = config
        self.device = device

        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            policy.parameters(),
            lr=config.bc_learning_rate,
            weight_decay=config.bc_weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = None  # Will be set in train()

        # Loss function
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        writer: Optional[SummaryWriter] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the policy using behavioral cloning.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            writer: TensorBoard writer (optional)

        Returns:
            Dictionary of training metrics over time
        """
        # Calculate total steps for scheduler
        total_steps = len(train_loader) * self.config.bc_epochs
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.bc_learning_rate,
            total_steps=total_steps,
            pct_start=0.1,
        )

        metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }

        global_step = 0
        best_val_acc = 0.0

        for epoch in range(self.config.bc_epochs):
            # Training
            self.policy.train()
            epoch_losses = []
            epoch_correct = 0
            epoch_total = 0

            for batch_idx, batch in enumerate(train_loader):
                # Move to device
                pack_features = batch['pack_features'].to(self.device)
                pool_features = batch['pool_features'].to(self.device)
                pack_mask = batch['pack_mask'].to(self.device)
                pool_mask = batch['pool_mask'].to(self.device)
                pick_index = batch['pick_index'].to(self.device)
                pack_num = batch['pack_number'][0].item()
                pick_num = batch['pick_number'][0].item()

                # Forward pass
                pick_logits, _ = self.policy(
                    pack_features, pool_features, pack_mask, pool_mask,
                    pack_num, pick_num
                )

                # Compute loss
                loss = self.criterion(pick_logits, pick_index)
                loss = (loss * pack_mask[:, 0]).mean()  # Weight by valid samples

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.bc_grad_clip
                )
                self.optimizer.step()
                self.scheduler.step()

                # Track metrics
                epoch_losses.append(loss.item())
                predictions = pick_logits.argmax(dim=-1)
                epoch_correct += (predictions == pick_index).sum().item()
                epoch_total += pick_index.size(0)

                global_step += 1

                # Logging
                if global_step % self.config.log_interval == 0:
                    avg_loss = np.mean(epoch_losses[-100:])
                    acc = epoch_correct / max(epoch_total, 1)
                    lr = self.scheduler.get_last_lr()[0]

                    logger.info(
                        f"Epoch {epoch+1}/{self.config.bc_epochs} "
                        f"Step {global_step} - Loss: {avg_loss:.4f} "
                        f"Acc: {acc:.4f} LR: {lr:.2e}"
                    )

                    if writer:
                        writer.add_scalar('bc/train_loss', avg_loss, global_step)
                        writer.add_scalar('bc/train_acc', acc, global_step)
                        writer.add_scalar('bc/learning_rate', lr, global_step)

            # Epoch summary
            train_loss = np.mean(epoch_losses)
            train_acc = epoch_correct / max(epoch_total, 1)
            metrics['train_loss'].append(train_loss)
            metrics['train_acc'].append(train_acc)

            # Validation
            if val_loader:
                val_loss, val_acc = self.evaluate(val_loader)
                metrics['val_loss'].append(val_loss)
                metrics['val_acc'].append(val_acc)

                logger.info(
                    f"Epoch {epoch+1} complete - "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
                )

                if writer:
                    writer.add_scalar('bc/val_loss', val_loss, epoch)
                    writer.add_scalar('bc/val_acc', val_acc, epoch)

                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_checkpoint(
                        os.path.join(self.config.checkpoint_dir, 'bc_best.pt'),
                        epoch, global_step, metrics
                    )

            # Save regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(
                    os.path.join(self.config.checkpoint_dir, f'bc_epoch_{epoch+1}.pt'),
                    epoch, global_step, metrics
                )

        return metrics

    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """Evaluate on a data loader."""
        self.policy.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in loader:
                pack_features = batch['pack_features'].to(self.device)
                pool_features = batch['pool_features'].to(self.device)
                pack_mask = batch['pack_mask'].to(self.device)
                pool_mask = batch['pool_mask'].to(self.device)
                pick_index = batch['pick_index'].to(self.device)

                pick_logits, _ = self.policy(
                    pack_features, pool_features, pack_mask, pool_mask
                )

                loss = self.criterion(pick_logits, pick_index).mean()
                total_loss += loss.item() * pick_index.size(0)

                predictions = pick_logits.argmax(dim=-1)
                total_correct += (predictions == pick_index).sum().item()
                total_samples += pick_index.size(0)

        return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        global_step: int,
        metrics: Dict[str, List[float]]
    ):
        """Save training checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config.to_dict(),
        }, path)
        logger.info(f"Saved checkpoint to {path}")


class PPODraftTrainer:
    """
    Stage 3: PPO Fine-tuning on Forge simulation.

    After behavioral cloning, this refines the policy using RL
    with shaped rewards on actual Forge draft simulations.
    """

    def __init__(
        self,
        policy: DraftPolicyNetwork,
        config: TrainingConfig,
        device: torch.device,
    ):
        self.policy = policy
        self.config = config
        self.device = device

        # Optimizer (lower learning rate than BC)
        self.optimizer = optim.Adam(
            policy.parameters(),
            lr=config.rl_learning_rate,
        )

        # Feature extractor for reward computation
        self.feature_extractor = CardFeatureExtractor(policy.config.card_encoder_config)

    def train(
        self,
        env: DraftEnvironment,
        writer: Optional[SummaryWriter] = None,
    ) -> Dict[str, List[float]]:
        """
        Train using PPO on Forge draft simulation.

        Args:
            env: Draft environment connected to Forge daemon
            writer: TensorBoard writer

        Returns:
            Dictionary of training metrics
        """
        metrics = {
            'episode_reward': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
        }

        total_drafts = 0
        global_step = 0

        while total_drafts < self.config.rl_total_drafts:
            # Collect a draft episode
            episode_data = self._collect_episode(env)

            if episode_data is None:
                logger.warning("Failed to collect episode, retrying...")
                time.sleep(1)
                continue

            total_drafts += 1
            global_step += len(episode_data['rewards'])

            # Compute returns and advantages
            returns, advantages = self._compute_gae(episode_data)
            episode_data['returns'] = returns
            episode_data['advantages'] = advantages

            # PPO update
            ppo_metrics = self._ppo_update(episode_data)

            # Track metrics
            episode_reward = sum(episode_data['rewards'])
            metrics['episode_reward'].append(episode_reward)
            metrics['policy_loss'].append(ppo_metrics['policy_loss'])
            metrics['value_loss'].append(ppo_metrics['value_loss'])
            metrics['entropy'].append(ppo_metrics['entropy'])

            # Logging
            if total_drafts % self.config.log_interval == 0:
                avg_reward = np.mean(metrics['episode_reward'][-100:])
                logger.info(
                    f"Draft {total_drafts}/{self.config.rl_total_drafts} - "
                    f"Reward: {episode_reward:.4f} Avg(100): {avg_reward:.4f}"
                )

                if writer:
                    writer.add_scalar('rl/episode_reward', episode_reward, total_drafts)
                    writer.add_scalar('rl/avg_reward_100', avg_reward, total_drafts)
                    writer.add_scalar('rl/policy_loss', ppo_metrics['policy_loss'], total_drafts)
                    writer.add_scalar('rl/value_loss', ppo_metrics['value_loss'], total_drafts)
                    writer.add_scalar('rl/entropy', ppo_metrics['entropy'], total_drafts)

            # Save checkpoint
            if total_drafts % self.config.save_interval == 0:
                self.save_checkpoint(
                    os.path.join(self.config.checkpoint_dir, f'rl_draft_{total_drafts}.pt'),
                    total_drafts, global_step, metrics
                )

        return metrics

    def _collect_episode(self, env: DraftEnvironment) -> Optional[Dict]:
        """Collect a single draft episode."""
        try:
            obs, info = env.reset()
        except Exception as e:
            logger.error(f"Failed to reset environment: {e}")
            return None

        episode_data = {
            'pack_features': [],
            'pool_features': [],
            'pack_masks': [],
            'pool_masks': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'rewards': [],
        }

        pool = []
        self.policy.eval()

        while not env.is_complete():
            # Prepare state tensors
            pack = obs.get('pack', [])
            if not pack:
                break

            pack_features = self._cards_to_features(pack)
            pool_features = self._pool_to_features(pool)

            pack_tensor = torch.tensor(pack_features, dtype=torch.float32, device=self.device).unsqueeze(0)
            pool_tensor = torch.tensor(pool_features, dtype=torch.float32, device=self.device).unsqueeze(0)
            pack_mask = torch.ones(1, len(pack), device=self.device)
            pool_mask = torch.ones(1, max(len(pool), 1), device=self.device)

            # Get action from policy
            with torch.no_grad():
                action, log_prob, probs, value = self.policy.get_action(
                    pack_tensor, pool_tensor, pack_mask, pool_mask,
                    obs.get('pack_num', 1), obs.get('pick_num', 1)
                )

            # Store pre-action data
            episode_data['pack_features'].append(pack_tensor)
            episode_data['pool_features'].append(pool_tensor)
            episode_data['pack_masks'].append(pack_mask)
            episode_data['pool_masks'].append(pool_mask)
            episode_data['actions'].append(action)
            episode_data['log_probs'].append(log_prob)
            episode_data['values'].append(value)

            # Take action
            action_idx = action.item()
            obs, reward, done, truncated, info = env.step(action_idx)

            # Compute shaped reward
            picked_card = pack[action_idx] if action_idx < len(pack) else pack[0]
            shaped_reward = self._compute_shaped_reward(picked_card, pool, obs.get('pack_num', 1))
            episode_data['rewards'].append(shaped_reward)

            # Update pool
            pool.append(picked_card.get('name', ''))

            if done:
                break

        # Convert lists to tensors
        if not episode_data['actions']:
            return None

        episode_data['actions'] = torch.cat(episode_data['actions'])
        episode_data['log_probs'] = torch.cat(episode_data['log_probs'])
        episode_data['values'] = torch.cat(episode_data['values'])

        return episode_data

    def _compute_shaped_reward(
        self,
        card: Dict,
        pool: List[str],
        pack_num: int
    ) -> float:
        """
        Compute shaped reward for a pick.

        Combines:
        - Card quality (rarity, stats)
        - Synergy with pool (color matching, types)
        - Curve considerations (CMC distribution)
        - Color discipline (staying on-color)
        """
        reward = 0.0

        # Card quality reward
        rarity = card.get('rarity', 'Common')
        rarity_values = {'Common': 0.1, 'Uncommon': 0.3, 'Rare': 0.6, 'Mythic': 0.8}
        quality_reward = rarity_values.get(rarity, 0.1)
        reward += self.config.reward_card_quality * quality_reward

        # Type bonus (creatures are generally valuable)
        card_type = card.get('type', '')
        if 'Creature' in card_type:
            reward += 0.1

        # Synergy reward (simplified - just color matching)
        card_colors = self._extract_colors(card.get('mana_cost', ''))
        pool_colors = self._get_pool_colors(pool)

        if card_colors and pool_colors:
            color_match = len(card_colors & pool_colors) / len(card_colors)
            reward += self.config.reward_synergy * color_match

        # Curve reward (penalize too many high-CMC cards early)
        cmc = card.get('cmc', 0)
        pool_size = len(pool)

        if pack_num == 1 and cmc > 5:
            reward -= 0.1  # Penalize early high-cost picks
        elif pool_size > 30 and cmc <= 2:
            reward += 0.1  # Reward late low-curve picks

        # Color discipline (penalize adding new colors late)
        if pack_num >= 2 and card_colors and pool_colors:
            new_colors = card_colors - pool_colors
            if new_colors:
                reward -= self.config.reward_color * 0.2

        return reward

    def _extract_colors(self, mana_cost: str) -> set:
        """Extract colors from mana cost string."""
        colors = set()
        for c in ['W', 'U', 'B', 'R', 'G']:
            if c in mana_cost.upper():
                colors.add(c)
        return colors

    def _get_pool_colors(self, pool: List[str]) -> set:
        """Get colors represented in the pool (simplified)."""
        # In practice, you'd look up actual card colors
        # This is a placeholder
        return set()

    def _cards_to_features(self, cards: List[Dict]) -> np.ndarray:
        """Convert cards to feature array."""
        input_dim = self.policy.config.card_encoder_config.input_dim
        features = np.zeros((15, input_dim), dtype=np.float32)

        for i, card in enumerate(cards[:15]):
            features[i] = self.feature_extractor.extract(card)

        return features

    def _pool_to_features(self, pool: List[str]) -> np.ndarray:
        """Convert pool to feature array."""
        input_dim = self.policy.config.card_encoder_config.input_dim
        pool_size = max(len(pool), 1)
        features = np.zeros((pool_size, input_dim), dtype=np.float32)
        return features

    def _compute_gae(self, episode_data: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE returns and advantages."""
        rewards = episode_data['rewards']
        values = episode_data['values'].squeeze().cpu().numpy()

        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        gae = 0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.config.rl_gamma * next_value - values[t]
            gae = delta + self.config.rl_gamma * self.config.rl_gae_lambda * gae
            advantages[t] = gae

        returns = advantages + values

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def _ppo_update(self, episode_data: Dict) -> Dict[str, float]:
        """Perform PPO update on episode data."""
        self.policy.train()

        n = len(episode_data['rewards'])
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.config.rl_n_epochs):
            # Process each timestep (could batch for efficiency)
            for t in range(n):
                pack_features = episode_data['pack_features'][t]
                pool_features = episode_data['pool_features'][t]
                pack_mask = episode_data['pack_masks'][t]
                pool_mask = episode_data['pool_masks'][t]
                action = episode_data['actions'][t:t+1]
                old_log_prob = episode_data['log_probs'][t]
                advantage = episode_data['advantages'][t]
                return_val = episode_data['returns'][t]

                # Forward pass
                log_prob, value, entropy = self.policy.evaluate_actions(
                    pack_features, pool_features, action, pack_mask, pool_mask
                )

                # Compute ratio
                ratio = torch.exp(log_prob - old_log_prob)

                # Clipped surrogate objective
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.config.rl_clip_epsilon,
                                   1 + self.config.rl_clip_epsilon) * advantage
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(value.squeeze(), return_val)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss +
                    self.config.rl_value_coef * value_loss +
                    self.config.rl_entropy_coef * entropy_loss
                )

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        return {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1),
        }

    def save_checkpoint(
        self,
        path: str,
        total_drafts: int,
        global_step: int,
        metrics: Dict[str, List[float]]
    ):
        """Save training checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'total_drafts': total_drafts,
            'global_step': global_step,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config.to_dict(),
        }, path)
        logger.info(f"Saved checkpoint to {path}")


def run_full_pipeline(config: TrainingConfig):
    """
    Run the full training pipeline.

    1. Behavioral Cloning on 17lands data
    2. (Optional) DAgger for distribution shift
    3. PPO fine-tuning on Forge
    4. (Optional) Transfer to gameplay
    """
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config.log_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Save config
    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create policy
    draft_config = DraftPolicyConfig()
    policy = DraftPolicyNetwork(draft_config).to(device)
    logger.info(f"Created policy with {sum(p.numel() for p in policy.parameters()):,} parameters")

    # TensorBoard
    writer = SummaryWriter(run_dir)

    # Stage 1: Behavioral Cloning
    if config.run_bc:
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 1: BEHAVIORAL CLONING")
        logger.info("=" * 60)

        # Load or create training data
        extractor = CardFeatureExtractor(draft_config.card_encoder_config)
        card_db = CardDatabase()

        # Try to load 17lands data, fall back to synthetic
        picks = create_synthetic_picks(10000)  # Use synthetic for now
        logger.info(f"Using {len(picks)} training picks")

        train_picks, val_picks, test_picks = create_data_splits(
            picks, SeventeenLandsConfig()
        )

        train_dataset = DraftDataset(train_picks, extractor, card_db)
        val_dataset = DraftDataset(val_picks, extractor, card_db)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.bc_batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.bc_batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Train
        bc_trainer = BehavioralCloning(policy, config, device)
        bc_metrics = bc_trainer.train(train_loader, val_loader, writer)

        logger.info(f"BC complete - Final train acc: {bc_metrics['train_acc'][-1]:.4f}")
        if bc_metrics['val_acc']:
            logger.info(f"Final val acc: {bc_metrics['val_acc'][-1]:.4f}")

    # Stage 3: RL Fine-tuning
    if config.run_rl:
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 3: PPO FINE-TUNING ON FORGE")
        logger.info("=" * 60)

        # Check if draft daemon is running
        from draft_environment import check_draft_daemon
        status = check_draft_daemon()

        if status['status'] != 'running':
            logger.warning("Draft daemon not running - skipping RL training")
            logger.info("Start the daemon with: java -jar forge.jar draft -p 17272")
        else:
            env = DraftEnvironment()

            rl_trainer = PPODraftTrainer(policy, config, device)
            rl_metrics = rl_trainer.train(env, writer)

            env.disconnect()

            logger.info(f"RL complete - Final avg reward: {np.mean(rl_metrics['episode_reward'][-100:]):.4f}")

    # Save final model
    final_path = os.path.join(config.checkpoint_dir, 'final_draft_policy.pt')
    policy.save(final_path, save_encoder_separately=True)
    logger.info(f"Saved final model to {final_path}")

    # Also save just the encoder for transfer
    encoder_path = os.path.join(config.checkpoint_dir, 'shared_encoder.pt')
    policy.card_encoder.save(encoder_path)
    logger.info(f"Saved shared encoder to {encoder_path}")

    writer.close()
    logger.info("\nTraining pipeline complete!")


def test_training():
    """Quick test of the training pipeline."""
    print("Testing Draft Training Pipeline")
    print("=" * 60)

    # Create minimal config
    config = TrainingConfig(
        bc_epochs=2,
        bc_batch_size=8,
        run_bc=True,
        run_rl=False,  # Skip RL for quick test
        log_dir="logs/test_draft",
        checkpoint_dir="checkpoints/test_draft",
    )

    # Device
    device = torch.device("cpu")

    # Create policy
    draft_config = DraftPolicyConfig()
    policy = DraftPolicyNetwork(draft_config).to(device)

    # Create synthetic data
    extractor = CardFeatureExtractor(draft_config.card_encoder_config)
    card_db = CardDatabase()

    picks = create_synthetic_picks(100)
    train_picks, val_picks, _ = create_data_splits(picks, SeventeenLandsConfig())

    train_dataset = DraftDataset(train_picks[:50], extractor, card_db)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Quick BC training
    bc_trainer = BehavioralCloning(policy, config, device)

    print("\nTraining for 2 epochs on 50 samples...")
    policy.train()

    for epoch in range(2):
        for batch in train_loader:
            pack_features = batch['pack_features'].to(device)
            pool_features = batch['pool_features'].to(device)
            pack_mask = batch['pack_mask'].to(device)
            pool_mask = batch['pool_mask'].to(device)
            pick_index = batch['pick_index'].to(device)

            pick_logits, _ = policy(
                pack_features, pool_features, pack_mask, pool_mask
            )

            loss = F.cross_entropy(pick_logits, pick_index)

            bc_trainer.optimizer.zero_grad()
            loss.backward()
            bc_trainer.optimizer.step()

        print(f"Epoch {epoch + 1} complete")

    print("\n" + "=" * 60)
    print("Training pipeline test completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Draft Training Pipeline")
    parser.add_argument("--mode", choices=["full", "bc", "rl", "test"],
                        default="test", help="Training mode")
    parser.add_argument("--bc-epochs", type=int, default=10)
    parser.add_argument("--rl-drafts", type=int, default=10000)
    parser.add_argument("--checkpoint-dir", default="checkpoints/draft")
    parser.add_argument("--log-dir", default="logs/draft_training")

    args = parser.parse_args()

    if args.mode == "test":
        test_training()
    else:
        config = TrainingConfig(
            bc_epochs=args.bc_epochs,
            rl_total_drafts=args.rl_drafts,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            run_bc=(args.mode in ["full", "bc"]),
            run_rl=(args.mode in ["full", "rl"]),
        )
        run_full_pipeline(config)
