#!/usr/bin/env python3
"""
Unified Training Pipeline for MTG RL

This module provides a complete training pipeline that:
1. Downloads/processes 17lands data
2. Trains draft encoder via behavioral cloning
3. Transfers to gameplay encoder
4. Fine-tunes via self-play RL
5. Evaluates against baseline agents

Usage:
    # Full pipeline
    python training_pipeline.py --mode full --sets FDN DSK BLB

    # Just behavioral cloning
    python training_pipeline.py --mode bc --sets FDN --epochs 10

    # Just RL fine-tuning (requires pretrained model)
    python training_pipeline.py --mode rl --checkpoint checkpoints/bc_best.pt

    # Evaluation only
    python training_pipeline.py --mode eval --checkpoint checkpoints/best.pt
"""

import os
import sys
import json
import time
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# Local imports
from entity_encoder import EntityEncoder, EntityEncoderConfig, EncoderMode
from shared_card_encoder import SharedCardEncoder, CardEncoderConfig, CardFeatureExtractor
from text_embeddings import PretrainedTextEmbedder, TextEmbeddingConfig, CardEmbeddingDatabase


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for the full training pipeline."""

    # Data
    data_dir: str = "data/17lands"
    sets: List[str] = field(default_factory=lambda: ["FDN", "DSK", "BLB"])

    # Model
    use_entity_encoder: bool = False  # Use simpler encoder for now (True = EntityEncoder)
    use_text_embeddings: bool = True  # Use pretrained text embeddings

    # Behavioral cloning
    bc_epochs: int = 10
    bc_batch_size: int = 256
    bc_lr: float = 1e-4
    bc_weight_decay: float = 0.01
    bc_warmup_steps: int = 1000
    bc_grad_clip: float = 1.0

    # RL fine-tuning
    rl_episodes: int = 10000
    rl_batch_size: int = 64
    rl_lr: float = 3e-5
    rl_gamma: float = 0.99
    rl_gae_lambda: float = 0.95
    rl_clip_eps: float = 0.2
    rl_entropy_coef: float = 0.01
    rl_value_coef: float = 0.5
    rl_ppo_epochs: int = 4

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    save_every: int = 1000
    eval_every: int = 500

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    mixed_precision: bool = True

    # Rate limiting (for deployment)
    max_inferences_per_second: int = 100
    max_batch_size: int = 32


# =============================================================================
# DRAFT POLICY NETWORK (for BC training)
# =============================================================================

class DraftPolicyHead(nn.Module):
    """
    Policy head for draft pick prediction.

    Takes card embeddings and outputs pick probabilities.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super().__init__()

        self.pool_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.pack_pool_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )

        self.pick_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        pack_embeddings: torch.Tensor,
        pool_embeddings: torch.Tensor,
        pack_mask: Optional[torch.Tensor] = None,
        pool_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute pick logits and value.

        Args:
            pack_embeddings: [batch, pack_size, embed_dim] - cards in current pack
            pool_embeddings: [batch, pool_size, embed_dim] - cards already drafted
            pack_mask: [batch, pack_size] - 1 for valid cards
            pool_mask: [batch, pool_size] - 1 for valid cards

        Returns:
            pick_logits: [batch, pack_size] - logits for each card
            value: [batch, 1] - estimated value of current state
        """
        batch_size, pack_size, embed_dim = pack_embeddings.shape

        # Encode pool context
        if pool_embeddings.shape[1] > 0:
            pool_encoded = self.pool_encoder(pool_embeddings)
            if pool_mask is not None:
                pool_encoded = pool_encoded * pool_mask.unsqueeze(-1)
            pool_context = pool_encoded.mean(dim=1, keepdim=True)  # [batch, 1, hidden]
        else:
            pool_context = torch.zeros(batch_size, 1, 256, device=pack_embeddings.device)

        # Reduce pack embeddings
        pack_reduced = self.pool_encoder(pack_embeddings)  # Reuse encoder

        # Cross-attention: pack attends to pool
        if pool_embeddings.shape[1] > 0:
            pool_reduced = self.pool_encoder(pool_embeddings)
            attn_mask = None
            if pool_mask is not None:
                attn_mask = (pool_mask == 0)

            pack_contextualized, _ = self.pack_pool_attention(
                pack_reduced, pool_reduced, pool_reduced,
                key_padding_mask=attn_mask
            )
        else:
            pack_contextualized = pack_reduced

        # Combine with pool context and score
        pool_context_expanded = pool_context.expand(-1, pack_size, -1)
        combined = torch.cat([pack_contextualized, pool_context_expanded], dim=-1)
        pick_logits = self.pick_scorer(combined).squeeze(-1)  # [batch, pack_size]

        # Apply mask
        if pack_mask is not None:
            pick_logits = pick_logits.masked_fill(pack_mask == 0, float('-inf'))

        # Value estimate
        value = self.value_head(pool_context.squeeze(1))

        return pick_logits, value


class DraftModel(nn.Module):
    """
    Complete draft model combining encoder and policy head.
    """

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config

        # Initialize encoder
        if config.use_entity_encoder:
            encoder_config = EntityEncoderConfig()
            self.encoder = EntityEncoder(encoder_config)
            self.embed_dim = encoder_config.output_dim
        else:
            encoder_config = CardEncoderConfig()
            self.encoder = SharedCardEncoder(encoder_config)
            self.embed_dim = encoder_config.output_dim

        # Policy head
        self.policy_head = DraftPolicyHead(self.embed_dim)

    def forward(
        self,
        pack_features: torch.Tensor,
        pool_features: torch.Tensor,
        pack_mask: Optional[torch.Tensor] = None,
        pool_mask: Optional[torch.Tensor] = None,
        pack_state: Optional[torch.Tensor] = None,
        pool_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for draft prediction.

        Args:
            pack_features: [batch, pack_size, feature_dim] - identity features
            pool_features: [batch, pool_size, feature_dim] - identity features
            pack_mask: [batch, pack_size]
            pool_mask: [batch, pool_size]
            pack_state: Optional state features (for EntityEncoder)
            pool_state: Optional state features (for EntityEncoder)

        Returns:
            pick_logits: [batch, pack_size]
            value: [batch, 1]
        """
        # Encode pack
        if isinstance(self.encoder, EntityEncoder):
            pack_emb, _ = self.encoder(
                pack_features, pack_state, pack_mask,
                mode=EncoderMode.DRAFT, return_pooled=True
            )
        else:
            pack_emb = self.encoder(pack_features, pack_mask)

        # Encode pool
        if pool_features.shape[1] > 0:
            if isinstance(self.encoder, EntityEncoder):
                pool_emb, _ = self.encoder(
                    pool_features, pool_state, pool_mask,
                    mode=EncoderMode.DRAFT, return_pooled=True
                )
            else:
                pool_emb = self.encoder(pool_features, pool_mask)
        else:
            pool_emb = torch.zeros(pack_features.shape[0], 0, self.embed_dim,
                                   device=pack_features.device)

        # Get pick logits and value
        return self.policy_head(pack_emb, pool_emb, pack_mask, pool_mask)


# =============================================================================
# DATA LOADING
# =============================================================================

class DraftPickDataset(Dataset):
    """
    Dataset for behavioral cloning on 17lands draft data.

    Loads preprocessed draft picks and converts to model input format.
    """

    def __init__(
        self,
        data_dir: str,
        sets: List[str],
        feature_extractor,
        text_embedder: Optional[PretrainedTextEmbedder] = None,
        max_samples: int = None,
    ):
        self.data_dir = Path(data_dir)
        self.sets = sets
        self.feature_extractor = feature_extractor
        self.text_embedder = text_embedder
        self.samples = []

        self._load_data(max_samples)

    def _load_data(self, max_samples: Optional[int]):
        """Load draft data from processed files."""
        for set_code in self.sets:
            processed_path = self.data_dir / f"{set_code}_processed.json"

            if not processed_path.exists():
                print(f"Warning: No processed data for {set_code}")
                continue

            with open(processed_path, 'r') as f:
                data = json.load(f)

            self.samples.extend(data.get('picks', []))

            if max_samples and len(self.samples) >= max_samples:
                self.samples = self.samples[:max_samples]
                break

        print(f"Loaded {len(self.samples)} draft picks")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Extract features for pack cards
        pack_features = []
        for card in sample['pack']:
            # Handle both EntityFeatureExtractor and CardFeatureExtractor
            if hasattr(self.feature_extractor, 'extract_identity'):
                # EntityFeatureExtractor expects EntityFeatures dataclass
                from entity_encoder import EntityFeatures
                entity = EntityFeatures(
                    card_name=card.get('name', ''),
                    mana_cost=card.get('mana_cost', ''),
                    card_types=card.get('type', '').split() if isinstance(card.get('type'), str) else card.get('types', []),
                    keywords=card.get('keywords', []),
                    base_power=card.get('power'),
                    base_toughness=card.get('toughness'),
                    rarity=card.get('rarity', 'common'),
                )
                features = self.feature_extractor.extract_identity(entity)
            else:
                features = self.feature_extractor.extract(card)
            pack_features.append(features)

        # Pad pack to 15 cards
        pack_size = len(pack_features)
        while len(pack_features) < 15:
            pack_features.append(np.zeros_like(pack_features[0]))

        # Extract features for pool cards
        pool_features = []
        for card in sample.get('pool', []):
            if hasattr(self.feature_extractor, 'extract_identity'):
                from entity_encoder import EntityFeatures
                entity = EntityFeatures(
                    card_name=card.get('name', ''),
                    mana_cost=card.get('mana_cost', ''),
                    card_types=card.get('type', '').split() if isinstance(card.get('type'), str) else card.get('types', []),
                    keywords=card.get('keywords', []),
                    base_power=card.get('power'),
                    base_toughness=card.get('toughness'),
                    rarity=card.get('rarity', 'common'),
                )
                features = self.feature_extractor.extract_identity(entity)
            else:
                features = self.feature_extractor.extract(card)
            pool_features.append(features)

        # Pad pool to 45 cards
        pool_size = len(pool_features)
        if pool_size == 0:
            pool_features = [np.zeros(pack_features[0].shape)]
            pool_size = 0
        while len(pool_features) < 45:
            pool_features.append(np.zeros_like(pool_features[0]))

        # Create masks
        pack_mask = np.zeros(15)
        pack_mask[:pack_size] = 1
        pool_mask = np.zeros(45)
        pool_mask[:pool_size] = 1

        # Get pick index
        pick_idx = sample.get('pick_index', 0)

        return {
            'pack_features': np.stack(pack_features).astype(np.float32),
            'pool_features': np.stack(pool_features).astype(np.float32),
            'pack_mask': pack_mask.astype(np.float32),
            'pool_mask': pool_mask.astype(np.float32),
            'pick_index': pick_idx,
        }


# =============================================================================
# BEHAVIORAL CLONING TRAINER
# =============================================================================

class BCTrainer:
    """Trainer for behavioral cloning on draft data."""

    def __init__(self, config: TrainingConfig, model: DraftModel):
        self.config = config
        self.model = model.to(config.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.bc_lr,
            weight_decay=config.bc_weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.bc_lr,
            epochs=config.bc_epochs,
            steps_per_epoch=1000,  # Will be updated
        )

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None

        # Logging
        self.writer = SummaryWriter(config.log_dir)

        # Checkpointing
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            pack_features = batch['pack_features'].to(self.config.device)
            pool_features = batch['pool_features'].to(self.config.device)
            pack_mask = batch['pack_mask'].to(self.config.device)
            pool_mask = batch['pool_mask'].to(self.config.device)
            pick_index = batch['pick_index'].to(self.config.device)

            # Forward pass
            self.optimizer.zero_grad()

            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    logits, _ = self.model(pack_features, pool_features, pack_mask, pool_mask)
                    loss = self.criterion(logits, pick_index)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.bc_grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, _ = self.model(pack_features, pool_features, pack_mask, pool_mask)
                loss = self.criterion(logits, pick_index)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.bc_grad_clip)
                self.optimizer.step()

            self.scheduler.step()

            # Metrics
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == pick_index).sum().item()
            total += pick_index.shape[0]

            # Log
            if batch_idx % 100 == 0:
                step = epoch * len(dataloader) + batch_idx
                self.writer.add_scalar('train/loss', loss.item(), step)
                self.writer.add_scalar('train/accuracy', correct / total, step)
                self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], step)

        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total,
        }

    def validate(self, dataloader: DataLoader) -> Dict:
        """Validate on held-out data."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                pack_features = batch['pack_features'].to(self.config.device)
                pool_features = batch['pool_features'].to(self.config.device)
                pack_mask = batch['pack_mask'].to(self.config.device)
                pool_mask = batch['pool_mask'].to(self.config.device)
                pick_index = batch['pick_index'].to(self.config.device)

                logits, _ = self.model(pack_features, pool_features, pack_mask, pool_mask)
                loss = self.criterion(logits, pick_index)

                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                correct += (preds == pick_index).sum().item()
                total += pick_index.shape[0]

        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total,
        }

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': asdict(self.config),
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})


# =============================================================================
# SELF-PLAY RL TRAINER
# =============================================================================

class SelfPlayTrainer:
    """
    Trainer for RL fine-tuning via self-play.

    Uses PPO with shaped rewards for draft quality.
    """

    def __init__(self, config: TrainingConfig, model: DraftModel):
        self.config = config
        self.model = model.to(config.device)

        # Separate actor and critic optimizers
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.rl_lr,
        )

        # Experience buffer
        self.buffer = []

        # Logging
        self.writer = SummaryWriter(config.log_dir + "/rl")

    def compute_rewards(self, draft_result: Dict) -> float:
        """
        Compute shaped reward for a draft.

        Factors:
        - Card quality (based on 17lands win rate data)
        - Color discipline (fewer colors = better mana)
        - Curve quality (good distribution of costs)
        - Synergy bonuses (tribal, mechanics)
        """
        reward = 0.0

        # Card quality (placeholder - would use actual 17lands data)
        avg_quality = sum(card.get('gih_wr', 0.5) for card in draft_result['pool']) / len(draft_result['pool'])
        reward += (avg_quality - 0.5) * 10  # Center around 0

        # Color discipline
        colors = set()
        for card in draft_result['pool']:
            colors.update(card.get('colors', []))
        color_penalty = max(0, len(colors) - 2) * 0.5
        reward -= color_penalty

        # Curve quality (want 1-2-3-4-5+ distribution roughly)
        curve = [0] * 6
        for card in draft_result['pool']:
            cmc = min(card.get('cmc', 0), 5)
            curve[cmc] += 1

        # Prefer having some cards at each CMC
        curve_bonus = sum(1 for c in curve[1:5] if c >= 3) * 0.5
        reward += curve_bonus

        return reward

    def collect_episode(self, env) -> List[Dict]:
        """Collect one draft episode."""
        # This would interface with Forge draft daemon
        # For now, return placeholder
        return []

    def ppo_update(self, episodes: List[List[Dict]]):
        """Perform PPO update on collected episodes."""
        # Flatten episodes
        all_states = []
        all_actions = []
        all_rewards = []
        all_values = []
        all_log_probs = []

        for episode in episodes:
            for step in episode:
                all_states.append(step['state'])
                all_actions.append(step['action'])
                all_rewards.append(step['reward'])
                all_values.append(step['value'])
                all_log_probs.append(step['log_prob'])

        # Compute advantages (GAE)
        advantages = self._compute_gae(all_rewards, all_values)

        # PPO epochs
        for _ in range(self.config.rl_ppo_epochs):
            # Sample mini-batches
            indices = np.random.permutation(len(all_states))

            for start in range(0, len(indices), self.config.rl_batch_size):
                batch_indices = indices[start:start + self.config.rl_batch_size]

                # Get batch
                batch_states = [all_states[i] for i in batch_indices]
                batch_actions = torch.tensor([all_actions[i] for i in batch_indices])
                batch_advantages = torch.tensor([advantages[i] for i in batch_indices])
                batch_old_log_probs = torch.tensor([all_log_probs[i] for i in batch_indices])
                batch_returns = batch_advantages + torch.tensor([all_values[i] for i in batch_indices])

                # Forward pass
                logits, values = self._forward_batch(batch_states)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # PPO loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.config.rl_clip_eps, 1 + self.config.rl_clip_eps)
                policy_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()

                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)

                # Total loss
                loss = policy_loss + self.config.rl_value_coef * value_loss - self.config.rl_entropy_coef * entropy

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def _compute_gae(self, rewards: List[float], values: List[float]) -> List[float]:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.config.rl_gamma * next_value - values[t]
            gae = delta + self.config.rl_gamma * self.config.rl_gae_lambda * gae
            advantages.insert(0, gae)

        return advantages

    def _forward_batch(self, states: List) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a batch of states."""
        # Convert states to tensors and run through model
        # Placeholder implementation
        batch_size = len(states)
        return (
            torch.zeros(batch_size, 15),  # logits
            torch.zeros(batch_size, 1),   # values
        )


# =============================================================================
# EVALUATION
# =============================================================================

class DraftEvaluator:
    """Evaluates draft model against baselines."""

    def __init__(self, model: DraftModel, config: TrainingConfig):
        self.model = model
        self.config = config

    def evaluate_vs_random(self, num_drafts: int = 100) -> Dict:
        """Evaluate against random drafter."""
        wins = 0
        for _ in range(num_drafts):
            # Simulate draft
            # Compare final deck quality
            pass
        return {'win_rate': wins / num_drafts}

    def evaluate_vs_raredraft(self, num_drafts: int = 100) -> Dict:
        """Evaluate against rare-drafter baseline."""
        wins = 0
        for _ in range(num_drafts):
            pass
        return {'win_rate': wins / num_drafts}

    def evaluate_pick_accuracy(self, dataloader: DataLoader) -> Dict:
        """Evaluate pick prediction accuracy on held-out data."""
        self.model.eval()
        correct = 0
        total = 0
        top3_correct = 0

        with torch.no_grad():
            for batch in dataloader:
                pack_features = batch['pack_features'].to(self.config.device)
                pool_features = batch['pool_features'].to(self.config.device)
                pack_mask = batch['pack_mask'].to(self.config.device)
                pool_mask = batch['pool_mask'].to(self.config.device)
                pick_index = batch['pick_index'].to(self.config.device)

                logits, _ = self.model(pack_features, pool_features, pack_mask, pool_mask)

                # Top-1 accuracy
                preds = logits.argmax(dim=-1)
                correct += (preds == pick_index).sum().item()

                # Top-3 accuracy
                _, top3 = logits.topk(3, dim=-1)
                for i in range(len(pick_index)):
                    if pick_index[i] in top3[i]:
                        top3_correct += 1

                total += pick_index.shape[0]

        return {
            'top1_accuracy': correct / total,
            'top3_accuracy': top3_correct / total,
        }


# =============================================================================
# GAME SIMULATION
# =============================================================================

class GameSimulator:
    """
    Simulates draft and gameplay using Forge daemon.

    This connects to the Forge Java process for accurate game simulation.
    """

    def __init__(self, forge_host: str = "localhost", forge_port: int = 17220):
        self.forge_host = forge_host
        self.forge_port = forge_port
        self.connected = False

    def connect(self) -> bool:
        """Connect to Forge daemon."""
        try:
            import socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.forge_host, self.forge_port))
            self.connected = True
            print(f"Connected to Forge daemon at {self.forge_host}:{self.forge_port}")
            return True
        except Exception as e:
            print(f"Failed to connect to Forge daemon: {e}")
            print("Please start Forge with: java -jar forge.jar --daemon")
            return False

    def start_draft(self, set_code: str, num_players: int = 8) -> Dict:
        """Start a new draft session."""
        if not self.connected:
            return {'error': 'Not connected'}

        request = {
            'action': 'start_draft',
            'set': set_code,
            'players': num_players,
        }
        return self._send_request(request)

    def get_pack(self) -> Dict:
        """Get current pack to pick from."""
        return self._send_request({'action': 'get_pack'})

    def make_pick(self, card_index: int) -> Dict:
        """Make a pick from the current pack."""
        return self._send_request({'action': 'pick', 'index': card_index})

    def get_pool(self) -> Dict:
        """Get cards drafted so far."""
        return self._send_request({'action': 'get_pool'})

    def start_game(self, opponent_deck: List[str]) -> Dict:
        """Start a game with the drafted deck."""
        return self._send_request({
            'action': 'start_game',
            'opponent_deck': opponent_deck,
        })

    def get_game_state(self) -> Dict:
        """Get current game state."""
        return self._send_request({'action': 'get_state'})

    def take_action(self, action: Dict) -> Dict:
        """Take an action in the game."""
        return self._send_request({'action': 'take_action', 'game_action': action})

    def _send_request(self, request: Dict) -> Dict:
        """Send request to Forge daemon."""
        if not self.connected:
            return {'error': 'Not connected'}

        try:
            self.socket.send(json.dumps(request).encode() + b'\n')
            response = self.socket.recv(65536).decode()
            return json.loads(response)
        except Exception as e:
            return {'error': str(e)}

    def disconnect(self):
        """Disconnect from Forge daemon."""
        if self.connected:
            self.socket.close()
            self.connected = False


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_behavioral_cloning(config: TrainingConfig):
    """Run behavioral cloning training."""
    print("\n" + "=" * 70)
    print("BEHAVIORAL CLONING TRAINING")
    print("=" * 70)

    # Check for data
    data_dir = Path(config.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} not found")
        print("Run: python scripts/download_17lands.py --sets " + " ".join(config.sets))
        return

    # Initialize model
    print("\nInitializing model...")
    model = DraftModel(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize feature extractor
    if config.use_entity_encoder:
        from entity_encoder import EntityFeatureExtractor, EntityEncoderConfig
        feature_extractor = EntityFeatureExtractor(EntityEncoderConfig())
    else:
        feature_extractor = CardFeatureExtractor(CardEncoderConfig())

    # Check for processed data
    has_data = False
    for set_code in config.sets:
        if (data_dir / f"{set_code}_processed.json").exists():
            has_data = True
            break

    if not has_data:
        print("\nNo processed data found. Creating synthetic data for testing...")
        create_synthetic_data(data_dir, config.sets)

    # Create datasets
    print("\nLoading data...")
    try:
        dataset = DraftPickDataset(
            config.data_dir,
            config.sets,
            feature_extractor,
            max_samples=10000,  # Limit for testing
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating synthetic dataset...")
        dataset = create_synthetic_dataset(1000, feature_extractor)

    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.bc_batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging
        pin_memory=True if config.device == 'cuda' else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.bc_batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Update scheduler
    trainer = BCTrainer(config, model)
    trainer.scheduler = torch.optim.lr_scheduler.OneCycleLR(
        trainer.optimizer,
        max_lr=config.bc_lr,
        epochs=config.bc_epochs,
        steps_per_epoch=len(train_loader),
    )

    # Training loop
    best_accuracy = 0
    for epoch in range(config.bc_epochs):
        print(f"\nEpoch {epoch + 1}/{config.bc_epochs}")

        train_metrics = trainer.train_epoch(train_loader, epoch)
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")

        val_metrics = trainer.validate(val_loader)
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")

        # Save checkpoint
        if val_metrics['accuracy'] > best_accuracy:
            best_accuracy = val_metrics['accuracy']
            trainer.save_checkpoint(
                f"{config.checkpoint_dir}/bc_best.pt",
                epoch,
                val_metrics,
            )
            print(f"  New best model saved!")

    print(f"\nTraining complete. Best accuracy: {best_accuracy:.4f}")
    return model


def create_synthetic_data(data_dir: Path, sets: List[str]):
    """Create synthetic training data for testing."""
    data_dir.mkdir(parents=True, exist_ok=True)

    # Generate fake cards
    card_names = [
        "Lightning Bolt", "Counterspell", "Dark Ritual", "Giant Growth", "Healing Salve",
        "Air Elemental", "Serra Angel", "Shivan Dragon", "Force of Nature", "Lord of the Pit",
        "Llanowar Elves", "Birds of Paradise", "Sol Ring", "Black Lotus", "Time Walk",
    ]

    for set_code in sets:
        picks = []
        for _ in range(500):  # 500 picks per set
            # Generate random pack
            pack = []
            for i in range(15):
                card = {
                    'name': random.choice(card_names),
                    'mana_cost': random.choice(['{R}', '{U}', '{B}', '{G}', '{W}', '{1}{R}', '{2}{U}']),
                    'type': random.choice(['Creature', 'Instant', 'Sorcery', 'Enchantment']),
                    'power': random.randint(1, 5) if random.random() > 0.5 else None,
                    'toughness': random.randint(1, 5) if random.random() > 0.5 else None,
                    'keywords': random.sample(['Flying', 'Haste', 'Trample'], k=random.randint(0, 2)),
                    'rarity': random.choice(['Common', 'Uncommon', 'Rare']),
                }
                pack.append(card)

            # Generate random pool
            pool_size = random.randint(0, 30)
            pool = [random.choice(pack) for _ in range(pool_size)]

            picks.append({
                'pack': pack,
                'pool': pool,
                'pick_index': random.randint(0, 14),
            })

        # Save
        with open(data_dir / f"{set_code}_processed.json", 'w') as f:
            json.dump({'picks': picks}, f)

        print(f"Created synthetic data for {set_code}: {len(picks)} picks")


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing."""

    def __init__(self, num_samples: int, feature_dim: int):
        self.num_samples = num_samples
        self.feature_dim = feature_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'pack_features': np.random.randn(15, self.feature_dim).astype(np.float32),
            'pool_features': np.random.randn(45, self.feature_dim).astype(np.float32),
            'pack_mask': np.ones(15, dtype=np.float32),
            'pool_mask': np.concatenate([np.ones(random.randint(0, 30)), np.zeros(45)])[:45].astype(np.float32),
            'pick_index': random.randint(0, 14),
        }


def create_synthetic_dataset(num_samples: int, feature_extractor) -> Dataset:
    """Create a synthetic dataset for testing."""
    # Determine feature dimension
    if hasattr(feature_extractor, 'config'):
        if hasattr(feature_extractor.config, 'identity_input_dim'):
            feature_dim = 508  # EntityEncoder
        else:
            feature_dim = feature_extractor.config.input_dim
    else:
        feature_dim = 94

    return SyntheticDataset(num_samples, feature_dim)


def play_representative_games(config: TrainingConfig, model: DraftModel, num_games: int = 5):
    """Play some representative draft games to verify model behavior."""
    print("\n" + "=" * 70)
    print("PLAYING REPRESENTATIVE GAMES")
    print("=" * 70)

    # Try to connect to Forge
    simulator = GameSimulator()
    if not simulator.connect():
        print("\nForge daemon not available. Running simulated drafts instead...")
        run_simulated_drafts(model, num_games)
        return

    # Run actual drafts
    for game_num in range(num_games):
        print(f"\n--- Game {game_num + 1}/{num_games} ---")

        # Start draft
        result = simulator.start_draft("FDN")
        if 'error' in result:
            print(f"Error starting draft: {result['error']}")
            continue

        # Make picks
        picks_made = 0
        while True:
            pack = simulator.get_pack()
            if 'error' in pack or not pack.get('cards'):
                break

            # Use model to pick
            # (Simplified - would need proper feature extraction)
            pick_idx = random.randint(0, len(pack['cards']) - 1)  # Random for now
            simulator.make_pick(pick_idx)
            picks_made += 1

        pool = simulator.get_pool()
        print(f"Draft complete. Made {picks_made} picks.")
        print(f"Pool size: {len(pool.get('cards', []))}")

    simulator.disconnect()


def run_simulated_drafts(model: DraftModel, num_drafts: int = 5):
    """Run simulated drafts without Forge."""
    print("\nRunning simulated drafts (no Forge connection)...")

    model.eval()

    # Determine feature dimension from model
    if isinstance(model.encoder, SharedCardEncoder):
        feature_dim = model.encoder.config.input_dim  # 94
    else:
        feature_dim = 508  # EntityEncoder

    for draft_num in range(num_drafts):
        print(f"\n--- Simulated Draft {draft_num + 1}/{num_drafts} ---")

        pool = []
        pick_log = []

        for pack_num in range(3):  # 3 packs
            for pick_num in range(15):  # 15 picks per pack
                # Create random pack
                pack_size = 15 - pick_num

                # Generate random features (normalized like real features would be)
                pack_features = torch.rand(1, 15, feature_dim) * 0.5
                pool_features = torch.rand(1, 45, feature_dim) * 0.5
                pack_mask = torch.zeros(1, 15)
                pack_mask[:, :pack_size] = 1
                pool_mask = torch.zeros(1, 45)
                pool_mask[:, :len(pool)] = 1 if len(pool) > 0 else 0

                with torch.no_grad():
                    logits, value = model(pack_features, pool_features, pack_mask, pool_mask)

                # Apply mask
                logits = logits.masked_fill(pack_mask == 0, float('-inf'))

                # Sample pick (with temperature for variety)
                probs = F.softmax(logits / 0.5, dim=-1)
                pick = torch.argmax(probs[0]).item()  # Greedy for evaluation

                pool.append(pick)
                pick_log.append(f"P{pack_num+1}p{pick_num+1}: card {pick}")

        print(f"  Drafted {len(pool)} cards over 3 packs")
        print(f"  Value estimate: {value.item():.3f}")
        print(f"  Sample picks: {', '.join(pick_log[:5])}...")


def main():
    parser = argparse.ArgumentParser(description="MTG RL Training Pipeline")
    parser.add_argument("--mode", choices=["bc", "rl", "eval", "full", "play"],
                       default="bc", help="Training mode")
    parser.add_argument("--sets", nargs="+", default=["FDN"],
                       help="Set codes to train on")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of BC epochs")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Checkpoint to load")
    parser.add_argument("--num-games", type=int, default=5,
                       help="Number of games to play")

    args = parser.parse_args()

    # Create config
    config = TrainingConfig(
        sets=args.sets,
        bc_epochs=args.epochs,
    )

    print("MTG RL Training Pipeline")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Sets: {args.sets}")
    print(f"Device: {config.device}")
    print("=" * 70)

    if args.mode == "bc" or args.mode == "full":
        model = run_behavioral_cloning(config)

        if args.mode == "full":
            play_representative_games(config, model, args.num_games)

    elif args.mode == "play":
        # Load model
        if args.checkpoint:
            model = DraftModel(config)
            checkpoint = torch.load(args.checkpoint, map_location=config.device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Creating new model (no checkpoint provided)")
            model = DraftModel(config)

        play_representative_games(config, model, args.num_games)

    elif args.mode == "eval":
        if not args.checkpoint:
            print("Error: --checkpoint required for evaluation")
            return

        model = DraftModel(config)
        checkpoint = torch.load(args.checkpoint, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        evaluator = DraftEvaluator(model, config)
        # Run evaluations...

    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
