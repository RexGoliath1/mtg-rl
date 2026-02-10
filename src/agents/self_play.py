#!/usr/bin/env python3
"""
Self-Play Training Infrastructure for MTG RL

Implements the core self-play training loop used by AlphaStar and OpenAI Five.
Key components:
1. ModelPool - Stores past model checkpoints for opponent sampling
2. EloTracker - Tracks relative skill levels of models
3. OpponentSampler - Samples opponents based on configurable strategies
4. SelfPlayTrainer - Orchestrates training against past selves

Supports two network backends:
- "alphazero": AlphaZeroNetwork from src.training.self_play (default)
- "ppo": MTGPolicyNetwork + PPOAgent (legacy)

Self-play provides natural curriculum learning:
- Early: Easy opponents (early versions of yourself)
- Late: Hard opponents (recent improved versions)

References:
- AlphaStar: https://deepmind.google/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii/
- OpenAI Five: https://arxiv.org/abs/1912.06680
"""

import os
import json
import time
import random
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Union
from collections import deque
from pathlib import Path
from datetime import datetime
import threading
import math

import numpy as np
import torch
import torch.nn as nn

from src.models.policy_network import MTGPolicyNetwork, TransformerConfig
from src.agents.ppo_agent import PPOAgent, PPOConfig
from src.environments.daemon_environment import DaemonMTGEnvironment
from src.training.self_play import AlphaZeroNetwork

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ELO RATING SYSTEM
# =============================================================================

class EloTracker:
    """
    Tracks Elo ratings for models in the self-play pool.

    Elo provides a relative measure of skill:
    - New models start at 1200
    - Win against higher-rated -> bigger gain
    - Win against lower-rated -> smaller gain
    - K-factor determines rating volatility
    """

    def __init__(self, k_factor: float = 32.0, initial_rating: float = 1200.0):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings: Dict[str, float] = {}
        self.games_played: Dict[str, int] = {}
        self.history: List[Dict] = []

    def get_rating(self, model_id: str) -> float:
        """Get current rating for a model."""
        return self.ratings.get(model_id, self.initial_rating)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A against player B."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update_ratings(self, model_a: str, model_b: str, score_a: float):
        """
        Update ratings after a match.

        Args:
            model_a: ID of first model
            model_b: ID of second model
            score_a: Score for model A (1.0 = win, 0.5 = draw, 0.0 = loss)
        """
        # Initialize if needed
        if model_a not in self.ratings:
            self.ratings[model_a] = self.initial_rating
        if model_a not in self.games_played:
            self.games_played[model_a] = 0
        if model_b not in self.ratings:
            self.ratings[model_b] = self.initial_rating
        if model_b not in self.games_played:
            self.games_played[model_b] = 0

        rating_a = self.ratings[model_a]
        rating_b = self.ratings[model_b]

        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = 1.0 - expected_a
        score_b = 1.0 - score_a

        # Update ratings
        new_rating_a = rating_a + self.k_factor * (score_a - expected_a)
        new_rating_b = rating_b + self.k_factor * (score_b - expected_b)

        self.ratings[model_a] = new_rating_a
        self.ratings[model_b] = new_rating_b
        self.games_played[model_a] += 1
        self.games_played[model_b] += 1

        # Record history
        self.history.append({
            'timestamp': time.time(),
            'model_a': model_a,
            'model_b': model_b,
            'score_a': score_a,
            'rating_a_before': rating_a,
            'rating_b_before': rating_b,
            'rating_a_after': new_rating_a,
            'rating_b_after': new_rating_b,
        })

    def get_leaderboard(self) -> List[Tuple[str, float, int]]:
        """Get sorted leaderboard: (model_id, rating, games_played)."""
        items = [(k, v, self.games_played.get(k, 0)) for k, v in self.ratings.items()]
        return sorted(items, key=lambda x: x[1], reverse=True)

    def save(self, path: str):
        """Save ratings to file."""
        data = {
            'ratings': self.ratings,
            'games_played': self.games_played,
            'history': self.history[-1000:],  # Keep last 1000 games
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """Load ratings from file."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            self.ratings = data.get('ratings', {})
            self.games_played = data.get('games_played', {})
            self.history = data.get('history', [])


# =============================================================================
# MODEL POOL
# =============================================================================

@dataclass
class ModelCheckpoint:
    """Metadata for a model checkpoint."""
    model_id: str
    path: str
    games_trained: int
    timestamp: float
    elo_rating: float = 1200.0
    win_rate_vs_random: float = 0.0
    network_type: str = "ppo"
    metadata: Dict = field(default_factory=dict)


class ModelPool:
    """
    Pool of model checkpoints for self-play training.

    Stores past versions of the model that can be used as opponents.
    Supports different sampling strategies:
    - recent: Prefer recent models
    - historical: Uniform over all models
    - elo_weighted: Prefer similar-skill models
    """

    def __init__(self,
                 pool_dir: str,
                 max_pool_size: int = 100,
                 device: torch.device = None):
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.max_pool_size = max_pool_size
        self.device = device or torch.device('cpu')

        self.checkpoints: List[ModelCheckpoint] = []
        self.elo_tracker = EloTracker()
        self._lock = threading.Lock()

        # Load existing checkpoints
        self._load_pool_state()

    def _load_pool_state(self):
        """Load pool state from disk."""
        state_path = self.pool_dir / 'pool_state.json'
        if state_path.exists():
            with open(state_path, 'r') as f:
                data = json.load(f)
            self.checkpoints = [
                ModelCheckpoint(**cp) for cp in data.get('checkpoints', [])
            ]

        elo_path = self.pool_dir / 'elo_ratings.json'
        self.elo_tracker.load(str(elo_path))

    def _save_pool_state(self):
        """Save pool state to disk."""
        state_path = self.pool_dir / 'pool_state.json'
        data = {
            'checkpoints': [asdict(cp) for cp in self.checkpoints],
        }
        with open(state_path, 'w') as f:
            json.dump(data, f, indent=2)

        elo_path = self.pool_dir / 'elo_ratings.json'
        self.elo_tracker.save(str(elo_path))

    def add_checkpoint(self,
                       model: nn.Module,
                       games_trained: int,
                       config: Union[TransformerConfig, None] = None,
                       network_type: str = "ppo",
                       metadata: Dict = None) -> str:
        """
        Add a new model checkpoint to the pool.

        Args:
            model: The network to checkpoint (MTGPolicyNetwork or AlphaZeroNetwork)
            games_trained: Number of games trained on
            config: Transformer configuration (used for PPO path; ignored for alphazero)
            network_type: "alphazero" or "ppo"
            metadata: Additional metadata to store

        Returns:
            model_id: Unique identifier for this checkpoint
        """
        with self._lock:
            # Generate unique ID
            timestamp = time.time()
            model_id = f"model_{games_trained}_{int(timestamp)}"

            # Save model weights
            model_path = self.pool_dir / f"{model_id}.pt"

            if network_type == "alphazero":
                # Use AlphaZeroNetwork's save format
                torch.save({
                    'state_dict': model.state_dict(),
                    'encoder_config': getattr(model, 'encoder_config', None),
                    'network_type': 'alphazero',
                    'games_trained': games_trained,
                }, model_path)
            else:
                # Legacy PPO format
                torch.save({
                    'state_dict': model.state_dict(),
                    'config': config,
                    'network_type': 'ppo',
                    'games_trained': games_trained,
                }, model_path)

            # Create checkpoint entry
            checkpoint = ModelCheckpoint(
                model_id=model_id,
                path=str(model_path),
                games_trained=games_trained,
                timestamp=timestamp,
                elo_rating=self.elo_tracker.get_rating(model_id),
                network_type=network_type,
                metadata=metadata or {}
            )

            self.checkpoints.append(checkpoint)

            # Prune old checkpoints if needed
            if len(self.checkpoints) > self.max_pool_size:
                self._prune_pool()

            self._save_pool_state()

            logger.info(f"Added checkpoint {model_id} ({network_type}) to pool (total: {len(self.checkpoints)})")
            return model_id

    def _prune_pool(self):
        """Remove old checkpoints to stay within max_pool_size."""
        if len(self.checkpoints) <= self.max_pool_size:
            return

        # Keep: most recent, highest elo, evenly spaced by games
        keep_indices = set()

        # Keep most recent (half of max_pool_size)
        recent_keep = max(1, self.max_pool_size // 2)
        for i in range(min(recent_keep, len(self.checkpoints))):
            keep_indices.add(len(self.checkpoints) - 1 - i)

        # Keep top by Elo (remaining budget)
        elo_keep = max(1, self.max_pool_size - len(keep_indices))
        elo_sorted = sorted(range(len(self.checkpoints)),
                          key=lambda i: self.checkpoints[i].elo_rating,
                          reverse=True)
        for i in elo_sorted[:elo_keep]:
            keep_indices.add(i)

        # Keep evenly spaced by games trained (only for larger pools)
        if len(self.checkpoints) > self.max_pool_size * 2:
            games_sorted = sorted(range(len(self.checkpoints)),
                                key=lambda i: self.checkpoints[i].games_trained)
            step = len(games_sorted) // (self.max_pool_size - len(keep_indices))
            for i in range(0, len(games_sorted), max(1, step)):
                keep_indices.add(games_sorted[i])

        # Remove checkpoints not in keep set
        to_remove = []
        for i, cp in enumerate(self.checkpoints):
            if i not in keep_indices and len(keep_indices) + len(to_remove) < len(self.checkpoints):
                to_remove.append(i)
                # Delete file
                try:
                    os.remove(cp.path)
                except Exception:
                    pass

        self.checkpoints = [cp for i, cp in enumerate(self.checkpoints) if i not in to_remove]

    def sample_opponent(self,
                        current_model_id: Optional[str] = None,
                        strategy: str = 'recent') -> Optional[ModelCheckpoint]:
        """
        Sample an opponent from the pool.

        Args:
            current_model_id: ID of current model (to avoid self-play)
            strategy: Sampling strategy
                - 'recent': 80% from last 10, 20% historical
                - 'uniform': Uniform random
                - 'elo_matched': Prefer similar Elo rating
                - 'curriculum': Early models first, then progress

        Returns:
            ModelCheckpoint or None if pool is empty
        """
        if not self.checkpoints:
            return None

        # Filter out current model
        candidates = [cp for cp in self.checkpoints
                     if cp.model_id != current_model_id]

        if not candidates:
            return self.checkpoints[-1] if self.checkpoints else None

        if strategy == 'recent':
            # 80% chance to pick from last 10 models
            if random.random() < 0.8 and len(candidates) > 10:
                return random.choice(candidates[-10:])
            return random.choice(candidates)

        elif strategy == 'uniform':
            return random.choice(candidates)

        elif strategy == 'elo_matched':
            if current_model_id:
                current_elo = self.elo_tracker.get_rating(current_model_id)
            else:
                current_elo = 1200.0

            # Weight by similarity to current Elo
            weights = []
            for cp in candidates:
                diff = abs(cp.elo_rating - current_elo)
                weight = math.exp(-diff / 200.0)  # Gaussian-like weighting
                weights.append(weight)

            total = sum(weights)
            weights = [w / total for w in weights]
            return random.choices(candidates, weights=weights)[0]

        elif strategy == 'curriculum':
            # Prefer earlier models at start, later models as training progresses
            # Weighted by games_trained (earlier = lower)
            weights = [1.0 / (1.0 + cp.games_trained / 10000.0) for cp in candidates]
            total = sum(weights)
            weights = [w / total for w in weights]
            return random.choices(candidates, weights=weights)[0]

        return random.choice(candidates)

    def load_model(self,
                   checkpoint: ModelCheckpoint,
                   config: Optional[TransformerConfig] = None,
                   device: Optional[torch.device] = None) -> nn.Module:
        """
        Load a model from checkpoint. Dispatches based on network_type.

        Args:
            checkpoint: The checkpoint metadata
            config: TransformerConfig (required for PPO, ignored for alphazero)
            device: Device to load model onto

        Returns:
            Loaded model (AlphaZeroNetwork or MTGPolicyNetwork)
        """
        load_device = device or self.device
        checkpoint_data = torch.load(checkpoint.path, map_location=load_device, weights_only=False)

        net_type = checkpoint_data.get('network_type', checkpoint.network_type)

        if net_type == "alphazero":
            encoder_config = checkpoint_data.get('encoder_config', None)
            network = AlphaZeroNetwork(encoder_config=encoder_config).to(load_device)
            network.load_state_dict(checkpoint_data['state_dict'])
            network.eval()
            return network
        else:
            # Legacy PPO path
            if config is None:
                config = checkpoint_data.get('config', TransformerConfig())
            model = MTGPolicyNetwork(config).to(load_device)
            model.load_state_dict(checkpoint_data['state_dict'])
            model.eval()
            return model

    def get_latest(self) -> Optional[ModelCheckpoint]:
        """Get the most recent checkpoint."""
        return self.checkpoints[-1] if self.checkpoints else None

    def get_best_by_elo(self) -> Optional[ModelCheckpoint]:
        """Get the highest-rated checkpoint."""
        if not self.checkpoints:
            return None
        return max(self.checkpoints, key=lambda cp: cp.elo_rating)

    def __len__(self):
        return len(self.checkpoints)


# =============================================================================
# ALPHAZERO AGENT ADAPTER
# =============================================================================

class AlphaZeroAgent:
    """
    Adapter that wraps AlphaZeroNetwork to match the PPOAgent.get_action() interface.

    This allows SelfPlayGame to use AlphaZeroNetwork without changing its game loop.
    The DaemonMTGEnvironment provides integer action indices and numpy observations,
    so we encode the game state through ForgeGameStateEncoder and sample from the
    policy head to produce an action index compatible with env.step().
    """

    def __init__(self, network: AlphaZeroNetwork, device: torch.device = None):
        self.network = network
        self.device = device or next(network.parameters()).device
        self.encoder = network.encoder

    @torch.no_grad()
    def get_action(
        self,
        game_state,
        action_mask: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, Optional[Dict]]:
        """
        Select an action given the game state and action mask.

        This mirrors PPOAgent.get_action() so SelfPlayGame.play() works unchanged.

        Args:
            game_state: GameState object from DaemonMTGEnvironment (has .to_observation())
            action_mask: numpy array of valid actions (1=valid, 0=invalid)
            deterministic: If True, pick the argmax action

        Returns:
            action: Selected action index
            info: Dict with value estimate (or None)
        """
        self.network.eval()

        # Encode the observation through the state encoder
        # game_state is a GameState from rl_environment with .to_observation()
        obs = game_state.to_observation()
        state_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Get policy logits and value from AlphaZero heads
        policy_logits = self.network.policy_head(state_tensor, return_logits=True)
        value = self.network.value_head(state_tensor)

        # Apply action mask: set invalid actions to -inf before softmax
        mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Trim or pad policy logits to match action_mask size
        num_actions = mask_tensor.shape[1]
        policy_dim = policy_logits.shape[1]

        if policy_dim < num_actions:
            # Pad policy logits with -inf for extra actions
            padding = torch.full(
                (1, num_actions - policy_dim), float('-inf'), device=self.device
            )
            policy_logits = torch.cat([policy_logits, padding], dim=1)
        elif policy_dim > num_actions:
            # Truncate to match environment action space
            policy_logits = policy_logits[:, :num_actions]

        # Mask invalid actions
        policy_logits = policy_logits.masked_fill(mask_tensor == 0, float('-inf'))

        # Check if any valid actions exist
        if (mask_tensor.sum() == 0):
            return -1, None

        # Softmax to get probabilities
        probs = torch.softmax(policy_logits, dim=-1)

        if deterministic:
            action = probs.argmax(dim=-1).item()
        else:
            action = torch.multinomial(probs, 1).item()

        return action, {
            'value': value,
            'action_probs': probs,
        }


# =============================================================================
# SELF-PLAY GAME RUNNER
# =============================================================================

@dataclass
class SelfPlayConfig:
    """Configuration for self-play training."""
    # Environment settings
    daemon_host: str = 'localhost'
    daemon_port: int = 17171
    deck1: str = 'test_red.dck'
    deck2: str = 'test_blue.dck'
    game_timeout: int = 120

    # Training settings
    total_games: int = 1_000_000
    games_per_checkpoint: int = 1000  # Add checkpoint every N games
    games_per_evaluation: int = 5000  # Evaluate vs baseline every N games

    # Self-play settings
    opponent_sampling_strategy: str = 'recent'  # recent, uniform, elo_matched
    self_play_ratio: float = 0.8  # Ratio of games against self vs pool

    # PPO settings (used when network_type="ppo")
    n_steps: int = 128
    batch_size: int = 64
    n_epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5

    # Network settings (used when network_type="ppo")
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4

    # Pool settings
    pool_dir: str = 'model_pool'
    max_pool_size: int = 100

    # Parallelism
    n_parallel_games: int = 4

    # Network type: "alphazero" (default) or "ppo" (legacy)
    network_type: str = "alphazero"


class SelfPlayGame:
    """
    Manages a single self-play game between two models.

    Both players can be the same model (true self-play) or
    different models (playing against historical versions).

    Accepts any agent that implements get_action(game_state, action_mask)
    returning (action_idx, info_dict_or_none).
    """

    def __init__(self,
                 env: DaemonMTGEnvironment,
                 player1_agent,
                 player2_agent,
                 player1_id: str,
                 player2_id: str,
                 collect_transitions: bool = True):
        self.env = env
        self.player1_agent = player1_agent
        self.player2_agent = player2_agent
        self.player1_id = player1_id
        self.player2_id = player2_id
        self.collect_transitions = collect_transitions

        self.transitions: List[Dict] = []

    def play(self) -> Dict:
        """
        Play a complete game and return results.

        Returns:
            Dict with: winner, player1_won, turns, transitions (if collected)
        """
        try:
            obs, info = self.env.reset()
        except Exception as e:
            logger.warning(f"Failed to reset game: {e}")
            return {'error': str(e), 'player1_won': False}

        game_transitions = []
        turn_count = 0

        while not self.env.game_over and turn_count < 1000:
            action_mask = self.env.get_action_mask()

            # Both agents use player1's perspective (Agent(1))
            # In self-play, we always train from player 1's perspective
            if self.env.current_state:
                action, action_info = self.player1_agent.get_action(
                    self.env.current_state, action_mask
                )
            else:
                action = -1
                action_info = None

            # Store transition for training
            if self.collect_transitions and action_info:
                game_transitions.append({
                    'action_info': action_info,
                    'action': action,
                    'state': self.env.current_state,
                })

            # Take action
            obs, reward, done, truncated, info = self.env.step(action)

            # Update last transition with reward
            if game_transitions and self.collect_transitions:
                game_transitions[-1]['reward'] = reward
                game_transitions[-1]['done'] = done

            turn_count += 1

            if done:
                break

        # Determine winner
        player1_won = self.env.won

        return {
            'player1_won': player1_won,
            'player1_id': self.player1_id,
            'player2_id': self.player2_id,
            'turns': turn_count,
            'transitions': game_transitions if self.collect_transitions else [],
        }


# =============================================================================
# SELF-PLAY TRAINER
# =============================================================================

class SelfPlayTrainer:
    """
    Main self-play training loop.

    Orchestrates:
    1. Model pool management
    2. Opponent sampling
    3. Game execution
    4. Training updates (PPO for "ppo" mode, placeholder for "alphazero")
    5. Elo tracking
    6. Checkpointing

    Supports two network backends via config.network_type:
    - "alphazero": AlphaZeroNetwork + AlphaZeroAgent (policy/value heads, MCTS-ready)
    - "ppo": MTGPolicyNetwork + PPOAgent (legacy transformer + PPO training)
    """

    def __init__(self, config: SelfPlayConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Setup directories
        self.run_dir = Path(f"runs/selfplay_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.run_dir / 'config.json', 'w') as f:
            json.dump(asdict(config), f, indent=2)

        # Initialize model pool
        pool_dir = self.run_dir / config.pool_dir
        self.model_pool = ModelPool(str(pool_dir), config.max_pool_size, self.device)

        if config.network_type == "alphazero":
            self._init_alphazero()
        else:
            self._init_ppo()

        # Metrics
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.recent_results = deque(maxlen=100)
        self.training_metrics: List[Dict] = []

        # Current model ID
        self.current_model_id = "initial"

        logger.info("SelfPlayTrainer initialized")
        logger.info(f"  Network type: {config.network_type}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Run dir: {self.run_dir}")
        logger.info(f"  Pool size: {len(self.model_pool)}")

    def _init_alphazero(self):
        """Initialize AlphaZero network and agent."""
        self.network = AlphaZeroNetwork(num_players=2).to(self.device)
        self.agent = AlphaZeroAgent(self.network, device=self.device)

        # Keep these as None since PPO-specific
        self.ppo_config = None
        self.transformer_config = None

        param_count = sum(p.numel() for p in self.network.parameters())
        logger.info(f"  AlphaZeroNetwork params: {param_count:,}")

    def _init_ppo(self):
        """Initialize PPO agent (legacy path)."""
        self.network = None  # PPOAgent owns the network

        self.ppo_config = PPOConfig(
            learning_rate=self.config.learning_rate,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_epsilon=self.config.clip_epsilon,
            n_steps=self.config.n_steps,
            n_epochs=self.config.n_epochs,
            batch_size=self.config.batch_size,
            entropy_coef=self.config.entropy_coef,
            value_coef=self.config.value_coef,
        )

        self.transformer_config = TransformerConfig(
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
        )

        self.agent = PPOAgent(
            ppo_config=self.ppo_config,
            transformer_config=self.transformer_config,
            device=self.device
        )

    def _create_environment(self) -> DaemonMTGEnvironment:
        """Create a new game environment."""
        return DaemonMTGEnvironment(
            host=self.config.daemon_host,
            port=self.config.daemon_port,
            deck1=self.config.deck1,
            deck2=self.config.deck2,
            timeout=self.config.game_timeout,
        )

    def _select_opponent(self):
        """
        Select an opponent for self-play.

        Returns:
            (opponent_agent, opponent_id)
        """
        # Decide: play against self or pool
        if random.random() < self.config.self_play_ratio or len(self.model_pool) == 0:
            # True self-play: current model vs itself
            return self.agent, self.current_model_id

        # Sample from pool
        opponent_checkpoint = self.model_pool.sample_opponent(
            current_model_id=self.current_model_id,
            strategy=self.config.opponent_sampling_strategy
        )

        if opponent_checkpoint is None:
            return self.agent, self.current_model_id

        # Load opponent model (dispatches by network_type)
        opponent_model = self.model_pool.load_model(
            opponent_checkpoint, self.transformer_config
        )

        # Wrap in appropriate agent
        if opponent_checkpoint.network_type == "alphazero":
            opponent_agent = AlphaZeroAgent(opponent_model, device=self.device)
        else:
            # Legacy PPO path
            opponent_agent = PPOAgent(
                ppo_config=self.ppo_config,
                transformer_config=self.transformer_config,
                device=self.device
            )
            opponent_agent.policy = opponent_model

        return opponent_agent, opponent_checkpoint.model_id

    def _play_training_game(self) -> Dict:
        """Play a single training game and collect transitions."""
        env = self._create_environment()
        opponent_agent, opponent_id = self._select_opponent()

        try:
            game = SelfPlayGame(
                env=env,
                player1_agent=self.agent,
                player2_agent=opponent_agent,
                player1_id=self.current_model_id,
                player2_id=opponent_id,
                collect_transitions=True
            )

            result = game.play()

            # Update Elo ratings
            score = 1.0 if result['player1_won'] else 0.0
            self.model_pool.elo_tracker.update_ratings(
                self.current_model_id, opponent_id, score
            )

            return result

        finally:
            env._cleanup()

    def _store_transitions(self, transitions: List[Dict]):
        """Store transitions in the PPO buffer (PPO mode only)."""
        if self.config.network_type != "ppo":
            return

        for t in transitions:
            if 'action_info' in t and t['action_info']:
                self.agent.store_transition(
                    t['action_info'],
                    t['action'],
                    t.get('reward', 0.0),
                    t.get('done', False)
                )

    def _training_step(self) -> Dict:
        """Perform a training step. Dispatches by network_type."""
        if self.config.network_type == "ppo":
            # PPO training step
            last_value = torch.tensor([0.0], device=self.device)
            metrics = self.agent.train_step(last_value)
            return metrics
        else:
            # AlphaZero: training is done via the Learner in src.training.self_play
            # In the self-play loop we just collect games; training happens externally
            # or could be wired in here when the replay buffer is ready.
            return {}

    def _checkpoint(self):
        """Save model checkpoint to pool."""
        if self.config.network_type == "alphazero":
            model_id = self.model_pool.add_checkpoint(
                model=self.network,
                games_trained=self.games_played,
                network_type="alphazero",
                metadata={
                    'wins': self.wins,
                    'losses': self.losses,
                    'recent_win_rate': sum(self.recent_results) / max(1, len(self.recent_results)),
                }
            )
            # Also save main model using AlphaZeroNetwork's format
            self.network.save(str(self.run_dir / 'current_model.pt'))
        else:
            model_id = self.model_pool.add_checkpoint(
                model=self.agent.policy,
                games_trained=self.games_played,
                config=self.transformer_config,
                network_type="ppo",
                metadata={
                    'wins': self.wins,
                    'losses': self.losses,
                    'recent_win_rate': sum(self.recent_results) / max(1, len(self.recent_results)),
                }
            )
            # Also save main model
            self.agent.save(str(self.run_dir / 'current_model.pt'))

        self.current_model_id = model_id

    def _log_progress(self, force: bool = False):
        """Log training progress."""
        if not force and self.games_played % 100 != 0:
            return

        win_rate = sum(self.recent_results) / max(1, len(self.recent_results))
        current_elo = self.model_pool.elo_tracker.get_rating(self.current_model_id)

        logger.info(
            f"[Games: {self.games_played:,}] "
            f"Win rate (100): {win_rate*100:.1f}% | "
            f"Elo: {current_elo:.0f} | "
            f"Pool size: {len(self.model_pool)}"
        )

        # Save metrics
        self.training_metrics.append({
            'games': self.games_played,
            'timestamp': time.time(),
            'win_rate': win_rate,
            'elo': current_elo,
            'pool_size': len(self.model_pool),
        })

    def train(self, resume: bool = False):
        """
        Main training loop.

        Args:
            resume: If True, try to resume from last checkpoint
        """
        logger.info("=" * 60)
        logger.info("SELF-PLAY TRAINING")
        logger.info("=" * 60)
        logger.info(f"Network type: {self.config.network_type}")
        logger.info(f"Total games: {self.config.total_games:,}")
        logger.info(f"Games per checkpoint: {self.config.games_per_checkpoint}")
        logger.info(f"Self-play ratio: {self.config.self_play_ratio}")
        logger.info("=" * 60)

        if resume:
            model_path = self.run_dir / 'current_model.pt'
            if model_path.exists():
                if self.config.network_type == "alphazero":
                    loaded = AlphaZeroNetwork.load(str(model_path), device=str(self.device))
                    self.network.load_state_dict(loaded.state_dict())
                    logger.info(f"Resumed AlphaZero model from {model_path}")
                else:
                    self.agent.load(str(model_path))
                    logger.info(f"Resumed PPO model from {model_path}")

        # Add initial model to pool
        if len(self.model_pool) == 0:
            self._checkpoint()

        start_time = time.time()
        games_since_checkpoint = 0
        consecutive_errors = 0
        max_consecutive_errors = 20

        try:
            while self.games_played < self.config.total_games:
                # Play a game
                result = self._play_training_game()

                if 'error' in result:
                    consecutive_errors += 1
                    logger.warning(f"Game error ({consecutive_errors}/{max_consecutive_errors}): {result['error']}")
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"Too many consecutive errors ({consecutive_errors}), stopping training")
                        break
                    continue

                consecutive_errors = 0

                self.games_played += 1
                games_since_checkpoint += 1

                # Track results
                won = result['player1_won']
                if won:
                    self.wins += 1
                else:
                    self.losses += 1
                self.recent_results.append(1.0 if won else 0.0)

                # Store transitions for training (PPO mode only)
                if result.get('transitions'):
                    self._store_transitions(result['transitions'])

                # Training step when buffer is full (PPO mode)
                if self.config.network_type == "ppo" and self.agent.buffer.ptr >= self.config.n_steps:
                    self._training_step()
                    self.agent.num_updates += 1

                # Checkpoint periodically
                if games_since_checkpoint >= self.config.games_per_checkpoint:
                    self._checkpoint()
                    games_since_checkpoint = 0

                # Log progress
                self._log_progress()

        except KeyboardInterrupt:
            logger.info("\nTraining interrupted by user")

        finally:
            # Final checkpoint
            self._checkpoint()

            # Save final metrics
            with open(self.run_dir / 'training_metrics.json', 'w') as f:
                json.dump(self.training_metrics, f, indent=2)

            elapsed = time.time() - start_time
            logger.info("\n" + "=" * 60)
            logger.info("TRAINING COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Total games: {self.games_played:,}")
            logger.info(f"Total time: {elapsed/3600:.1f} hours")
            logger.info(f"Games/hour: {self.games_played / (elapsed/3600):.0f}")
            logger.info(f"Final win rate: {sum(self.recent_results)/max(1,len(self.recent_results))*100:.1f}%")
            logger.info(f"Final Elo: {self.model_pool.elo_tracker.get_rating(self.current_model_id):.0f}")

            # Print leaderboard
            logger.info("\nLeaderboard:")
            for model_id, elo, games in self.model_pool.elo_tracker.get_leaderboard()[:10]:
                logger.info(f"  {model_id}: {elo:.0f} ({games} games)")


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_against_random(agent,
                           config: SelfPlayConfig,
                           n_games: int = 100) -> float:
    """
    Evaluate agent against random baseline.

    Accepts any agent with get_action(game_state, action_mask, deterministic=True).

    Returns win rate.
    """
    wins = 0

    for i in range(n_games):
        env = DaemonMTGEnvironment(
            host=config.daemon_host,
            port=config.daemon_port,
            deck1=config.deck1,
            deck2=config.deck2,
        )

        try:
            obs, info = env.reset()

            while not env.game_over:
                action_mask = env.get_action_mask()

                # Agent action
                if env.current_state:
                    action, _ = agent.get_action(env.current_state, action_mask, deterministic=True)
                else:
                    action = -1

                obs, reward, done, truncated, info = env.step(action)

                if done:
                    break

            if env.won:
                wins += 1

        finally:
            env._cleanup()

    return wins / n_games


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Self-Play Training for MTG RL")
    parser.add_argument("--games", type=int, default=100000, help="Total games to play")
    parser.add_argument("--checkpoint-interval", type=int, default=1000, help="Games between checkpoints")
    parser.add_argument("--host", type=str, default="localhost", help="Daemon host")
    parser.add_argument("--port", type=int, default=17171, help="Daemon port")
    parser.add_argument("--deck1", type=str, default="test_red.dck", help="First deck")
    parser.add_argument("--deck2", type=str, default="test_blue.dck", help="Second deck")
    parser.add_argument("--strategy", type=str, default="recent",
                       choices=["recent", "uniform", "elo_matched"],
                       help="Opponent sampling strategy")
    parser.add_argument("--network-type", type=str, default="alphazero",
                       choices=["alphazero", "ppo"],
                       help="Network backend: alphazero (default) or ppo (legacy)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")

    args = parser.parse_args()

    config = SelfPlayConfig(
        daemon_host=args.host,
        daemon_port=args.port,
        deck1=args.deck1,
        deck2=args.deck2,
        total_games=args.games,
        games_per_checkpoint=args.checkpoint_interval,
        opponent_sampling_strategy=args.strategy,
        network_type=args.network_type,
    )

    trainer = SelfPlayTrainer(config)
    trainer.train(resume=args.resume)


if __name__ == "__main__":
    main()
