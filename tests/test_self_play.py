#!/usr/bin/env python3
"""
Regression Tests for Self-Play Training Infrastructure

Tests:
1. Turn tracking across games
2. Error persistence and recovery
3. Model pool operations
4. Elo rating system
5. Game state serialization
6. Multi-agent coordination

Run with: pytest tests/test_self_play.py -v
"""

import os
import sys
import time
import tempfile
import shutil
from pathlib import Path

import pytest
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from self_play import (
    EloTracker, ModelPool, ModelCheckpoint,
    SelfPlayConfig, SelfPlayTrainer
)
from src.models.policy_network import MTGPolicyNetwork, TransformerConfig
from src.agents.ppo_agent import PPOConfig
from src.environments.rl_environment import GameState, CardState


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def transformer_config():
    """Create a small transformer config for testing."""
    return TransformerConfig(
        d_model=64,
        n_heads=4,
        n_layers=2,
        card_embedding_dim=92,
    )


@pytest.fixture
def ppo_config():
    """Create PPO config for testing."""
    return PPOConfig(
        learning_rate=3e-4,
        n_steps=16,
        batch_size=8,
    )


@pytest.fixture
def small_model(transformer_config):
    """Create a small model for testing."""
    return MTGPolicyNetwork(transformer_config)


@pytest.fixture
def elo_tracker():
    """Create an Elo tracker for testing."""
    return EloTracker(k_factor=32.0, initial_rating=1200.0)


@pytest.fixture
def model_pool(temp_dir, transformer_config):
    """Create a model pool for testing."""
    return ModelPool(
        pool_dir=temp_dir,
        max_pool_size=10,
        device=torch.device('cpu')
    )


# =============================================================================
# ELO TRACKER TESTS
# =============================================================================

class TestEloTracker:
    """Tests for the Elo rating system."""

    def test_initial_rating(self, elo_tracker):
        """New models should start at initial rating."""
        assert elo_tracker.get_rating("new_model") == 1200.0

    def test_expected_score_equal_ratings(self, elo_tracker):
        """Equal ratings should give 0.5 expected score."""
        score = elo_tracker.expected_score(1200.0, 1200.0)
        assert abs(score - 0.5) < 0.001

    def test_expected_score_higher_rating(self, elo_tracker):
        """Higher-rated player should have expected score > 0.5."""
        score = elo_tracker.expected_score(1400.0, 1200.0)
        assert score > 0.5

    def test_win_increases_rating(self, elo_tracker):
        """Winning should increase rating."""
        initial = elo_tracker.get_rating("model_a")
        elo_tracker.update_ratings("model_a", "model_b", 1.0)
        assert elo_tracker.get_rating("model_a") > initial

    def test_loss_decreases_rating(self, elo_tracker):
        """Losing should decrease rating."""
        initial = elo_tracker.get_rating("model_a")
        elo_tracker.update_ratings("model_a", "model_b", 0.0)
        assert elo_tracker.get_rating("model_a") < initial

    def test_draw_small_change(self, elo_tracker):
        """Draw between equal opponents should result in minimal change."""
        elo_tracker.update_ratings("model_a", "model_b", 0.5)
        # Both should be close to initial
        assert abs(elo_tracker.get_rating("model_a") - 1200.0) < 1.0

    def test_upset_win_bigger_gain(self, elo_tracker):
        """Win against higher-rated opponent should give bigger gain."""
        # Set up different ratings
        elo_tracker.ratings["weak"] = 1000.0
        elo_tracker.ratings["strong"] = 1400.0

        weak_before = elo_tracker.get_rating("weak")
        elo_tracker.update_ratings("weak", "strong", 1.0)
        weak_gain = elo_tracker.get_rating("weak") - weak_before

        # Reset and test against equal opponent
        elo_tracker.ratings["weak2"] = 1000.0
        elo_tracker.ratings["equal"] = 1000.0
        weak2_before = elo_tracker.get_rating("weak2")
        elo_tracker.update_ratings("weak2", "equal", 1.0)
        equal_gain = elo_tracker.get_rating("weak2") - weak2_before

        assert weak_gain > equal_gain

    def test_save_and_load(self, elo_tracker, temp_dir):
        """Ratings should persist across save/load."""
        elo_tracker.update_ratings("model_a", "model_b", 1.0)
        elo_tracker.update_ratings("model_a", "model_c", 0.5)

        path = os.path.join(temp_dir, "elo.json")
        elo_tracker.save(path)

        new_tracker = EloTracker()
        new_tracker.load(path)

        assert new_tracker.get_rating("model_a") == elo_tracker.get_rating("model_a")
        assert new_tracker.games_played["model_a"] == 2

    def test_leaderboard_sorted(self, elo_tracker):
        """Leaderboard should be sorted by rating descending."""
        elo_tracker.ratings["best"] = 1500.0
        elo_tracker.ratings["worst"] = 900.0
        elo_tracker.ratings["mid"] = 1200.0

        leaderboard = elo_tracker.get_leaderboard()
        ratings = [r for _, r, _ in leaderboard]
        assert ratings == sorted(ratings, reverse=True)


# =============================================================================
# MODEL POOL TESTS
# =============================================================================

class TestModelPool:
    """Tests for the model checkpoint pool."""

    def test_empty_pool(self, model_pool):
        """Empty pool should have length 0."""
        assert len(model_pool) == 0

    def test_add_checkpoint(self, model_pool, small_model, transformer_config):
        """Adding checkpoint should increase pool size."""
        model_id = model_pool.add_checkpoint(
            model=small_model,
            games_trained=1000,
            config=transformer_config,
        )
        assert len(model_pool) == 1
        assert model_id is not None

    def test_checkpoint_persistence(self, model_pool, small_model, transformer_config, temp_dir):
        """Checkpoints should persist to disk."""
        _ = model_pool.add_checkpoint(
            model=small_model,
            games_trained=1000,
            config=transformer_config,
        )

        # Check file exists
        checkpoint_files = list(Path(temp_dir).glob("*.pt"))
        assert len(checkpoint_files) == 1

    def test_load_model(self, model_pool, small_model, transformer_config):
        """Should be able to load model from checkpoint."""
        model_pool.add_checkpoint(
            model=small_model,
            games_trained=1000,
            config=transformer_config,
        )

        checkpoint = model_pool.get_latest()
        loaded_model = model_pool.load_model(checkpoint, transformer_config)

        assert loaded_model is not None
        # Check weights match
        for p1, p2 in zip(small_model.parameters(), loaded_model.parameters()):
            assert torch.allclose(p1, p2)

    def test_sample_opponent_empty_pool(self, model_pool):
        """Sampling from empty pool should return None."""
        opponent = model_pool.sample_opponent()
        assert opponent is None

    def test_sample_opponent_single_model(self, model_pool, small_model, transformer_config):
        """With one model, should always return that model."""
        model_pool.add_checkpoint(small_model, 1000, transformer_config)

        opponent = model_pool.sample_opponent(strategy='uniform')
        assert opponent is not None

    def test_sample_excludes_current(self, model_pool, small_model, transformer_config):
        """Should exclude current model from sampling when there are others."""
        _ = model_pool.add_checkpoint(small_model, 1000, transformer_config)
        id2 = model_pool.add_checkpoint(small_model, 2000, transformer_config)

        # Sample excluding id2
        for _ in range(10):
            opponent = model_pool.sample_opponent(current_model_id=id2, strategy='uniform')
            if opponent:
                assert opponent.model_id != id2

    def test_pool_pruning(self, temp_dir, small_model, transformer_config):
        """Pool should prune when exceeding max size."""
        pool = ModelPool(temp_dir, max_pool_size=5)

        for i in range(10):
            pool.add_checkpoint(small_model, i * 1000, transformer_config)

        assert len(pool) <= 5

    def test_get_latest(self, model_pool, small_model, transformer_config):
        """Should return most recently added checkpoint."""
        model_pool.add_checkpoint(small_model, 1000, transformer_config)
        time.sleep(0.01)  # Ensure different timestamps
        model_pool.add_checkpoint(small_model, 2000, transformer_config)

        latest = model_pool.get_latest()
        assert latest.games_trained == 2000


# =============================================================================
# TURN TRACKING TESTS
# =============================================================================

class TestTurnTracking:
    """Tests for turn tracking during games."""

    def test_game_state_turn_parsing(self):
        """Game state should correctly parse turn number."""
        data = {
            'turn': 5,
            'phase': 'MAIN1',
            'game_state': {
                'active_player': 'Agent(1)',
                'priority_player': 'Agent(1)',
                'is_game_over': False,
                'players': [
                    {'name': 'Agent(1)', 'life': 20},
                    {'name': 'Agent(2)', 'life': 18},
                ],
                'stack': [],
            }
        }

        state = GameState.from_dict(data, 'Agent(1)')
        assert state.turn == 5

    def test_game_state_phase_tracking(self):
        """Game state should track phase correctly."""
        phases = ['UNTAP', 'UPKEEP', 'DRAW', 'MAIN1', 'COMBAT_BEGIN', 'MAIN2', 'END_OF_TURN']

        for phase in phases:
            data = {
                'turn': 1,
                'phase': phase,
                'game_state': {
                    'active_player': 'Agent(1)',
                    'priority_player': 'Agent(1)',
                    'is_game_over': False,
                    'players': [{'name': 'Agent(1)', 'life': 20}],
                    'stack': [],
                }
            }

            state = GameState.from_dict(data, 'Agent(1)')
            assert state.phase == phase

    def test_turn_increments_correctly(self):
        """Turn number should increment across game states."""
        turns = []

        for turn_num in [1, 2, 3, 4, 5]:
            data = {
                'turn': turn_num,
                'phase': 'MAIN1',
                'game_state': {
                    'active_player': 'Agent(1)',
                    'priority_player': 'Agent(1)',
                    'is_game_over': False,
                    'players': [{'name': 'Agent(1)', 'life': 20}],
                    'stack': [],
                }
            }
            state = GameState.from_dict(data, 'Agent(1)')
            turns.append(state.turn)

        assert turns == [1, 2, 3, 4, 5]


# =============================================================================
# ERROR PERSISTENCE TESTS
# =============================================================================

class TestErrorPersistence:
    """Tests for error handling and recovery."""

    def test_checkpoint_survives_crash(self, temp_dir, small_model, transformer_config):
        """Checkpoint should survive simulated crash."""
        pool = ModelPool(temp_dir, max_pool_size=10)
        model_id = pool.add_checkpoint(small_model, 1000, transformer_config)

        # Simulate crash by creating new pool instance
        pool2 = ModelPool(temp_dir, max_pool_size=10)

        assert len(pool2) == 1
        assert pool2.get_latest().model_id == model_id

    def test_elo_survives_crash(self, temp_dir):
        """Elo ratings should survive simulated crash."""
        tracker1 = EloTracker()
        tracker1.update_ratings("a", "b", 1.0)
        tracker1.save(os.path.join(temp_dir, "elo.json"))

        tracker2 = EloTracker()
        tracker2.load(os.path.join(temp_dir, "elo.json"))

        assert tracker2.get_rating("a") == tracker1.get_rating("a")

    def test_corrupted_checkpoint_handling(self, temp_dir, small_model, transformer_config):
        """Should handle corrupted checkpoint gracefully."""
        pool = ModelPool(temp_dir, max_pool_size=10)
        pool.add_checkpoint(small_model, 1000, transformer_config)

        # Corrupt the checkpoint file
        checkpoint = pool.get_latest()
        with open(checkpoint.path, 'w') as f:
            f.write("corrupted data")

        # Try to load - should raise exception
        with pytest.raises(Exception):
            pool.load_model(checkpoint, transformer_config)

    def test_missing_checkpoint_handling(self, temp_dir, transformer_config):
        """Should handle missing checkpoint file gracefully."""
        pool = ModelPool(temp_dir, max_pool_size=10)

        # Create fake checkpoint entry
        checkpoint = ModelCheckpoint(
            model_id="missing",
            path=os.path.join(temp_dir, "nonexistent.pt"),
            games_trained=1000,
            timestamp=time.time(),
        )
        pool.checkpoints.append(checkpoint)

        # Try to load - should raise exception
        with pytest.raises(Exception):
            pool.load_model(checkpoint, transformer_config)

    def test_partial_game_state(self):
        """Should handle incomplete game state data."""
        data = {
            'turn': 1,
            # Missing phase
            'game_state': {
                # Missing active_player
                'is_game_over': False,
                'players': [],  # Empty players
            }
        }

        # Should not crash
        state = GameState.from_dict(data, 'Agent(1)')
        assert state.turn == 1


# =============================================================================
# MULTI-AGENT COORDINATION TESTS
# =============================================================================

class TestMultiAgentCoordination:
    """Tests for multi-agent game coordination."""

    def test_both_players_tracked(self):
        """Both players' states should be tracked."""
        data = {
            'turn': 1,
            'phase': 'MAIN1',
            'game_state': {
                'active_player': 'Agent(1)',
                'priority_player': 'Agent(1)',
                'is_game_over': False,
                'players': [
                    {'name': 'Agent(1)', 'life': 20, 'hand_size': 7},
                    {'name': 'Agent(2)', 'life': 18, 'hand_size': 6},
                ],
                'stack': [],
            }
        }

        state = GameState.from_dict(data, 'Agent(1)')

        assert state.our_player.life == 20
        assert state.opponent.life == 18
        assert state.our_player.hand_size == 7
        assert state.opponent.hand_size == 6

    def test_active_player_tracking(self):
        """Active player should be correctly identified."""
        data = {
            'turn': 1,
            'phase': 'MAIN1',
            'game_state': {
                'active_player': 'Agent(2)',
                'priority_player': 'Agent(1)',
                'is_game_over': False,
                'players': [
                    {'name': 'Agent(1)', 'life': 20},
                    {'name': 'Agent(2)', 'life': 20},
                ],
                'stack': [],
            }
        }

        state = GameState.from_dict(data, 'Agent(1)')
        assert state.active_player == 'Agent(2)'
        assert state.priority_player == 'Agent(1)'

    def test_combat_state_tracking(self):
        """Combat state should track attackers."""
        data = {
            'turn': 3,
            'phase': 'COMBAT_DECLARE_ATTACKERS',
            'game_state': {
                'active_player': 'Agent(1)',
                'priority_player': 'Agent(1)',
                'is_game_over': False,
                'players': [
                    {'name': 'Agent(1)', 'life': 20},
                    {'name': 'Agent(2)', 'life': 20},
                ],
                'stack': [],
                'combat': {
                    'attacking_player': 'Agent(1)',
                    'attackers': [
                        {'card_id': 1, 'name': 'Goblin Guide', 'power': 2},
                        {'card_id': 2, 'name': 'Monastery Swiftspear', 'power': 1},
                    ]
                }
            }
        }

        state = GameState.from_dict(data, 'Agent(1)')
        assert state.combat is not None
        assert len(state.combat.attackers) == 2


# =============================================================================
# GAME STATE SERIALIZATION TESTS
# =============================================================================

class TestGameStateSerialization:
    """Tests for game state serialization and observation generation."""

    def test_observation_shape(self):
        """Observation should have correct shape."""
        data = {
            'turn': 1,
            'phase': 'MAIN1',
            'game_state': {
                'active_player': 'Agent(1)',
                'priority_player': 'Agent(1)',
                'is_game_over': False,
                'players': [
                    {'name': 'Agent(1)', 'life': 20},
                    {'name': 'Agent(2)', 'life': 20},
                ],
                'stack': [],
            }
        }

        state = GameState.from_dict(data, 'Agent(1)')
        obs = state.to_observation()

        assert obs.shape == (38,)  # Expected observation dimension

    def test_observation_normalization(self):
        """Observation values should be normalized to reasonable range."""
        data = {
            'turn': 50,
            'phase': 'MAIN1',
            'game_state': {
                'active_player': 'Agent(1)',
                'priority_player': 'Agent(1)',
                'is_game_over': False,
                'players': [
                    {'name': 'Agent(1)', 'life': 40, 'mana_pool': {'total': 10}},
                    {'name': 'Agent(2)', 'life': -5},  # Can go negative
                ],
                'stack': [],
            }
        }

        state = GameState.from_dict(data, 'Agent(1)')
        obs = state.to_observation()

        # Most values should be in [-1, 2] range
        assert np.all(obs >= -2.0)
        assert np.all(obs <= 2.0)

    def test_card_state_to_vector(self):
        """Card state should convert to vector correctly."""
        card = CardState(
            card_id=1,
            name="Lightning Bolt",
            card_type="Instant",
            mana_cost="{R}",
            cmc=1,
        )

        vec = card.to_vector()
        assert vec.shape[0] >= 20  # Minimum expected dimensions

    def test_battlefield_serialization(self):
        """Battlefield cards should be serialized correctly."""
        data = {
            'turn': 1,
            'phase': 'MAIN1',
            'game_state': {
                'active_player': 'Agent(1)',
                'priority_player': 'Agent(1)',
                'is_game_over': False,
                'players': [
                    {
                        'name': 'Agent(1)',
                        'life': 20,
                        'battlefield': [
                            {'id': 1, 'name': 'Mountain', 'types': 'Basic Land', 'is_land': True},
                            {'id': 2, 'name': 'Goblin Guide', 'types': 'Creature', 'power': 2, 'toughness': 2, 'is_creature': True},
                        ]
                    },
                    {'name': 'Agent(2)', 'life': 20},
                ],
                'stack': [],
            }
        }

        state = GameState.from_dict(data, 'Agent(1)')

        assert len(state.our_player.battlefield) == 2
        assert state.our_player.battlefield[0].is_land
        assert state.our_player.battlefield[1].is_creature


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the full self-play system."""

    def test_trainer_initialization(self, temp_dir):
        """Trainer should initialize without errors."""
        config = SelfPlayConfig(
            pool_dir=temp_dir,
            total_games=10,
            games_per_checkpoint=5,
            d_model=64,
            n_heads=4,
            n_layers=2,
        )

        trainer = SelfPlayTrainer(config)
        assert trainer is not None
        assert trainer.games_played == 0

    def test_model_weights_transfer(self, small_model, transformer_config):
        """Model weights should transfer correctly between pool and training."""
        # Get initial weights

        # Modify weights
        with torch.no_grad():
            for param in small_model.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        # Save to temp checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save({'state_dict': small_model.state_dict()}, f.name)

            # Load back
            loaded = MTGPolicyNetwork(transformer_config)
            data = torch.load(f.name, weights_only=False)
            loaded.load_state_dict(data['state_dict'])

            # Verify weights match
            for name, param in loaded.named_parameters():
                assert torch.allclose(param, dict(small_model.named_parameters())[name])

            os.unlink(f.name)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
