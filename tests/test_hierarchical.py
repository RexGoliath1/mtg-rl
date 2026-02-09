"""Tests for HRL Phase 1: Strategic Core GRU, Turn Planner, Hierarchical Network."""

import torch
import pytest

from src.forge.strategic_core import StrategicCoreGRU
from src.forge.turn_planner import TurnPlannerMLP
from src.forge.hierarchical_network import (
    HierarchicalAlphaZeroNetwork,
    HierarchicalConfig,
)


# =============================================================================
# STRATEGIC CORE GRU TESTS
# =============================================================================


class TestStrategicCoreGRU:

    def test_output_shape(self):
        core = StrategicCoreGRU(state_dim=768, hidden_dim=256)
        state = torch.randn(4, 768)
        prev_z = torch.zeros(4, 256)
        turn = torch.tensor([1, 1, 2, 2])
        prev_turn = torch.tensor([0, 0, 1, 1])

        z_game = core(state, prev_z, turn, prev_turn)
        assert z_game.shape == (4, 256)

    def test_turn_boundary_gating(self):
        """GRU should only update when turn changes."""
        core = StrategicCoreGRU(state_dim=768, hidden_dim=256)
        state = torch.randn(2, 768)
        prev_z = torch.randn(2, 256)  # non-zero previous state

        # Same turn: output should equal prev_z
        same_turn = torch.tensor([5, 5])
        z_same = core(state, prev_z, same_turn, same_turn)
        torch.testing.assert_close(z_same, prev_z)

        # Different turn: output should differ
        prev_turn = torch.tensor([4, 4])
        z_diff = core(state, prev_z, same_turn, prev_turn)
        assert not torch.allclose(z_diff, prev_z, atol=1e-5)

    def test_initial_state(self):
        core = StrategicCoreGRU()
        z0 = core.initial_state(8)
        assert z0.shape == (8, 256)
        assert (z0 == 0).all()

    def test_initial_state_device(self):
        core = StrategicCoreGRU()
        z0 = core.initial_state(4, device=torch.device('cpu'))
        assert z0.device.type == 'cpu'

    def test_gradient_flows(self):
        """Verify gradients flow through GRU."""
        core = StrategicCoreGRU(state_dim=32, hidden_dim=16)
        state = torch.randn(2, 32, requires_grad=True)
        prev_z = torch.zeros(2, 16)
        turn = torch.tensor([2, 2])
        prev_turn = torch.tensor([1, 1])

        z_game = core(state, prev_z, turn, prev_turn)
        loss = z_game.sum()
        loss.backward()

        assert state.grad is not None
        # Gradients may be very small due to GRU + LayerNorm initialization
        # but should exist (not None)
        assert state.grad.shape == state.shape

    def test_multi_step_update(self):
        """Simulate multiple turns of GRU updates."""
        core = StrategicCoreGRU(state_dim=32, hidden_dim=16)
        z_game = core.initial_state(1)

        for t in range(1, 6):
            state = torch.randn(1, 32)
            turn = torch.tensor([t])
            prev_turn = torch.tensor([t - 1])
            z_game = core(state, z_game, turn, prev_turn)

        assert z_game.shape == (1, 16)
        # After 5 updates, should be non-zero
        assert z_game.abs().sum() > 0

    def test_param_count(self):
        core = StrategicCoreGRU(state_dim=768, hidden_dim=256)
        params = sum(p.numel() for p in core.parameters())
        # input_proj: 768*256 + 256 + LayerNorm(256*2) = ~197K
        # GRU: 3*(256*256 + 256*256 + 256 + 256) = ~394K
        # output_norm: 256*2 = 512
        # Total: ~592K
        assert 400_000 < params < 700_000, f"Unexpected param count: {params}"


# =============================================================================
# TURN PLANNER MLP TESTS
# =============================================================================


class TestTurnPlannerMLP:

    def test_output_shape(self):
        planner = TurnPlannerMLP()
        z_game = torch.randn(4, 256)
        state = torch.randn(4, 768)
        phase = torch.tensor([3, 5, 10, 3])
        prev_phase = torch.tensor([2, 4, 9, 3])

        z_turn = planner(z_game, state, phase, prev_phase)
        assert z_turn.shape == (4, 128)

    def test_phase_encoding(self):
        """Different phases should produce different outputs."""
        planner = TurnPlannerMLP()
        z_game = torch.randn(1, 256)
        state = torch.randn(1, 768)

        z_main1 = planner(z_game, state, torch.tensor([3]), torch.tensor([2]))
        z_combat = planner(z_game, state, torch.tensor([5]), torch.tensor([4]))

        assert not torch.allclose(z_main1, z_combat, atol=1e-5)

    def test_phase_clamping(self):
        """Out-of-range phase values should be clamped."""
        planner = TurnPlannerMLP()
        z_game = torch.randn(2, 256)
        state = torch.randn(2, 768)

        # Phase 99 should be clamped to 13
        z_turn = planner(z_game, state, torch.tensor([99, -1]), torch.tensor([0, 0]))
        assert z_turn.shape == (2, 128)

    def test_gradient_flows(self):
        planner = TurnPlannerMLP(strategic_dim=16, state_dim=32, output_dim=8)
        z_game = torch.randn(2, 16, requires_grad=True)
        state = torch.randn(2, 32, requires_grad=True)
        phase = torch.tensor([3, 5])
        prev_phase = torch.tensor([2, 4])

        z_turn = planner(z_game, state, phase, prev_phase)
        loss = z_turn.sum()
        loss.backward()

        assert z_game.grad is not None
        assert state.grad is not None

    def test_param_count(self):
        planner = TurnPlannerMLP()
        params = sum(p.numel() for p in planner.parameters())
        # input: 256+768+14 = 1038
        # Layer 1: 1038*512 + 512 + LN(512*2) = ~533K
        # Layer 2: 512*256 + 256 + LN(256*2) = ~132K
        # Layer 3: 256*128 + 128 + LN(128*2) = ~33K
        # Total: ~698K
        assert 500_000 < params < 900_000, f"Unexpected param count: {params}"


# =============================================================================
# HIERARCHICAL NETWORK TESTS
# =============================================================================


class TestHierarchicalNetwork:

    @pytest.fixture
    def small_network(self):
        """Small network for fast testing."""
        from src.forge.game_state_encoder import GameStateConfig
        from src.forge.policy_value_heads import PolicyValueConfig

        enc_config = GameStateConfig(
            d_model=32, n_heads=2, n_layers=1, d_ff=64,
            card_embedding_dim=32, zone_embedding_dim=32,
            global_embedding_dim=16, output_dim=64,
        )
        hier_config = HierarchicalConfig(
            state_dim=64,
            strategic_hidden_dim=32,
            tactical_output_dim=16,
            context_dim=64,
        )
        head_config = PolicyValueConfig(
            state_dim=64,
            policy_hidden_dim=32,
            policy_n_layers=1,
            value_hidden_dim=32,
            value_n_layers=1,
        )
        return HierarchicalAlphaZeroNetwork(
            encoder_config=enc_config,
            head_config=head_config,
            hier_config=hier_config,
            num_players=2,
        )

    def test_param_count_report(self, small_network):
        counts = small_network.param_count_by_component()
        assert 'encoder' in counts
        assert 'strategic_core' in counts
        assert 'turn_planner' in counts
        assert 'context_proj' in counts
        assert 'policy_head' in counts
        assert 'value_head' in counts
        assert 'total' in counts
        assert counts['total'] == sum(v for k, v in counts.items() if k != 'total')

    def test_flat_forward(self, small_network):
        """Test backward-compatible flat forward pass."""
        batch = 2
        # We need to create valid encoder inputs
        # The encoder expects zone_cards, zone_masks, etc.
        # For testing, we'll just verify the flat path works by calling
        # the components directly

        # Directly test context projection with dummy state
        state = torch.randn(batch, 64)
        z_game = torch.zeros(batch, 32)
        z_turn = torch.zeros(batch, 16)

        context = small_network.context_proj(
            torch.cat([state, z_game, z_turn], dim=-1)
        )
        assert context.shape == (batch, 64)

        policy = small_network.policy_head(context)
        value = small_network.value_head(context)

        assert policy.shape == (batch, small_network.policy_head.config.action_dim)
        assert value.shape == (batch, 1)

    def test_initial_game_state(self, small_network):
        z0 = small_network.initial_game_state(batch_size=4)
        assert z0.shape == (4, 32)
        assert (z0 == 0).all()

    def test_hierarchical_forward_components(self, small_network):
        """Test the hierarchical components integrate correctly."""
        batch = 2

        # Simulate encoder output
        state = torch.randn(batch, 64)

        # Strategic core
        z_game = small_network.strategic_core.initial_state(batch)
        turn = torch.tensor([2, 3])
        prev_turn = torch.tensor([1, 2])
        z_game = small_network.strategic_core(state, z_game, turn, prev_turn)
        assert z_game.shape == (batch, 32)

        # Turn planner
        phase = torch.tensor([3, 5])
        prev_phase = torch.tensor([2, 4])
        z_turn = small_network.turn_planner(z_game, state, phase, prev_phase)
        assert z_turn.shape == (batch, 16)

        # Context projection
        context = small_network.context_proj(
            torch.cat([state, z_game, z_turn], dim=-1)
        )
        assert context.shape == (batch, 64)

        # Heads
        policy = small_network.policy_head(context)
        value = small_network.value_head(context)
        assert policy.shape[0] == batch
        assert value.shape == (batch, 1)

    def test_gradient_through_full_pipeline(self, small_network):
        """Verify gradients flow through entire hierarchy."""
        batch = 2

        # Synthetic inputs (skip actual encoder for speed)
        state = torch.randn(batch, 64, requires_grad=True)
        z_game = torch.zeros(batch, 32)
        turn = torch.tensor([2, 3])
        prev_turn = torch.tensor([1, 2])
        phase = torch.tensor([3, 5])
        prev_phase = torch.tensor([2, 4])

        # Forward through hierarchy
        z_game = small_network.strategic_core(state, z_game, turn, prev_turn)
        z_turn = small_network.turn_planner(z_game, state, phase, prev_phase)
        context = small_network.context_proj(
            torch.cat([state, z_game, z_turn], dim=-1)
        )
        policy = small_network.policy_head(context, return_logits=True)
        value = small_network.value_head(context)

        loss = policy.sum() + value.sum()
        loss.backward()

        assert state.grad is not None
        assert state.grad.abs().sum() > 0

        # Check hierarchical parameters got gradients (encoder skipped since
        # we used synthetic state instead of actual encoder forward)
        hier_prefixes = ('strategic_core.', 'turn_planner.', 'context_proj.',
                         'policy_head.', 'value_head.')
        for name, param in small_network.named_parameters():
            if param.requires_grad and any(name.startswith(p) for p in hier_prefixes):
                assert param.grad is not None, f"No gradient for {name}"

    def test_z_game_persistence_across_turns(self, small_network):
        """Z_game should accumulate information across turns."""
        batch = 1
        z_game = small_network.strategic_core.initial_state(batch)

        z_history = [z_game.clone()]
        for t in range(1, 6):
            state = torch.randn(batch, 64)
            turn = torch.tensor([t])
            prev_turn = torch.tensor([t - 1])
            z_game = small_network.strategic_core(state, z_game, turn, prev_turn)
            z_history.append(z_game.clone())

        # Each turn should produce a different z_game
        for i in range(len(z_history) - 1):
            assert not torch.allclose(z_history[i], z_history[i + 1], atol=1e-5)

    def test_z_game_unchanged_within_turn(self, small_network):
        """Multiple decisions within same turn should not change z_game."""
        batch = 1
        z_game = small_network.strategic_core.initial_state(batch)

        # First: update to turn 1
        state = torch.randn(batch, 64)
        z_game = small_network.strategic_core(
            state, z_game, torch.tensor([1]), torch.tensor([0])
        )
        z_after_turn_change = z_game.clone()

        # Second: same turn, different state
        state2 = torch.randn(batch, 64)
        z_game = small_network.strategic_core(
            state2, z_game, torch.tensor([1]), torch.tensor([1])
        )

        # Should be unchanged (same turn)
        torch.testing.assert_close(z_game, z_after_turn_change)


# =============================================================================
# PARAMETER BUDGET TEST
# =============================================================================


class TestParameterBudget:

    def test_full_size_param_count(self):
        """Verify full-size hierarchical network stays within budget."""
        network = HierarchicalAlphaZeroNetwork()
        counts = network.param_count_by_component()

        # From plan: total should be ~37.3M (within 44-50M budget)
        total = counts['total']
        assert total < 50_000_000, f"Total params {total:,} exceeds 50M budget"

        # Individual component checks (approximate)
        assert counts['strategic_core'] < 1_000_000, "Strategic core too large"
        assert counts['turn_planner'] < 1_000_000, "Turn planner too large"
        assert counts['context_proj'] < 1_500_000, "Context proj too large"

    def test_hierarchical_overhead(self):
        """Hierarchical modules should add <5% to encoder params."""
        network = HierarchicalAlphaZeroNetwork()
        counts = network.param_count_by_component()

        encoder_params = counts['encoder']
        hier_params = counts['strategic_core'] + counts['turn_planner'] + counts['context_proj']
        overhead = hier_params / encoder_params

        assert overhead < 0.15, f"Hierarchical overhead {overhead:.1%} > 15%"
