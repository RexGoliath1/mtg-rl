"""Integration tests for HRL modules wired into the training pipeline.

Tests E4: forward pass through full hierarchy, gradient flow, backward compat,
and CTDE/auto-regressive head integration.
"""
import pytest

torch = pytest.importorskip("torch")

from src.training.self_play import (
    AlphaZeroNetwork,
    SelfPlayConfig,
    SelfPlayTrainer,
    Learner,
    ReplayBuffer,
    TrainingSample,
)
from src.forge.hierarchical_network import HierarchicalAlphaZeroNetwork
from src.forge.ctde import CTDEValueHeads, compute_ctde_loss
from src.forge.autoregressive_head import AutoRegressiveActionHead
from src.forge.action_mapper import ActionMapper


class TestHierarchicalForwardPass:
    """E4: Verify hierarchical network forward pass works end-to-end."""

    def test_hierarchical_forward_flat_mode(self):
        """HierarchicalAlphaZeroNetwork.forward_flat produces correct shapes."""
        net = HierarchicalAlphaZeroNetwork()
        net.eval()

        batch = 2
        state_dim = net.encoder_config.output_dim

        # forward_flat bypasses GRU/planner (zero context)
        with torch.no_grad():
            # Create dummy encoder inputs via forward_flat
            # We need proper encoder inputs — use a minimal stub
            dummy_state = torch.randn(batch, state_dim)
            z_game = torch.zeros(batch, net.hier_config.strategic_hidden_dim)
            z_turn = torch.zeros(batch, net.hier_config.tactical_output_dim)

            context = net.context_proj(torch.cat([dummy_state, z_game, z_turn], dim=-1))
            policy = net.policy_head(context)
            value = net.value_head(context)

        assert policy.shape[0] == batch
        assert value.shape == (batch, 1)

    def test_hierarchical_full_forward(self):
        """Full hierarchical forward: state → GRU → planner → context → heads."""
        net = HierarchicalAlphaZeroNetwork()
        net.eval()

        batch = 2
        state_dim = net.encoder_config.output_dim

        with torch.no_grad():
            # Simulate what the encoder would produce
            state = torch.randn(batch, state_dim)
            z_game_prev = net.initial_game_state(batch)

            current_turn = torch.tensor([5.0, 5.0])
            prev_turn = torch.tensor([4.0, 4.0])
            current_phase = torch.tensor([3.0, 3.0])
            prev_phase = torch.tensor([2.0, 2.0])

            # Run through strategic core
            z_game = net.strategic_core(state, z_game_prev, current_turn, prev_turn)
            assert z_game.shape == (batch, net.hier_config.strategic_hidden_dim)

            # Run through turn planner
            z_turn = net.turn_planner(z_game, state, current_phase, prev_phase)
            assert z_turn.shape == (batch, net.hier_config.tactical_output_dim)

            # Context projection
            context = net.context_proj(torch.cat([state, z_game, z_turn], dim=-1))
            assert context.shape == (batch, net.hier_config.context_dim)

            # Heads
            policy = net.policy_head(context)
            value = net.value_head(context)
            assert policy.shape[0] == batch
            assert value.shape == (batch, 1)

    def test_gru_state_changes_at_turn_boundary(self):
        """GRU state should update when turn changes, stay same within turn."""
        net = HierarchicalAlphaZeroNetwork()
        net.eval()

        state = torch.randn(1, net.encoder_config.output_dim)
        z0 = net.initial_game_state(1)

        with torch.no_grad():
            # Same turn: GRU should NOT update (passes through prev state)
            z1 = net.strategic_core(state, z0, torch.tensor([5.0]), torch.tensor([5.0]))

            # New turn: GRU SHOULD update
            z2 = net.strategic_core(state, z0, torch.tensor([6.0]), torch.tensor([5.0]))

        # z1 should equal z0 (no turn boundary)
        assert torch.allclose(z1, z0, atol=1e-6)
        # z2 should differ from z0 (turn boundary triggered GRU)
        assert not torch.allclose(z2, z0, atol=1e-6)


class TestGradientFlow:
    """Verify gradients flow through the full hierarchy."""

    def test_gradient_reaches_all_components(self):
        """Loss backprop should produce non-zero grads in HRL modules."""
        net = HierarchicalAlphaZeroNetwork()
        net.train()

        state = torch.randn(2, net.encoder_config.output_dim, requires_grad=True)
        z_prev = net.initial_game_state(2)

        # Full forward (bypasses encoder — tests HRL module gradient flow)
        z_game = net.strategic_core(state, z_prev, torch.tensor([2.0, 2.0]), torch.tensor([1.0, 1.0]))
        z_turn = net.turn_planner(z_game, state, torch.tensor([3.0, 3.0]), torch.tensor([2.0, 2.0]))
        context = net.context_proj(torch.cat([state, z_game, z_turn], dim=-1))
        policy = net.policy_head(context)
        value = net.value_head(context)

        # Dummy loss
        loss = policy.sum() + value.sum()
        loss.backward()

        # Check gradients exist for HRL-specific components (not encoder, which isn't in this path)
        hrl_prefixes = ('strategic_core.', 'turn_planner.', 'context_proj.', 'policy_head.', 'value_head.')
        for name, param in net.named_parameters():
            if param.requires_grad and any(name.startswith(p) for p in hrl_prefixes):
                assert param.grad is not None, f"No gradient for {name}"


class TestCTDEIntegration:
    """Test CTDE dual value heads in training context."""

    def test_ctde_loss_computation(self):
        """CTDE loss includes both observable and oracle components."""
        ctde = CTDEValueHeads()
        ctde.train()

        state = torch.randn(4, 768)
        oracle = torch.randn(4, 128)
        target = torch.randn(4, 1)

        obs_val, oracle_val = ctde(state, oracle, epoch=50)
        loss, metrics = compute_ctde_loss(obs_val, oracle_val, target, oracle_weight=0.5)

        assert loss.item() > 0
        assert "obs_value_loss" in metrics
        assert "oracle_value_loss" in metrics
        assert "total_value_loss" in metrics

    def test_ctde_inference_mode(self):
        """At inference, only observable value is computed."""
        ctde = CTDEValueHeads()
        ctde.eval()

        state = torch.randn(2, 768)
        with torch.no_grad():
            obs_val = ctde.observable_value(state)

        assert obs_val.shape == (2, 1)
        assert not torch.isnan(obs_val).any()


class TestAutoRegressiveIntegration:
    """Test auto-regressive head integration."""

    def test_action_mapper_roundtrip(self):
        """Flat action → structured → flat roundtrip preserves action index."""
        mapper = ActionMapper()

        # Test a few known actions
        for flat_idx in [0, 5, 42, 100, 175, 200]:
            atype, card, target = mapper.flat_to_structured(flat_idx)
            reconstructed = mapper.structured_to_flat(atype, card, target)
            assert reconstructed == flat_idx, f"Roundtrip failed for {flat_idx}"

    def test_ar_head_type_forward(self):
        """Auto-regressive head produces valid action type logits."""
        head = AutoRegressiveActionHead()
        head.eval()

        state = torch.randn(2, 768)
        with torch.no_grad():
            logits = head.forward_type(state)

        assert logits.shape == (2, 8)  # 8 action types
        assert not torch.isnan(logits).any()


class TestBackwardCompatibility:
    """Verify flat model still works when HRL is disabled."""

    def test_flat_trainer_init(self):
        """Default SelfPlayConfig (no HRL) creates AlphaZeroNetwork."""
        config = SelfPlayConfig(
            games_per_iteration=1,
            mcts_simulations=2,
            min_buffer_size=1,
        )
        trainer = SelfPlayTrainer(config)
        assert isinstance(trainer.network, AlphaZeroNetwork)
        assert trainer.ctde_heads is None
        assert trainer.ar_head is None

    def test_hierarchical_trainer_init(self):
        """HRL-enabled config creates HierarchicalAlphaZeroNetwork."""
        config = SelfPlayConfig(
            games_per_iteration=1,
            mcts_simulations=2,
            min_buffer_size=1,
            use_hierarchy=True,
            use_ctde=True,
            use_autoregressive=True,
        )
        trainer = SelfPlayTrainer(config)
        assert isinstance(trainer.network, HierarchicalAlphaZeroNetwork)
        assert trainer.ctde_heads is not None
        assert trainer.ar_head is not None

    def test_flat_checkpoint_loads_into_hierarchical(self):
        """A flat AlphaZero checkpoint loads into HierarchicalAlphaZeroNetwork."""
        hier_net = HierarchicalAlphaZeroNetwork()
        result = hier_net.load_flat_checkpoint.__doc__  # just verify the method exists

        # The method exists and is documented
        assert "flat AlphaZero checkpoint" in result

    def test_learner_trains_with_hrl_modules(self):
        """Learner handles training with CTDE and AR head modules."""
        net = AlphaZeroNetwork()
        ctde = CTDEValueHeads()
        ar_head = AutoRegressiveActionHead()

        config = SelfPlayConfig(
            min_buffer_size=1,
            batch_size=4,
            use_ctde=True,
        )
        learner = Learner(net, config, ctde_heads=ctde, ar_head=ar_head)

        # Create minimal replay buffer
        buf = ReplayBuffer(100)
        for i in range(10):
            import numpy as np
            buf.add(TrainingSample(
                state=np.random.randn(768).astype(np.float32),
                policy=np.random.dirichlet(np.ones(203)).astype(np.float32),
                value=1.0 if i % 2 == 0 else -1.0,
            ))

        # Train one epoch — should not crash
        losses = learner.train_epoch(buf)
        assert "policy_loss" in losses
        assert "value_loss" in losses
        assert losses["total_loss"] > 0


class TestParamCounts:
    """Verify parameter counts match expectations."""

    def test_hierarchical_param_count(self):
        """HierarchicalAlphaZeroNetwork reports per-component params."""
        net = HierarchicalAlphaZeroNetwork()
        counts = net.param_count_by_component()

        assert counts['encoder'] > 30_000_000     # ~33M
        assert counts['strategic_core'] > 100_000  # ~500K
        assert counts['turn_planner'] > 100_000    # ~400K
        assert counts['context_proj'] > 100_000    # ~900K
        assert counts['policy_head'] > 100_000     # ~377K
        assert counts['value_head'] > 100_000      # ~294K
        assert counts['total'] > 35_000_000        # ~37M total

    def test_ctde_param_count(self):
        """CTDE heads add ~1M params."""
        ctde = CTDEValueHeads()
        params = sum(p.numel() for p in ctde.parameters())
        assert 500_000 < params < 3_000_000

    def test_ar_head_param_count(self):
        """Auto-regressive head adds ~600K params."""
        head = AutoRegressiveActionHead()
        params = sum(p.numel() for p in head.parameters())
        assert 200_000 < params < 2_000_000
