"""Integration tests for Forge components.

Tests PolicyValueNetwork forward passes, MCTS node operations,
GameConfig validation, and checkpoint save/load roundtrips.
All tests use synthetic data (no HDF5 or external files needed).
"""
import tempfile
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.forge.policy_value_heads import (
    ActionConfig,
    PolicyValueConfig,
    PolicyValueNetwork,
)
from src.forge.mcts import MCTSNode
from src.forge.game_config import FORMATS, get_format, FORMAT_NAMES


# =============================================================================
# POLICY/VALUE NETWORK TESTS
# =============================================================================


class _StubEncoder(torch.nn.Module):
    """Simple encoder that accepts **kwargs for PolicyValueNetwork compatibility."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, input_tensor, **kwargs):
        return self.linear(input_tensor)


class TestPolicyValueNetwork:
    """Test the combined policy-value network with a simple encoder stub."""

    @pytest.fixture
    def action_config(self):
        return ActionConfig()

    @pytest.fixture
    def config(self, action_config):
        return PolicyValueConfig(
            state_dim=768,
            action_config=action_config,
        )

    @pytest.fixture
    def stub_encoder(self, config):
        """Simple encoder for testing (avoids needing HDF5 data)."""
        return _StubEncoder(32, config.state_dim)

    @pytest.fixture
    def network(self, stub_encoder, config):
        return PolicyValueNetwork(stub_encoder, config)

    def test_forward_pass_shapes(self, network, config, action_config):
        """Policy and value heads produce correct output shapes."""
        batch_size = 4
        x = torch.randn(batch_size, 32)
        policy_logits, value = network(input_tensor=x)

        assert policy_logits.shape == (batch_size, action_config.total_actions)
        assert value.shape == (batch_size, 1)

    def test_no_nan_output(self, network):
        """Network should not produce NaN for normal inputs."""
        x = torch.randn(8, 32)
        policy_logits, value = network(input_tensor=x)

        assert not torch.isnan(policy_logits).any()
        assert not torch.isnan(value).any()

    def test_action_masking(self, network, action_config):
        """Masked actions should get -inf after masking."""
        batch_size = 2
        x = torch.randn(batch_size, 32)
        policy_logits, _ = network(input_tensor=x)

        # Create mask: only first 5 actions legal
        mask = torch.zeros(batch_size, action_config.total_actions)
        mask[:, :5] = 1

        masked_logits = policy_logits.masked_fill(mask == 0, float('-inf'))
        probs = torch.nn.functional.softmax(masked_logits, dim=-1)

        # Probability should be zero for masked actions
        assert (probs[:, 5:] == 0).all()
        # Probabilities should sum to ~1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5)

    def test_value_range(self, network):
        """Value output should be bounded (tanh activation)."""
        x = torch.randn(16, 32)
        _, value = network(input_tensor=x)

        assert (value >= -1.0).all()
        assert (value <= 1.0).all()

    def test_gradient_flows(self, network):
        """Gradients should flow through both heads."""
        x = torch.randn(4, 32, requires_grad=True)
        policy_logits, value = network(input_tensor=x)

        loss = policy_logits.sum() + value.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0


# =============================================================================
# ACTION CONFIG TESTS
# =============================================================================


class TestActionConfig:
    """Test action space configuration."""

    def test_total_actions_203(self):
        """Default ActionConfig should yield 203 total actions."""
        config = ActionConfig()
        assert config.total_actions == 203

    def test_action_indices_non_overlapping(self):
        """Action type ranges should not overlap."""
        config = ActionConfig()
        from src.forge.policy_value_heads import ActionType

        ranges = []
        for action_type in ActionType:
            start, end = config.get_action_indices(action_type)
            ranges.append((start, end, action_type.name))

        # Sort by start index
        ranges.sort()

        # Check no overlaps
        for i in range(len(ranges) - 1):
            _, end_i, name_i = ranges[i]
            start_j, _, name_j = ranges[i + 1]
            assert end_i <= start_j, f"{name_i} [{end_i}] overlaps with {name_j} [{start_j}]"

    def test_all_actions_covered(self):
        """Every index 0..total_actions-1 should belong to some action type."""
        config = ActionConfig()
        from src.forge.policy_value_heads import ActionType

        covered = set()
        for action_type in ActionType:
            start, end = config.get_action_indices(action_type)
            for i in range(start, end):
                covered.add(i)

        assert len(covered) == config.total_actions


# =============================================================================
# MCTS NODE TESTS
# =============================================================================


class TestMCTSNode:
    """Test MCTS tree node operations."""

    def test_initial_state(self):
        """Fresh node should have zero visits and no children."""
        node = MCTSNode(prior=0.5)
        assert node.visit_count == 0
        assert node.value == 0.0
        assert not node.is_expanded
        assert node.prior == 0.5

    def test_expand_creates_children(self):
        """Expanding a node should create children for legal actions."""
        node = MCTSNode()
        priors = np.array([0.3, 0.5, 0.2])
        legal_actions = [0, 1, 2]
        node.expand(priors, legal_actions, game_state={"turn": 1})

        assert node.is_expanded
        assert len(node.children) == 3
        assert 0 in node.children
        assert 1 in node.children
        assert 2 in node.children

    def test_child_priors(self):
        """Children should have correct prior probabilities."""
        node = MCTSNode()
        priors = np.array([0.3, 0.7])
        legal_actions = [0, 1]
        node.expand(priors, legal_actions, game_state={})

        assert abs(node.children[0].prior - 0.3) < 1e-6
        assert abs(node.children[1].prior - 0.7) < 1e-6

    def test_value_averaging(self):
        """Node value should be average of backpropagated values."""
        node = MCTSNode()
        # Simulate 3 visits with values 1.0, 0.5, 0.0
        node.visit_count = 3
        node.value_sum = 1.5

        assert abs(node.value - 0.5) < 1e-6

    def test_parent_child_links(self):
        """Children should reference their parent."""
        parent = MCTSNode()
        # priors array must be large enough to index by action id
        priors = np.zeros(50)
        priors[42] = 1.0
        parent.expand(priors, [42], game_state={})

        child = parent.children[42]
        assert child.parent is parent
        assert child.action == 42


# =============================================================================
# GAME CONFIG TESTS
# =============================================================================


class TestGameConfig:
    """Test format configuration validation."""

    def test_commander_config(self):
        fmt = FORMATS["commander"]
        assert fmt.starting_life == 40
        assert fmt.deck_size == 100
        assert fmt.max_players == 4
        assert fmt.allow_commander is True
        assert fmt.singleton is True

    def test_modern_config(self):
        fmt = FORMATS["modern"]
        assert fmt.starting_life == 20
        assert fmt.deck_size == 60
        assert fmt.max_players == 2
        assert fmt.allow_commander is False

    def test_standard_config(self):
        fmt = FORMATS["standard"]
        assert fmt.starting_life == 20
        assert fmt.deck_size == 60

    def test_get_format_case_insensitive(self):
        fmt = get_format("Commander")
        assert fmt.name == "commander"

    def test_get_format_invalid(self):
        with pytest.raises(ValueError):
            get_format("vintage")

    def test_all_formats_registered(self):
        assert len(FORMAT_NAMES) >= 3
        assert "commander" in FORMAT_NAMES
        assert "modern" in FORMAT_NAMES
        assert "standard" in FORMAT_NAMES

    def test_format_frozen(self):
        """FormatConfig should be immutable."""
        fmt = FORMATS["commander"]
        with pytest.raises(Exception):
            fmt.starting_life = 20


# =============================================================================
# CHECKPOINT ROUNDTRIP TESTS
# =============================================================================


class TestCheckpointRoundtrip:
    """Test save/load for policy-value network."""

    def test_save_load_roundtrip(self):
        """Model weights should survive save/load cycle."""
        action_cfg = ActionConfig(max_hand_size=3, max_battlefield=5, max_targets=2, max_modes=2, max_costs=2)
        config = PolicyValueConfig(state_dim=64, action_config=action_cfg)
        encoder = _StubEncoder(16, 64)
        net = PolicyValueNetwork(encoder, config)
        net.eval()

        # Get original output
        x = torch.randn(2, 16)
        with torch.no_grad():
            orig_policy, orig_value = net(input_tensor=x)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name

        try:
            torch.save(net.state_dict(), path)

            # Load into fresh network
            net2 = PolicyValueNetwork(_StubEncoder(16, 64), config)
            net2.load_state_dict(torch.load(path, weights_only=True))
            net2.eval()

            with torch.no_grad():
                loaded_policy, loaded_value = net2(input_tensor=x)

            torch.testing.assert_close(orig_policy, loaded_policy)
            torch.testing.assert_close(orig_value, loaded_value)
        finally:
            os.unlink(path)

    def test_state_dict_keys(self):
        """State dict should have expected top-level keys."""
        action_cfg = ActionConfig(max_hand_size=3, max_battlefield=5, max_targets=2, max_modes=2, max_costs=2)
        config = PolicyValueConfig(state_dim=64, action_config=action_cfg)
        encoder = _StubEncoder(16, 64)
        net = PolicyValueNetwork(encoder, config)

        keys = set(net.state_dict().keys())
        # Should have state_encoder, policy_head, and value_head params
        assert any("state_encoder" in k for k in keys)
        assert any("policy_head" in k for k in keys)
        assert any("value_head" in k for k in keys)
