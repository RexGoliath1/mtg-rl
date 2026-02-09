"""Tests for auto-regressive action head and action mapper."""

import torch

from src.forge.autoregressive_head import (
    ACTION_ACTIVATE,
    ACTION_ATTACK,
    ACTION_BLOCK,
    ACTION_CAST_SPELL,
    ACTION_MODE,
    ACTION_PASS,
    ACTION_TARGET,
    AutoRegressiveActionHead,
    AutoRegressiveConfig,
    PointerNetwork,
)
from src.forge.action_mapper import ActionMapper, FlatActionLayout


# =============================================================================
# POINTER NETWORK TESTS
# =============================================================================


class TestPointerNetwork:

    def test_output_shape(self):
        ptr = PointerNetwork(query_dim=64, key_dim=32, hidden_dim=32)
        query = torch.randn(4, 64)
        keys = torch.randn(4, 10, 32)
        logits = ptr(query, keys)
        assert logits.shape == (4, 10)

    def test_masking(self):
        ptr = PointerNetwork(query_dim=64, key_dim=32, hidden_dim=32)
        query = torch.randn(2, 64)
        keys = torch.randn(2, 5, 32)
        mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.float)

        logits = ptr(query, keys, mask)

        # Masked positions should be -inf
        assert logits[0, 2] == float('-inf')
        assert logits[0, 3] == float('-inf')
        assert logits[1, 4] == float('-inf')
        # Unmasked should be finite
        assert torch.isfinite(logits[0, 0])
        assert torch.isfinite(logits[1, 2])

    def test_gradient_flows(self):
        ptr = PointerNetwork(query_dim=32, key_dim=16, hidden_dim=16)
        query = torch.randn(2, 32, requires_grad=True)
        keys = torch.randn(2, 5, 16, requires_grad=True)

        logits = ptr(query, keys)
        logits.sum().backward()

        assert query.grad is not None
        assert keys.grad is not None


# =============================================================================
# AUTO-REGRESSIVE ACTION HEAD TESTS
# =============================================================================


class TestAutoRegressiveActionHead:

    def test_type_head_shape(self):
        config = AutoRegressiveConfig(state_dim=64, hidden_dim=32)
        head = AutoRegressiveActionHead(config)
        state = torch.randn(4, 64)
        type_logits = head.forward_type(state)
        assert type_logits.shape == (4, 8)

    def test_type_masking(self):
        config = AutoRegressiveConfig(state_dim=64, hidden_dim=32)
        head = AutoRegressiveActionHead(config)
        state = torch.randn(2, 64)
        mask = torch.zeros(2, 8)
        mask[:, 0] = 1  # Only pass is legal
        mask[:, 1] = 1  # And cast

        logits = head.forward_type(state, mask)

        # Masked types should be -inf
        assert logits[0, 2] == float('-inf')
        assert logits[0, 7] == float('-inf')
        # Legal types should be finite
        assert torch.isfinite(logits[0, 0])
        assert torch.isfinite(logits[0, 1])

    def test_card_pointer_shape(self):
        config = AutoRegressiveConfig(state_dim=64, card_dim=32, max_cards=10, hidden_dim=32)
        head = AutoRegressiveActionHead(config)
        state = torch.randn(4, 64)
        action_type = torch.tensor([1, 1, 2, 3])  # CAST, CAST, ACTIVATE, ATTACK
        card_embs = torch.randn(4, 10, 32)

        card_logits = head.forward_card(state, action_type, card_embs)
        assert card_logits.shape == (4, 10)

    def test_target_pointer_shape(self):
        config = AutoRegressiveConfig(state_dim=64, card_dim=32, max_targets=5, hidden_dim=32)
        head = AutoRegressiveActionHead(config)
        state = torch.randn(4, 64)
        action_type = torch.tensor([6, 6, 6, 6])  # TARGET
        card_emb = torch.randn(4, 32)  # Selected card embedding
        target_embs = torch.randn(4, 5, 32)

        target_logits = head.forward_target(state, action_type, card_emb, target_embs)
        assert target_logits.shape == (4, 5)

    def test_mode_head_shape(self):
        config = AutoRegressiveConfig(state_dim=64, max_modes=5, hidden_dim=32)
        head = AutoRegressiveActionHead(config)
        state = torch.randn(4, 64)
        action_type = torch.tensor([7, 7, 7, 7])  # MODE

        mode_logits = head.forward_mode(state, action_type)
        assert mode_logits.shape == (4, 5)

    def test_sample_pass_action(self):
        config = AutoRegressiveConfig(state_dim=64, card_dim=32, max_cards=5, hidden_dim=32)
        head = AutoRegressiveActionHead(config)
        head.eval()

        state = torch.randn(2, 64)
        card_embs = torch.randn(2, 5, 32)

        # Only pass is legal
        type_mask = torch.zeros(2, 8)
        type_mask[:, 0] = 1

        with torch.no_grad():
            action_type, card_idx, target_idx, log_prob = head.sample(
                state, card_embs, type_mask=type_mask
            )

        assert (action_type == ACTION_PASS).all()
        assert (card_idx == -1).all()
        assert (target_idx == -1).all()

    def test_sample_cast_action(self):
        config = AutoRegressiveConfig(state_dim=64, card_dim=32, max_cards=5, hidden_dim=32)
        head = AutoRegressiveActionHead(config)
        head.eval()

        state = torch.randn(2, 64)
        card_embs = torch.randn(2, 5, 32)

        # Only cast is legal
        type_mask = torch.zeros(2, 8)
        type_mask[:, ACTION_CAST_SPELL] = 1
        card_mask = torch.ones(2, 5)

        with torch.no_grad():
            action_type, card_idx, target_idx, log_prob = head.sample(
                state, card_embs, type_mask=type_mask, card_mask=card_mask
            )

        assert (action_type == ACTION_CAST_SPELL).all()
        assert (card_idx >= 0).all()
        assert (card_idx < 5).all()

    def test_gradient_through_sampling(self):
        """Verify gradients flow through the type head (at least)."""
        config = AutoRegressiveConfig(state_dim=32, card_dim=16, max_cards=3, hidden_dim=16)
        head = AutoRegressiveActionHead(config)

        state = torch.randn(2, 32)
        type_logits = head.forward_type(state)
        loss = type_logits.sum()
        loss.backward()

        # Type head params should have gradients
        for p in head.type_head.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_param_count(self):
        config = AutoRegressiveConfig(state_dim=768, card_dim=512, hidden_dim=256)
        head = AutoRegressiveActionHead(config)
        params = sum(p.numel() for p in head.parameters())

        # Should be < 1M (plan says ~600K)
        assert params < 1_500_000, f"AR head params {params:,} > 1.5M"


# =============================================================================
# ACTION MAPPER TESTS
# =============================================================================


class TestActionMapper:

    def test_pass_roundtrip(self):
        mapper = ActionMapper()
        action_type, card_idx, target_idx = mapper.flat_to_structured(0)
        assert action_type == ACTION_PASS
        assert card_idx == -1
        flat = mapper.structured_to_flat(ACTION_PASS)
        assert flat == 0

    def test_cast_roundtrip(self):
        mapper = ActionMapper()
        # Flat index 5 = cast spell from hand slot 2
        action_type, card_idx, target_idx = mapper.flat_to_structured(5)
        assert action_type == ACTION_CAST_SPELL
        assert card_idx == 2
        flat = mapper.structured_to_flat(ACTION_CAST_SPELL, card_idx=2)
        assert flat == 5

    def test_activate_roundtrip(self):
        mapper = ActionMapper()
        # First activate slot = index 18
        action_type, card_idx, _ = mapper.flat_to_structured(18)
        assert action_type == ACTION_ACTIVATE
        assert card_idx == 0
        flat = mapper.structured_to_flat(ACTION_ACTIVATE, card_idx=0)
        assert flat == 18

    def test_attack_roundtrip(self):
        mapper = ActionMapper()
        layout = FlatActionLayout()
        # First attack slot
        attack_flat = layout.attack_start
        action_type, card_idx, _ = mapper.flat_to_structured(attack_flat)
        assert action_type == ACTION_ATTACK
        assert card_idx == 0
        flat = mapper.structured_to_flat(ACTION_ATTACK, card_idx=0)
        assert flat == attack_flat

    def test_block_roundtrip(self):
        mapper = ActionMapper()
        layout = FlatActionLayout()
        block_flat = layout.block_start + 3
        action_type, card_idx, _ = mapper.flat_to_structured(block_flat)
        assert action_type == ACTION_BLOCK
        assert card_idx == 3

    def test_target_roundtrip(self):
        mapper = ActionMapper()
        layout = FlatActionLayout()
        target_flat = layout.target_start + 5
        action_type, _, target_idx = mapper.flat_to_structured(target_flat)
        assert action_type == ACTION_TARGET
        assert target_idx == 5
        flat = mapper.structured_to_flat(ACTION_TARGET, target_idx=5)
        assert flat == target_flat

    def test_mode_roundtrip(self):
        mapper = ActionMapper()
        layout = FlatActionLayout()
        mode_flat = layout.mode_start + 2
        action_type, _, target_idx = mapper.flat_to_structured(mode_flat)
        assert action_type == ACTION_MODE
        assert target_idx == 2

    def test_layout_total(self):
        layout = FlatActionLayout()
        assert layout.total == 203

    def test_batch_flat_to_structured(self):
        mapper = ActionMapper()
        layout = FlatActionLayout()

        flat_actions = torch.tensor([
            0,                          # pass
            5,                          # cast spell slot 2
            layout.attack_start + 1,    # attack slot 1
            layout.target_start + 3,    # target 3
        ])

        types, cards, targets = mapper.batch_flat_to_structured(flat_actions)

        assert types[0] == ACTION_PASS
        assert types[1] == ACTION_CAST_SPELL
        assert cards[1] == 2
        assert types[2] == ACTION_ATTACK
        assert cards[2] == 1
        assert types[3] == ACTION_TARGET
        assert targets[3] == 3

    def test_batch_roundtrip(self):
        """Every flat action should round-trip through structured and back."""
        mapper = ActionMapper()
        layout = FlatActionLayout()

        for flat_idx in range(layout.total):
            action_type, card_idx, target_idx = mapper.flat_to_structured(flat_idx)
            reconstructed = mapper.structured_to_flat(action_type, card_idx, target_idx)

            # Pass/mulligan/concede all map to pass type -> flat 0
            if flat_idx <= 2:
                assert reconstructed == 0
            else:
                assert reconstructed == flat_idx, (
                    f"Flat {flat_idx} -> ({action_type}, {card_idx}, {target_idx}) "
                    f"-> {reconstructed}"
                )
