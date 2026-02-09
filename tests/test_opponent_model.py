"""Tests for opponent belief model and informed determinization."""

import torch

from src.forge.opponent_model import (
    CardAttentionBlock,
    OpponentBeliefModel,
    OpponentModelConfig,
    compute_belief_loss,
    informed_determinize,
)


# =============================================================================
# CARD ATTENTION BLOCK TESTS
# =============================================================================


class TestCardAttentionBlock:

    def test_output_shape(self):
        block = CardAttentionBlock(dim=64, n_heads=4)
        x = torch.randn(4, 10, 64)
        out = block(x)
        assert out.shape == (4, 10, 64)

    def test_with_mask(self):
        block = CardAttentionBlock(dim=64, n_heads=4)
        x = torch.randn(2, 8, 64)
        mask = torch.tensor([
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0],
        ], dtype=torch.float)
        out = block(x, mask)
        assert out.shape == (2, 8, 64)

    def test_gradient_flows(self):
        block = CardAttentionBlock(dim=32, n_heads=2)
        x = torch.randn(2, 5, 32, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None


# =============================================================================
# OPPONENT BELIEF MODEL TESTS
# =============================================================================


class TestOpponentBeliefModel:

    def test_output_shapes(self):
        config = OpponentModelConfig(state_dim=64, hidden_dim=32, n_attention_heads=2)
        model = OpponentBeliefModel(config)

        state = torch.randn(4, 64)
        card_pool = torch.randn(4, 20, 64)

        card_probs, hand_size_logits = model(state, card_pool)

        assert card_probs.shape == (4, 20)
        assert hand_size_logits.shape == (4, 16)  # max_hand_size + 1 = 16

    def test_output_range(self):
        config = OpponentModelConfig(state_dim=64, hidden_dim=32, n_attention_heads=2)
        model = OpponentBeliefModel(config)
        model.eval()

        state = torch.randn(4, 64)
        card_pool = torch.randn(4, 10, 64)

        with torch.no_grad():
            card_probs, _ = model(state, card_pool)

        # Card probs should be in [0, 1] (sigmoid output)
        assert (card_probs >= 0).all()
        assert (card_probs <= 1).all()

    def test_masked_positions_zero(self):
        config = OpponentModelConfig(state_dim=64, hidden_dim=32, n_attention_heads=2)
        model = OpponentBeliefModel(config)
        model.eval()

        state = torch.randn(2, 64)
        card_pool = torch.randn(2, 8, 64)
        mask = torch.tensor([
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
        ], dtype=torch.float)

        with torch.no_grad():
            card_probs, _ = model(state, card_pool, mask)

        # Masked positions should be zero
        assert (card_probs[0, 3:] == 0).all()
        assert (card_probs[1, 4:] == 0).all()

    def test_predict_hand_convenience(self):
        config = OpponentModelConfig(state_dim=64, hidden_dim=32, n_attention_heads=2)
        model = OpponentBeliefModel(config)
        model.eval()

        state = torch.randn(2, 64)
        card_pool = torch.randn(2, 10, 64)

        with torch.no_grad():
            probs = model.predict_hand(state, card_pool)

        assert probs.shape == (2, 10)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_gradient_flows(self):
        config = OpponentModelConfig(state_dim=32, hidden_dim=16, n_attention_heads=2)
        model = OpponentBeliefModel(config)

        state = torch.randn(2, 32, requires_grad=True)
        card_pool = torch.randn(2, 5, 32, requires_grad=True)

        card_probs, hand_size_logits = model(state, card_pool)
        loss = card_probs.sum() + hand_size_logits.sum()
        loss.backward()

        assert state.grad is not None
        assert card_pool.grad is not None

    def test_param_count(self):
        config = OpponentModelConfig(state_dim=768, hidden_dim=256, n_attention_heads=4)
        model = OpponentBeliefModel(config)
        params = sum(p.numel() for p in model.parameters())

        # Should be < 2.5M (plan estimated ~800K for simpler version;
        # attention layers + FFN expansion add overhead but we're well
        # within the 37.3M total budget)
        assert params < 2_500_000, f"Opponent model params {params:,} > 2.5M"

    def test_batch_size_one(self):
        """Model should work with batch size 1."""
        config = OpponentModelConfig(state_dim=64, hidden_dim=32, n_attention_heads=2)
        model = OpponentBeliefModel(config)
        model.eval()

        state = torch.randn(1, 64)
        card_pool = torch.randn(1, 5, 64)

        with torch.no_grad():
            card_probs, hand_size_logits = model(state, card_pool)

        assert card_probs.shape == (1, 5)
        assert hand_size_logits.shape == (1, 16)


# =============================================================================
# BELIEF LOSS TESTS
# =============================================================================


class TestBeliefLoss:

    def test_perfect_prediction(self):
        """Perfect predictions should give low loss."""
        card_probs = torch.tensor([[0.99, 0.99, 0.01, 0.01, 0.01]])
        hand_size_logits = torch.zeros(1, 16)
        hand_size_logits[0, 2] = 10.0  # Predict hand size = 2

        actual_hand = torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.float)
        actual_size = torch.tensor([2], dtype=torch.long)

        loss, metrics = compute_belief_loss(
            card_probs, hand_size_logits, actual_hand, actual_size
        )

        assert loss.item() < 0.5  # Should be low
        assert 'card_bce_loss' in metrics
        assert 'hand_size_loss' in metrics
        assert 'total_belief_loss' in metrics

    def test_with_mask(self):
        """Loss should only consider valid card positions."""
        card_probs = torch.tensor([[0.5, 0.5, 0.5, 0.0, 0.0]])
        hand_size_logits = torch.zeros(1, 16)
        actual_hand = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.float)
        actual_size = torch.tensor([1], dtype=torch.long)
        mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float)

        loss, metrics = compute_belief_loss(
            card_probs, hand_size_logits, actual_hand, actual_size, mask
        )

        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_gradient_flows(self):
        card_probs = torch.tensor([[0.5, 0.3, 0.7]], requires_grad=True)
        hand_size_logits = torch.randn(1, 16, requires_grad=True)
        actual_hand = torch.tensor([[1, 0, 1]], dtype=torch.float)
        actual_size = torch.tensor([2], dtype=torch.long)

        loss, _ = compute_belief_loss(
            card_probs, hand_size_logits, actual_hand, actual_size
        )
        loss.backward()

        assert card_probs.grad is not None
        assert hand_size_logits.grad is not None


# =============================================================================
# INFORMED DETERMINIZE TESTS
# =============================================================================


class TestInformedDeterminize:

    def test_correct_hand_size(self):
        card_probs = torch.tensor([0.9, 0.8, 0.7, 0.2, 0.1])
        hand = informed_determinize(card_probs, hand_size=3)
        assert len(hand) == 3

    def test_unique_indices(self):
        """Sampled cards should be unique (no replacement)."""
        card_probs = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
        hand = informed_determinize(card_probs, hand_size=3)
        assert len(hand) == len(set(hand.tolist()))

    def test_empty_hand(self):
        card_probs = torch.tensor([0.5, 0.3, 0.7])
        hand = informed_determinize(card_probs, hand_size=0)
        assert len(hand) == 0

    def test_zero_probs(self):
        card_probs = torch.tensor([0.0, 0.0, 0.0])
        hand = informed_determinize(card_probs, hand_size=2)
        assert len(hand) == 0

    def test_high_prob_cards_favored(self):
        """Cards with high probability should be selected more often."""
        torch.manual_seed(42)
        card_probs = torch.tensor([0.99, 0.99, 0.01, 0.01, 0.01])

        # Sample many times and count
        counts = torch.zeros(5)
        for _ in range(100):
            hand = informed_determinize(card_probs, hand_size=2)
            for idx in hand:
                counts[idx] += 1

        # High-prob cards should be selected much more often
        assert counts[0] > counts[3]
        assert counts[1] > counts[4]

    def test_temperature_effect(self):
        """Low temperature should make selection more peaked."""
        torch.manual_seed(42)
        card_probs = torch.tensor([0.6, 0.3, 0.1])

        # Low temperature: almost always picks top card
        low_temp_counts = torch.zeros(3)
        for _ in range(100):
            hand = informed_determinize(card_probs, hand_size=1, temperature=0.1)
            low_temp_counts[hand[0]] += 1

        # High temperature: more uniform
        high_temp_counts = torch.zeros(3)
        for _ in range(100):
            hand = informed_determinize(card_probs, hand_size=1, temperature=5.0)
            high_temp_counts[hand[0]] += 1

        # Low temp should be more concentrated on index 0
        assert low_temp_counts[0] > high_temp_counts[0]

    def test_hand_size_exceeds_pool(self):
        """If hand_size > available cards, return all available."""
        card_probs = torch.tensor([0.5, 0.5, 0.5])
        hand = informed_determinize(card_probs, hand_size=10)
        assert len(hand) == 3  # Only 3 cards available

    def test_returns_valid_indices(self):
        card_probs = torch.tensor([0.5, 0.3, 0.7, 0.1])
        hand = informed_determinize(card_probs, hand_size=2)
        for idx in hand:
            assert 0 <= idx < 4
