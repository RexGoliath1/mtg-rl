"""Tests for training pipeline."""
import pytest
import torch
from training_pipeline import DraftModel, TrainingConfig


class TestDraftModel:
    def test_forward_pass(self):
        """Test model forward pass."""
        config = TrainingConfig()
        config.use_entity_encoder = False
        model = DraftModel(config)
        
        pack = torch.rand(2, 15, 94)
        pool = torch.rand(2, 45, 94)
        pack_mask = torch.ones(2, 15)
        pool_mask = torch.ones(2, 45)
        
        logits, value = model(pack, pool, pack_mask, pool_mask)
        
        assert logits.shape == (2, 15)
        assert value.shape == (2, 1)
    
    def test_masked_output(self):
        """Test that masked positions get -inf logits."""
        config = TrainingConfig()
        config.use_entity_encoder = False
        model = DraftModel(config)
        
        pack = torch.rand(2, 15, 94)
        pool = torch.rand(2, 45, 94)
        pack_mask = torch.zeros(2, 15)
        pack_mask[:, :5] = 1  # Only first 5 valid
        pool_mask = torch.ones(2, 45)
        
        logits, _ = model(pack, pool, pack_mask, pool_mask)
        
        # Masked positions should be -inf
        assert torch.isinf(logits[:, 5:]).all()
