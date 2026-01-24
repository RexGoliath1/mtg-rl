"""Tests for card encoders."""
import pytest
import torch
from shared_card_encoder import SharedCardEncoder, CardEncoderConfig, CardFeatureExtractor


class TestSharedCardEncoder:
    def test_output_shape(self, sample_card_features, sample_mask):
        """Test encoder produces correct output shape."""
        config = CardEncoderConfig()
        encoder = SharedCardEncoder(config)
        
        output = encoder(sample_card_features, sample_mask)
        
        assert output.shape == (4, 15, config.output_dim)
    
    def test_no_nan_output(self, sample_card_features, sample_mask):
        """Test encoder doesn't produce NaN."""
        encoder = SharedCardEncoder()
        output = encoder(sample_card_features, sample_mask)
        
        assert not torch.isnan(output).any()
    
    def test_encode_single(self):
        """Test single card encoding."""
        encoder = SharedCardEncoder()
        single = torch.rand(94)
        
        output = encoder.encode_single(single)
        
        assert output.shape == (256,)
    
    def test_pooled_representation(self, sample_card_features, sample_mask):
        """Test pooled output."""
        encoder = SharedCardEncoder()
        
        pooled = encoder.get_pooled_representation(
            sample_card_features, sample_mask, "mean"
        )
        
        assert pooled.shape == (4, 256)


class TestCardFeatureExtractor:
    def test_extract_basic_card(self):
        """Test feature extraction for basic card."""
        config = CardEncoderConfig()
        extractor = CardFeatureExtractor(config)
        
        card = {
            'name': 'Lightning Bolt',
            'mana_cost': '{R}',
            'type': 'Instant',
            'keywords': [],
            'rarity': 'Common',
        }
        
        features = extractor.extract(card)
        
        assert features.shape == (config.input_dim,)
        assert features.sum() > 0  # Should have some non-zero features
    
    def test_extract_creature(self):
        """Test feature extraction for creature."""
        config = CardEncoderConfig()
        extractor = CardFeatureExtractor(config)
        
        card = {
            'name': 'Serra Angel',
            'mana_cost': '{3}{W}{W}',
            'type': 'Creature - Angel',
            'power': 4,
            'toughness': 4,
            'keywords': ['Flying', 'Vigilance'],
            'rarity': 'Uncommon',
        }
        
        features = extractor.extract(card)
        
        assert features.shape == (config.input_dim,)
