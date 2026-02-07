"""Tests for card encoders."""
import torch
from shared_card_encoder import SharedCardEncoder, CardEncoderConfig, CardFeatureExtractor
from hybrid_card_encoder import (
    HybridCardEncoder,
    HybridEncoderConfig,
    StructuralFeatureExtractor,
)


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


class TestHybridCardEncoder:
    """Tests for v2 hybrid encoder."""

    def test_output_shape(self):
        """Test encoder produces correct output shape."""
        config = HybridEncoderConfig()
        encoder = HybridCardEncoder(config)

        batch_size, num_cards = 4, 15
        text_emb = torch.randn(batch_size, num_cards, config.text_embedding_dim)
        struct_feat = torch.randn(batch_size, num_cards, config.structural_dim)
        mask = torch.ones(batch_size, num_cards)

        output = encoder(text_emb, struct_feat, mask)

        assert output.shape == (batch_size, num_cards, config.output_dim)

    def test_no_nan_output(self):
        """Test encoder doesn't produce NaN."""
        config = HybridEncoderConfig()
        encoder = HybridCardEncoder(config)

        text_emb = torch.randn(2, 10, config.text_embedding_dim)
        struct_feat = torch.randn(2, 10, config.structural_dim)

        output = encoder(text_emb, struct_feat)

        assert not torch.isnan(output).any()

    def test_pooled_representation(self):
        """Test pooled output."""
        config = HybridEncoderConfig()
        encoder = HybridCardEncoder(config)

        text_emb = torch.randn(4, 15, config.text_embedding_dim)
        struct_feat = torch.randn(4, 15, config.structural_dim)
        mask = torch.ones(4, 15)
        mask[:, 10:] = 0  # Last 5 are padding

        pooled = encoder.get_pooled_representation(
            text_emb, struct_feat, mask, "mean"
        )

        assert pooled.shape == (4, config.output_dim)


class TestStructuralFeatureExtractor:
    """Tests for structural feature extraction."""

    def test_extract_instant(self):
        """Test extraction for instant."""
        config = HybridEncoderConfig()
        extractor = StructuralFeatureExtractor(config)

        card = {
            'name': 'Lightning Bolt',
            'mana_cost': '{R}',
            'type_line': 'Instant',
            'rarity': 'common',
        }

        features = extractor.extract(card)

        assert features.shape == (config.structural_dim,)
        assert features.sum() > 0

    def test_extract_creature(self):
        """Test extraction for creature with P/T."""
        config = HybridEncoderConfig()
        extractor = StructuralFeatureExtractor(config)

        card = {
            'name': 'Tarmogoyf',
            'mana_cost': '{1}{G}',
            'type_line': 'Creature - Lhurgoyf',
            'power': '2',
            'toughness': '3',
            'rarity': 'mythic',
            'cmc': 2,
        }

        features = extractor.extract(card)

        assert features.shape == (config.structural_dim,)
