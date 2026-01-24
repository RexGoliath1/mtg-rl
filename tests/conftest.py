"""Shared test fixtures."""
import pytest
import torch


@pytest.fixture
def device():
    """Return available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_card_features():
    """Sample card features for testing."""
    return torch.rand(4, 15, 94)


@pytest.fixture
def sample_mask():
    """Sample mask for testing."""
    mask = torch.ones(4, 15)
    mask[:, 10:] = 0
    return mask
