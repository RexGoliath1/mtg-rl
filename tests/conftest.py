"""Shared test fixtures."""
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.fixture
def device():
    """Return available device."""
    if not HAS_TORCH:
        pytest.skip("torch not installed")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_card_features():
    """Sample card features for testing."""
    if not HAS_TORCH:
        pytest.skip("torch not installed")
    return torch.rand(4, 15, 94)


@pytest.fixture
def sample_mask():
    """Sample mask for testing."""
    if not HAS_TORCH:
        pytest.skip("torch not installed")
    mask = torch.ones(4, 15)
    mask[:, 10:] = 0
    return mask
