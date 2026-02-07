"""Shared test fixtures."""
import os
import sys
import pytest

# Add project root and scripts/ to path so tests can import from both
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "scripts"))

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
