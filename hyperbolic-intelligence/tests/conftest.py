# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.
# Patent Pending. For commercial licensing: eirikreisena@gmail.com

"""
Pytest configuration and shared fixtures for CGT test suite.
"""

import pytest
import torch
import numpy as np
import random


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


@pytest.fixture(scope="session")
def seed():
    """Global random seed for reproducibility."""
    return 42


@pytest.fixture(autouse=True)
def set_random_seeds(seed):
    """Set random seeds before each test."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.fixture(scope="session")
def global_device():
    """Session-scoped device fixture."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def tolerance():
    """Numerical tolerance for assertions."""
    return {
        'rtol': 1e-5,
        'atol': 1e-6
    }
