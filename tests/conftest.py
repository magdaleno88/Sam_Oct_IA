"""Pytest configuration and shared fixtures."""

from typing import Generator

import pytest
import tensorflow as tf


@pytest.fixture(scope="session", autouse=True)
def setup_tensorflow() -> Generator[None, None, None]:
    """Setup TensorFlow for testing."""
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    
    # Configure TensorFlow to use deterministic operations if possible
    tf.config.experimental.enable_op_determinism()
    
    yield
    
    # Cleanup if needed
    pass


@pytest.fixture(autouse=True)
def reset_random_seed() -> Generator[None, None, None]:
    """Reset random seeds before each test for reproducibility."""
    tf.random.set_seed(42)
    yield

