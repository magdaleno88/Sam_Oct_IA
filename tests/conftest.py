"""Pytest configuration and shared fixtures."""

from typing import Generator

import pytest

# Import tensorflow with error handling for cases where it might not be available during rebuild
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    # Create a mock tensorflow module for type checking
    class MockTensorFlow:
        @staticmethod
        def random():
            class Random:
                @staticmethod
                def set_seed(seed):
                    pass
            return Random()
        
        class config:
            class experimental:
                @staticmethod
                def enable_op_determinism():
                    pass
    
    tf = MockTensorFlow()


@pytest.fixture(scope="session", autouse=True)
def setup_tensorflow() -> Generator[None, None, None]:
    """Setup TensorFlow for testing."""
    if not TENSORFLOW_AVAILABLE:
        pytest.skip("TensorFlow is not available")
    
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
    if not TENSORFLOW_AVAILABLE:
        pytest.skip("TensorFlow is not available")
    
    tf.random.set_seed(42)
    yield

