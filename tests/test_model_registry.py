"""Unit tests for model registry system."""

from typing import Any

import pytest
import torch

from sam_ml.modeling.models import (
    BaseLightningModel,
    get_model,
    list_models,
    register_model,
)
from sam_ml.modeling.models.base import BaseLightningModel as BaseModel
from sam_ml.modeling.models.registry import MODEL_REGISTRY


class TestModelRegistry:
    """Tests for model registry functionality."""

    def test_list_models_initially_empty(self) -> None:
        """Test that list_models returns empty list initially (before imports)."""
        # Note: After importing simple_cnn_lightning, models will be registered
        models = list_models()
        # Should at least have simple_cnn after imports
        assert isinstance(models, list)

    def test_list_models_returns_registered_models(self) -> None:
        """Test that list_models returns all registered models."""
        # Import to trigger registration
        from sam_ml.modeling.models import simple_cnn_lightning  # noqa: F401
        
        models = list_models()
        assert "simple_cnn" in models
        assert isinstance(models, list)

    def test_get_model_simple_cnn(self) -> None:
        """Test getting simple_cnn model from registry."""
        # Import to trigger registration
        from sam_ml.modeling.models import simple_cnn_lightning  # noqa: F401
        
        model = get_model("simple_cnn", num_classes=5, learning_rate=1e-4)
        
        assert isinstance(model, BaseLightningModel)
        assert model.num_classes == 5
        assert model.learning_rate == 1e-4

    def test_get_model_with_custom_params(self) -> None:
        """Test getting model with custom parameters."""
        # Import to trigger registration
        from sam_ml.modeling.models import simple_cnn_lightning  # noqa: F401
        
        model = get_model(
            "simple_cnn",
            num_classes=3,
            learning_rate=1e-3,
            optimizer="sgd",
            weight_decay=0.01,
        )
        
        assert model.num_classes == 3
        assert model.learning_rate == 1e-3
        assert model.optimizer_name == "sgd"
        assert model.weight_decay == 0.01

    def test_get_model_not_found_raises_keyerror(self) -> None:
        """Test that getting non-existent model raises KeyError."""
        with pytest.raises(KeyError, match="not found in registry"):
            get_model("nonexistent_model")

    def test_get_model_forward_pass(self) -> None:
        """Test that model from registry can perform forward pass."""
        # Import to trigger registration
        from sam_ml.modeling.models import simple_cnn_lightning  # noqa: F401
        
        model = get_model("simple_cnn", num_classes=5)
        
        # Create dummy input
        batch_size = 2
        x = torch.randn(batch_size, 3, 512, 512)
        
        # Forward pass
        output = model.forward(x)
        
        assert output.shape == (batch_size, 5)
        assert isinstance(output, torch.Tensor)

    def test_register_model_decorator(self) -> None:
        """Test registering a new model using the decorator."""
        # Create a simple test model
        class TestModelLightning(BaseLightningModel):
            def _create_model(self) -> None:
                import torch.nn as nn
                self.model = nn.Sequential(
                    nn.Linear(10, 32),
                    nn.ReLU(),
                    nn.Linear(32, self.num_classes),
                )

        # Register it
        @register_model("test_model")
        def create_test_model(**kwargs: Any) -> BaseLightningModel:
            return TestModelLightning(**kwargs)

        # Verify it's registered
        assert "test_model" in list_models()

        # Get it from registry
        model = get_model("test_model", num_classes=3)
        assert isinstance(model, BaseLightningModel)
        assert model.num_classes == 3

    def test_register_model_duplicate_key_raises_error(self) -> None:
        """Test that registering duplicate key raises ValueError."""
        # Import to trigger registration
        from sam_ml.modeling.models import simple_cnn_lightning  # noqa: F401

        # Verify simple_cnn is already registered
        assert "simple_cnn" in MODEL_REGISTRY

        # Try to register with existing key - should raise ValueError
        with pytest.raises(ValueError, match="already registered"):
            @register_model("simple_cnn")
            def duplicate_model(**kwargs: Any) -> BaseLightningModel:
                class TestModel(BaseLightningModel):
                    def _create_model(self) -> None:
                        import torch.nn as nn
                        self.model = nn.Linear(10, self.num_classes)
                return TestModel(**kwargs)


class TestSimpleCNNModel:
    """Tests for SimpleCNN model specifically."""

    def test_simple_cnn_architecture(self) -> None:
        """Test SimpleCNN model architecture."""
        from sam_ml.modeling.models.simple_cnn import SimpleCNN

        model = SimpleCNN(num_classes=5)
        
        # Test forward pass
        x = torch.randn(2, 3, 512, 512)
        output = model(x)
        
        assert output.shape == (2, 5)
        assert isinstance(output, torch.Tensor)

    def test_simple_cnn_lightning_integration(self) -> None:
        """Test SimpleCNNLightning integration with BaseLightningModel."""
        # Import to trigger registration
        from sam_ml.modeling.models import simple_cnn_lightning  # noqa: F401
        
        model = get_model("simple_cnn", num_classes=5)
        
        # Test that it has all BaseLightningModel features
        assert hasattr(model, "training_step")
        assert hasattr(model, "validation_step")
        assert hasattr(model, "test_step")
        assert hasattr(model, "configure_optimizers")
        assert model.model is not None

    def test_simple_cnn_training_step(self) -> None:
        """Test SimpleCNN training step."""
        # Import to trigger registration
        from sam_ml.modeling.models import simple_cnn_lightning  # noqa: F401
        
        model = get_model("simple_cnn", num_classes=5)
        
        # Create dummy batch
        batch = (
            torch.randn(4, 3, 512, 512),  # inputs
            torch.randint(0, 5, (4,)),  # targets
        )
        
        # Mock log method
        from unittest.mock import patch
        with patch.object(model, "log") as mock_log:
            loss = model.training_step(batch, batch_idx=0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert mock_log.called

    def test_simple_cnn_optimizer_config(self) -> None:
        """Test SimpleCNN optimizer configuration."""
        # Import to trigger registration
        from sam_ml.modeling.models import simple_cnn_lightning  # noqa: F401
        
        model = get_model("simple_cnn", num_classes=5, optimizer="sgd")
        
        config = model.configure_optimizers()
        
        assert isinstance(config, dict)
        assert "optimizer" in config
        assert "lr_scheduler" in config
        assert isinstance(config["optimizer"], torch.optim.SGD)


class TestRegistryIntegration:
    """Integration tests for the registry system."""

    def test_full_workflow(self) -> None:
        """Test complete workflow: register -> list -> get -> use."""
        # Import to trigger registration
        from sam_ml.modeling.models import simple_cnn_lightning  # noqa: F401
        
        # List models
        models = list_models()
        assert "simple_cnn" in models
        
        # Get model
        model = get_model("simple_cnn", num_classes=5)
        
        # Use model
        x = torch.randn(2, 3, 512, 512)
        output = model(x)
        
        assert output.shape == (2, 5)

    def test_multiple_models_registration(self) -> None:
        """Test that multiple models can be registered."""
        # Create and register a second model
        class Model2Lightning(BaseLightningModel):
            def _create_model(self) -> None:
                import torch.nn as nn
                self.model = nn.Linear(10, self.num_classes)

        @register_model("test_model_2")
        def create_model_2(**kwargs: Any) -> BaseLightningModel:
            return Model2Lightning(**kwargs)

        # Import to trigger simple_cnn registration
        from sam_ml.modeling.models import simple_cnn_lightning  # noqa: F401

        # Both should be available
        models = list_models()
        assert "simple_cnn" in models
        assert "test_model_2" in models

        # Both should be retrievable
        model1 = get_model("simple_cnn", num_classes=5)
        model2 = get_model("test_model_2", num_classes=5)
        
        assert isinstance(model1, BaseLightningModel)
        assert isinstance(model2, BaseLightningModel)
