"""PyTorch Lightning wrapper for SimpleCNN model."""

from typing import Any, Tuple

import torch
from pytorch_lightning import LightningModule

from sam_ml.config import get_model_config
from sam_ml.modeling.models.base import BaseLightningModel
from sam_ml.modeling.models.registry import register_model
from sam_ml.modeling.models.simple_cnn import SimpleCNN


@register_model("simple_cnn")
def create_simple_cnn(
    num_classes: int | None = None,
    learning_rate: float | None = None,
    optimizer: str | None = None,
    weight_decay: float | None = None,
    input_shape: Tuple[int, int, int] | None = None,
    **kwargs: Any,
) -> "SimpleCNNLightning":
    """
    Factory function to create SimpleCNNLightning model.
    
    This function is registered in the model registry with key "simple_cnn".
    
    Args:
        num_classes: Number of output classes (defaults to config)
        learning_rate: Learning rate for optimizer (defaults to config)
        optimizer: Optimizer name ('adam' or 'sgd') (defaults to config)
        weight_decay: Weight decay (L2 regularization) coefficient (defaults to config)
        input_shape: Shape of input images (channels, height, width) (defaults to config)
        **kwargs: Additional arguments passed to BaseLightningModel
        
    Returns:
        SimpleCNNLightning model instance
    """
    # Get defaults from config
    model_config = get_model_config()
    
    return SimpleCNNLightning(
        num_classes=num_classes if num_classes is not None else model_config.num_classes,
        learning_rate=learning_rate if learning_rate is not None else model_config.learning_rate,
        optimizer=optimizer if optimizer is not None else model_config.optimizer,
        weight_decay=weight_decay if weight_decay is not None else model_config.weight_decay,
        input_shape=input_shape if input_shape is not None else model_config.input_shape,
        **kwargs,
    )


class SimpleCNNLightning(BaseLightningModel):
    """
    PyTorch Lightning wrapper for SimpleCNN model.
    
    This class integrates SimpleCNN with the training pipeline,
    providing automatic logging, optimization, and validation.
    """
    
    def __init__(
        self,
        num_classes: int | None = None,
        learning_rate: float | None = None,
        optimizer: str | None = None,
        weight_decay: float | None = None,
        input_shape: Tuple[int, int, int] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize SimpleCNNLightning model.
        
        Args:
            num_classes: Number of output classes (defaults to config)
            learning_rate: Learning rate for optimizer (defaults to config)
            optimizer: Optimizer name ('adam' or 'sgd') (defaults to config)
            weight_decay: Weight decay (L2 regularization) coefficient (defaults to config)
            input_shape: Shape of input images (channels, height, width) (defaults to config)
            **kwargs: Additional arguments passed to BaseLightningModel
        """
        # Get defaults from config
        model_config = get_model_config()
        
        # Store custom parameters before calling super().__init__()
        self.input_shape = input_shape if input_shape is not None else model_config.input_shape
        
        # Call parent constructor
        super().__init__(
            num_classes=num_classes,
            learning_rate=learning_rate,
            optimizer=optimizer,
            weight_decay=weight_decay,
            **kwargs,
        )
    
    def _create_model(self) -> None:
        """Create the SimpleCNN model architecture."""
        self.model = SimpleCNN(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Model predictions
        """
        return self.model(x)
