"""Base model class for PyTorch Lightning models."""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from sam_ml.config import get_model_config, get_scheduler_config


class BaseLightningModel(LightningModule):
    """
    Base class for all PyTorch Lightning models in the SAM-AI project.
    
    This class provides a foundation for model implementations with:
    - Standard training/validation/test step structure
    - Placeholder methods for MLFlow integration (to be implemented)
    - Common configuration and logging patterns
    
    Subclasses should implement:
    - `forward()`: Forward pass of the model
    - `_create_model()`: Model architecture creation
    - `_compute_loss()`: Loss computation logic
    - `_compute_metrics()`: Metric computation logic
    """

    def __init__(
        self,
        num_classes: int | None = None,
        learning_rate: float | None = None,
        optimizer: str | None = None,
        weight_decay: float | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the base model.
        
        Args:
            num_classes: Number of output classes for classification (defaults to config)
            learning_rate: Learning rate for optimizer (defaults to config)
            optimizer: Optimizer name ('adam', 'sgd', etc.) (defaults to config)
            weight_decay: Weight decay (L2 regularization) coefficient (defaults to config)
            **kwargs: Additional arguments passed to subclasses
        """
        super().__init__()
        
        # Get defaults from config
        model_config = get_model_config()
        
        self.num_classes: int = num_classes if num_classes is not None else model_config.num_classes
        self.learning_rate: float = learning_rate if learning_rate is not None else model_config.learning_rate
        self.optimizer_name: str = (optimizer or model_config.optimizer).lower()
        self.weight_decay: float = weight_decay if weight_decay is not None else model_config.weight_decay
        
        self.save_hyperparameters()
        
        # Model architecture (to be created by subclasses)
        self.model: Optional[nn.Module] = None
        
        # Loss function
        self.criterion: nn.Module = nn.CrossEntropyLoss()
        
        # Initialize model architecture
        self._create_model()

    def _create_model(self) -> None:
        """
        Create the model architecture.
        
        This method should be overridden by subclasses to define
        the specific model architecture.
        """
        raise NotImplementedError(
            "Subclasses must implement _create_model() to define the model architecture"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output tensor
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call _create_model() first.")
        return self.model(x)

    def _compute_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the loss between predictions and targets.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Loss tensor
        """
        return self.criterion(predictions, targets)

    def _compute_metrics(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute metrics from predictions and targets.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of metric names and values
        """
        # Convert logits to predicted classes
        pred_classes = torch.argmax(predictions, dim=1)
        
        # Compute accuracy
        accuracy = (pred_classes == targets).float().mean()
        
        return {
            "accuracy": accuracy,
        }

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Training step for a single batch.
        
        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Index of the current batch
            
        Returns:
            Loss tensor
        """
        inputs, targets = batch
        predictions = self.forward(inputs)
        loss = self._compute_loss(predictions, targets)
        
        # Compute metrics
        metrics = self._compute_metrics(predictions, targets)
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        for metric_name, metric_value in metrics.items():
            self.log(
                f"train_{metric_name}",
                metric_value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
        
        # TODO: Add MLFlow logging here in the future
        # self._log_to_mlflow("train", loss, metrics, batch_idx)
        
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """
        Validation step for a single batch.
        
        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Index of the current batch
        """
        inputs, targets = batch
        predictions = self.forward(inputs)
        loss = self._compute_loss(predictions, targets)
        
        # Compute metrics
        metrics = self._compute_metrics(predictions, targets)
        
        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for metric_name, metric_value in metrics.items():
            self.log(
                f"val_{metric_name}",
                metric_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        
        # TODO: Add MLFlow logging here in the future
        # self._log_to_mlflow("val", loss, metrics, batch_idx)

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """
        Test step for a single batch.
        
        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Index of the current batch
        """
        inputs, targets = batch
        predictions = self.forward(inputs)
        loss = self._compute_loss(predictions, targets)
        
        # Compute metrics
        metrics = self._compute_metrics(predictions, targets)
        
        # Log metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        for metric_name, metric_value in metrics.items():
            self.log(
                f"test_{metric_name}",
                metric_value,
                on_step=False,
                on_epoch=True,
            )
        
        # TODO: Add MLFlow logging here in the future
        # self._log_to_mlflow("test", loss, metrics, batch_idx)

    def configure_optimizers(
        self,
    ) -> Optimizer | Dict[str, Any]:
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            Optimizer or dictionary with optimizer and scheduler configuration
        """
        # Select optimizer
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(
                f"Unsupported optimizer: {self.optimizer_name}. "
                "Supported optimizers: 'adam', 'sgd'"
            )
        
        # Configure learning rate scheduler from config
        scheduler_config = get_scheduler_config()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.mode,
            factor=scheduler_config.factor,
            patience=scheduler_config.patience,
            verbose=scheduler_config.verbose,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": scheduler_config.monitor,
            },
        }

    # TODO: Implement MLFlow logging methods
    # def _log_to_mlflow(
    #     self,
    #     stage: str,
    #     loss: torch.Tensor,
    #     metrics: Dict[str, torch.Tensor],
    #     batch_idx: int,
    # ) -> None:
    #     """
    #     Log metrics to MLFlow.
    #
    #     Args:
    #         stage: Training stage ('train', 'val', 'test')
    #         loss: Loss value
    #         metrics: Dictionary of metric names and values
    #         batch_idx: Current batch index
    #     """
    #     pass
    #
    # def on_train_start(self) -> None:
    #     """Initialize MLFlow run at the start of training."""
    #     pass
    #
    # def on_train_end(self) -> None:
    #     """Finalize MLFlow run at the end of training."""
    #     pass
