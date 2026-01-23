# Creating New Models

This guide explains how to create new model architectures and integrate them into the SAM-AI pipeline. It's designed for team members who may not be expert Python/PyTorch developers, making it easy to implement models from research papers and run experiments.

## Table of Contents

- [Overview](#overview)
- [Step 1: Create Your Model Architecture](#step-1-create-your-model-architecture)
- [Step 2: Integrate with BaseLightningModel](#step-2-integrate-with-baselightningmodel)
- [Step 3: Create a Training Script](#step-3-create-a-training-script)
- [Step 4: Register Your Model](#step-4-register-your-model)
- [Step 5: Run Your Experiment](#step-5-run-your-experiment)
- [Complete Example](#complete-example)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

The SAM-AI project uses a **model registry system** that allows you to:
- Create your model implementation in isolation
- Register it with a unique key
- Run experiments with minimal code changes
- Share models with the team easily

**Key Concept**: You only need to:
1. Create your model architecture (PyTorch `nn.Module`)
2. Wrap it in `BaseLightningModel`
3. Register it with a unique key
4. Use that key in your training script

The pipeline handles everything else automatically!

## Step 1: Create Your Model Architecture

Start by creating a PyTorch model architecture. This is your core model design based on the research paper you're implementing.

### Example: Simple CNN Model

Create a file `sam_ml/modeling/models/simple_cnn.py`:

```python
"""Simple CNN model for diabetic retinopathy detection."""

from typing import Tuple

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Simple convolutional neural network for image classification.
    
    Architecture:
    - 3 convolutional blocks
    - 2 fully connected layers
    - Output layer with num_classes neurons
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (3, 512, 512),  # (channels, height, width)
        num_classes: int = 5,
    ) -> None:
        """
        Initialize the SimpleCNN model.
        
        Args:
            input_shape: Shape of input images (channels, height, width)
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 512x512 -> 256x256
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 256x256 -> 128x128
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128x128 -> 64x64
        )
        
        # Calculate flattened size: 128 channels * 64 * 64 = 524,288
        self.flattened_size = 128 * 64 * 64
        
        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        # Output layer
        self.output = nn.Linear(256, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, flattened_size)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        
        # Output layer
        x = self.output(x)
        
        return x
```

### Key Points for Your Model

1. **Inherit from `nn.Module`**: Your model must inherit from `torch.nn.Module`
2. **Implement `__init__`**: Define all layers here
3. **Implement `forward`**: Define the forward pass
4. **Type Hints**: Always include type hints for parameters and return types
5. **Docstrings**: Document what your model does

## Step 2: Integrate with BaseLightningModel

Now wrap your model architecture in `BaseLightningModel` to get training, validation, and logging capabilities automatically.

Create a file `sam_ml/modeling/models/simple_cnn_lightning.py`:

```python
"""PyTorch Lightning wrapper for SimpleCNN model."""

from typing import Any, Tuple

import torch
from pytorch_lightning import LightningModule

from sam_ml.modeling.models.base import BaseLightningModel
from sam_ml.modeling.models.simple_cnn import SimpleCNN


class SimpleCNNLightning(BaseLightningModel):
    """
    PyTorch Lightning wrapper for SimpleCNN model.
    
    This class integrates SimpleCNN with the training pipeline,
    providing automatic logging, optimization, and validation.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        learning_rate: float = 1e-4,
        optimizer: str = "adam",
        weight_decay: float = 1e-4,
        input_shape: Tuple[int, int, int] = (3, 512, 512),
        **kwargs: Any,
    ) -> None:
        """
        Initialize SimpleCNNLightning model.
        
        Args:
            num_classes: Number of output classes
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer name ('adam' or 'sgd')
            weight_decay: Weight decay (L2 regularization) coefficient
            input_shape: Shape of input images (channels, height, width)
            **kwargs: Additional arguments passed to BaseLightningModel
        """
        # Store custom parameters before calling super().__init__()
        self.input_shape = input_shape
        
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
```

### What You Get Automatically

By inheriting from `BaseLightningModel`, you automatically get:
- ✅ Training step with loss computation and logging
- ✅ Validation step with metrics
- ✅ Test step
- ✅ Optimizer configuration (Adam/SGD)
- ✅ Learning rate scheduler
- ✅ Metric logging (loss, accuracy)
- ✅ Hyperparameter saving

### Optional: Customize Loss or Metrics

If you need custom loss or metrics, override these methods:

```python
def _compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Custom loss computation (optional)."""
    # Default is CrossEntropyLoss, but you can customize:
    return self.criterion(predictions, targets)

def _compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Custom metrics computation (optional)."""
    # Default includes accuracy, but you can add more:
    metrics = super()._compute_metrics(predictions, targets)
    # Add your custom metrics here
    return metrics
```

## Step 3: Create a Training Script

Create a training script that uses your model. This script will be specific to your experiment.

Create a file `sam_ml/modeling/train_simple_cnn.py`:

```python
"""Training script for SimpleCNN model."""

from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from sam_ml.modeling.models.simple_cnn_lightning import SimpleCNNLightning
# TODO: Import your dataset class when available
# from sam_ml.datasets import DiabeticRetinopathyDataset


def train_simple_cnn(
    data_dir: str = "data/processed/ddr2019",
    batch_size: int = 32,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    num_classes: int = 5,
    output_dir: str = "outputs/simple_cnn",
    gpus: Optional[int] = None,
) -> None:
    """
    Train SimpleCNN model.
    
    Args:
        data_dir: Directory containing processed dataset
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        num_classes: Number of output classes
        output_dir: Directory to save model checkpoints and logs
        gpus: Number of GPUs to use (None for CPU)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # TODO: Create datasets when dataset classes are available
    # For now, this is a template
    # train_dataset = DiabeticRetinopathyDataset(
    #     data_dir=data_dir,
    #     split="train",
    # )
    # val_dataset = DiabeticRetinopathyDataset(
    #     data_dir=data_dir,
    #     split="val",
    # )
    # 
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=4,
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=4,
    # )
    
    # Create model
    model = SimpleCNNLightning(
        num_classes=num_classes,
        learning_rate=learning_rate,
        optimizer="adam",
        weight_decay=1e-4,
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path / "checkpoints",
        filename="simple-cnn-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=10,
        verbose=True,
    )
    
    # Setup logger
    logger = CSVLogger(
        save_dir=output_path,
        name="logs",
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        accelerator="gpu" if gpus else "cpu",
        devices=gpus if gpus else 1,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    
    # Train model
    # trainer.fit(model, train_loader, val_loader)
    print("Training script ready. Uncomment dataset creation and trainer.fit() when datasets are available.")


if __name__ == "__main__":
    train_simple_cnn(
        data_dir="data/processed/ddr2019",
        batch_size=32,
        num_epochs=50,
        learning_rate=1e-4,
    )
```

## Step 4: Register Your Model

To make your model available in the pipeline, register it in the model registry.

### Create Model Registry

Create a file `sam_ml/modeling/models/registry.py`:

```python
"""Model registry for managing all available models."""

from typing import Any, Callable, Dict, Type

from sam_ml.modeling.models.base import BaseLightningModel


# Model registry dictionary
# Key: unique model identifier (string)
# Value: function that creates the model instance
MODEL_REGISTRY: Dict[str, Callable[..., BaseLightningModel]] = {}


def register_model(key: str) -> Callable:
    """
    Decorator to register a model in the registry.
    
    Usage:
        @register_model("simple_cnn")
        def create_simple_cnn(**kwargs) -> BaseLightningModel:
            return SimpleCNNLightning(**kwargs)
    
    Args:
        key: Unique identifier for the model
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., BaseLightningModel]) -> Callable:
        if key in MODEL_REGISTRY:
            raise ValueError(f"Model key '{key}' is already registered!")
        MODEL_REGISTRY[key] = func
        return func
    return decorator


def get_model(key: str, **kwargs: Any) -> BaseLightningModel:
    """
    Get a model instance from the registry.
    
    Args:
        key: Model identifier
        **kwargs: Arguments to pass to model constructor
        
    Returns:
        Model instance
        
    Raises:
        KeyError: If model key is not found
    """
    if key not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise KeyError(
            f"Model '{key}' not found in registry. "
            f"Available models: {available}"
        )
    
    return MODEL_REGISTRY[key](**kwargs)


def list_models() -> list[str]:
    """
    List all registered model keys.
    
    Returns:
        List of model keys
    """
    return list(MODEL_REGISTRY.keys())
```

### Register Your Model

Update `sam_ml/modeling/models/simple_cnn_lightning.py` to register it:

```python
"""PyTorch Lightning wrapper for SimpleCNN model."""

from typing import Any, Tuple

import torch
from pytorch_lightning import LightningModule

from sam_ml.modeling.models.base import BaseLightningModel
from sam_ml.modeling.models.registry import register_model
from sam_ml.modeling.models.simple_cnn import SimpleCNN


@register_model("simple_cnn")
def create_simple_cnn(
    num_classes: int = 5,
    learning_rate: float = 1e-4,
    optimizer: str = "adam",
    weight_decay: float = 1e-4,
    input_shape: Tuple[int, int, int] = (3, 512, 512),
    **kwargs: Any,
) -> "SimpleCNNLightning":
    """
    Factory function to create SimpleCNNLightning model.
    
    This function is registered in the model registry with key "simple_cnn".
    """
    return SimpleCNNLightning(
        num_classes=num_classes,
        learning_rate=learning_rate,
        optimizer=optimizer,
        weight_decay=weight_decay,
        input_shape=input_shape,
        **kwargs,
    )


class SimpleCNNLightning(BaseLightningModel):
    # ... (same as before)
```

### Update Model Exports

Update `sam_ml/modeling/models/__init__.py`:

```python
"""Model architectures for diabetic retinopathy detection."""

from sam_ml.modeling.models.base import BaseLightningModel
from sam_ml.modeling.models.registry import get_model, list_models, register_model

# Import models to register them
from sam_ml.modeling.models import simple_cnn_lightning  # noqa: F401

__all__ = [
    "BaseLightningModel",
    "get_model",
    "list_models",
    "register_model",
]
```

## Step 5: Run Your Experiment

Now you can use your model in a unified training pipeline!

### Option 1: Use Registry in Training Script

Update your training script to use the registry:

```python
from sam_ml.modeling.models import get_model

# Instead of:
# model = SimpleCNNLightning(...)

# Use:
model = get_model("simple_cnn", num_classes=5, learning_rate=1e-4)
```

### Option 2: Unified Training Pipeline

Create a unified training script `sam_ml/modeling/train.py`:

```python
"""Unified training script for all models."""

import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from sam_ml.modeling.models import get_model, list_models


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train a model")
    
    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list_models(),
        help=f"Model to train. Available: {', '.join(list_models())}",
    )
    
    # Model hyperparameters
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    
    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--data-dir", type=str, default="data/processed/ddr2019")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--gpus", type=int, default=None)
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir) / args.model
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get model from registry
    model = get_model(
        args.model,
        num_classes=args.num_classes,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path / "checkpoints",
        filename=f"{args.model}-{{epoch:02d}}-{{val_loss:.2f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=10,
    )
    
    # Setup logger
    logger = CSVLogger(
        save_dir=output_path,
        name="logs",
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        accelerator="gpu" if args.gpus else "cpu",
        devices=args.gpus if args.gpus else 1,
    )
    
    # TODO: Load datasets and train
    # trainer.fit(model, train_loader, val_loader)
    print(f"Model '{args.model}' ready for training!")


if __name__ == "__main__":
    main()
```

### Run Your Experiment

You can run experiments using the training script shortcut:

```bash
# List available models and options
uv run train-model --help

# Train your model
uv run train-model \
    --model simple_cnn \
    --num-classes 5 \
    --learning-rate 1e-4 \
    --batch-size 32 \
    --num-epochs 50

# Train with custom optimizer
uv run train-model \
    --model simple_cnn \
    --optimizer sgd \
    --weight-decay 0.01 \
    --num-epochs 100
```

**Alternative**: You can also use the Python module directly:

```bash
uv run python -m sam_ml.modeling.train --model simple_cnn --num-classes 5
```

## Complete Example

Here's a complete example showing all steps together:

### 1. Model Architecture (`simple_cnn.py`)
```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 256 * 256, num_classes)
    
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

### 2. Lightning Wrapper (`simple_cnn_lightning.py`)
```python
from sam_ml.modeling.models.base import BaseLightningModel
from sam_ml.modeling.models.registry import register_model
from sam_ml.modeling.models.simple_cnn import SimpleCNN

@register_model("simple_cnn")
def create_simple_cnn(**kwargs):
    return SimpleCNNLightning(**kwargs)

class SimpleCNNLightning(BaseLightningModel):
    def _create_model(self) -> None:
        self.model = SimpleCNN(num_classes=self.num_classes)
```

### 3. Run Experiment
```bash
# Using the script shortcut (recommended)
uv run train-model --model simple_cnn

# Or using Python module directly
uv run python -m sam_ml.modeling.train --model simple_cnn
```

That's it! Your model is now integrated into the pipeline.

## Best Practices

### 1. Naming Conventions

- **Model architecture**: `snake_case.py` (e.g., `simple_cnn.py`)
- **Lightning wrapper**: `snake_case_lightning.py` (e.g., `simple_cnn_lightning.py`)
- **Registry key**: `snake_case` (e.g., `"simple_cnn"`)
- **Training script**: `train_<model_name>.py` (e.g., `train_simple_cnn.py`)

### 2. File Organization

```
sam_ml/modeling/models/
├── base.py                    # BaseLightningModel (don't modify)
├── registry.py                # Model registry (don't modify)
├── simple_cnn.py              # Your model architecture
├── simple_cnn_lightning.py   # Your Lightning wrapper
└── __init__.py               # Exports
```

### 3. Documentation

Always document:
- What your model does
- What paper it's based on
- Key hyperparameters
- Expected input/output shapes

### 4. Testing

Create a simple test to verify your model works:

```python
def test_simple_cnn():
    from sam_ml.modeling.models import get_model
    
    model = get_model("simple_cnn", num_classes=5)
    x = torch.randn(2, 3, 512, 512)  # batch_size=2
    y = model(x)
    assert y.shape == (2, 5)
```

### 5. Version Control

- Commit your model files
- Use descriptive commit messages: "Add SimpleCNN model from Paper X"
- Don't commit large model checkpoints (use `.gitignore`)

## Troubleshooting

### Model Not Found

**Error**: `KeyError: Model 'my_model' not found in registry`

**Solution**: 
1. Make sure you imported your model module in `__init__.py`
2. Check that you used `@register_model("my_model")` decorator
3. Verify the key matches exactly (case-sensitive)

### Shape Mismatch Errors

**Error**: `RuntimeError: Expected input shape (X, Y, Z) but got (A, B, C)`

**Solution**:
1. Check your input data shape
2. Verify model expects correct input shape
3. Add shape checks in `forward()` method

### Import Errors

**Error**: `ImportError: cannot import name 'MyModel'`

**Solution**:
1. Check file names match import statements
2. Verify `__init__.py` exports your model
3. Make sure you're in the correct directory

### Training Not Starting

**Error**: Model loads but training doesn't start

**Solution**:
1. Check that datasets are properly loaded
2. Verify DataLoader is created correctly
3. Ensure `trainer.fit()` is called with correct arguments

## Next Steps

1. **Create your model architecture** following Step 1
2. **Wrap it in BaseLightningModel** following Step 2
3. **Register it** following Step 4
4. **Test it** with a simple script
5. **Run experiments** using the unified training pipeline

For questions or help, refer to:
- [Modeling Documentation](modeling.md) - Base model API
- [Testing Documentation](testing.md) - How to test your model
- [Code Style Documentation](code-style.md) - Coding standards
