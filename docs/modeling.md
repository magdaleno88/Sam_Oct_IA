# Modeling

The modeling module provides a foundation for training and evaluating deep learning models for diabetic retinopathy detection using PyTorch Lightning.

## Base Model Class

The project includes a `BaseLightningModel` class that provides a standardized foundation for all model implementations.

### Quick Start

```python
from sam_ml.modeling.models import BaseLightningModel
import torch.nn as nn

class MyModel(BaseLightningModel):
    def _create_model(self) -> None:
        """Create the model architecture."""
        self.model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_classes),
        )

# Create and use the model
model = MyModel(num_classes=5, learning_rate=1e-4)
```

### Features

- **PyTorch Lightning Integration**: Built on PyTorch Lightning for easy training
- **Standardized Training Loop**: Consistent training/validation/test steps
- **Automatic Logging**: Built-in metric logging with proper flags
- **Optimizer Configuration**: Easy configuration of optimizers and schedulers
- **MLFlow Ready**: Placeholder methods for future MLFlow integration

## Base Model API

### Initialization

```python
model = BaseLightningModel(
    num_classes: int = 5,
    learning_rate: float = 1e-4,
    optimizer: str = "adam",
    weight_decay: float = 0.0,
    **kwargs: Any,
)
```

**Parameters:**
- `num_classes`: Number of output classes for classification
- `learning_rate`: Learning rate for optimizer
- `optimizer`: Optimizer name ('adam' or 'sgd')
- `weight_decay`: Weight decay (L2 regularization) coefficient
- `**kwargs`: Additional arguments passed to subclasses

### Required Methods

Subclasses must implement:

- `_create_model()`: Define the model architecture

### Optional Overrides

Subclasses can override:

- `forward()`: Forward pass (default uses `self.model`)
- `_compute_loss()`: Loss computation (default: CrossEntropyLoss)
- `_compute_metrics()`: Metric computation (default: accuracy)

### Supported Optimizers

- `adam`: Adam optimizer (default)
- `sgd`: SGD optimizer with momentum=0.9

### Learning Rate Scheduler

The base model automatically configures a `ReduceLROnPlateau` scheduler that:
- Monitors validation loss
- Reduces learning rate by factor of 0.5
- Waits 5 epochs before reducing
- Uses minimum mode (reduces when loss stops decreasing)

## Training with PyTorch Lightning

```python
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

# Create model
model = MyModel(num_classes=5)

# Create trainer
trainer = Trainer(
    max_epochs=10,
    callbacks=[ModelCheckpoint(monitor="val_loss")],
)

# Train
trainer.fit(model, train_dataloader, val_dataloader)
```

## Metrics

The base model automatically computes and logs:

- **Loss**: Cross-entropy loss (configurable)
- **Accuracy**: Classification accuracy

Metrics are logged with appropriate prefixes:
- `train_loss`, `train_accuracy` for training
- `val_loss`, `val_accuracy` for validation
- `test_loss`, `test_accuracy` for testing

## Future: MLFlow Integration

The base model includes placeholder methods for MLFlow integration:

- `_log_to_mlflow()`: Log metrics to MLFlow
- `on_train_start()`: Initialize MLFlow run
- `on_train_end()`: Finalize MLFlow run

These will be implemented in future updates.

## Best Practices

1. **Always implement `_create_model()`**: This is required and defines your architecture
2. **Use hyperparameters**: The base class automatically saves hyperparameters
3. **Override metrics if needed**: Customize `_compute_metrics()` for additional metrics
4. **Test your model**: Use the provided test fixtures to verify your implementation

## Example: Complete Model Implementation

```python
from sam_ml.modeling.models import BaseLightningModel
import torch
import torch.nn as nn

class SimpleClassifier(BaseLightningModel):
    """Simple classifier for diabetic retinopathy."""
    
    def _create_model(self) -> None:
        """Create a simple feedforward network."""
        self.model = nn.Sequential(
            nn.Linear(512 * 512 * 3, 1024),  # Assuming flattened 512x512 RGB images
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with flattening."""
        # Flatten image if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.model(x)

# Usage
model = SimpleClassifier(
    num_classes=5,
    learning_rate=1e-3,
    optimizer="adam",
    weight_decay=1e-4,
)
```

## Testing

The base model includes comprehensive tests. When creating new models, ensure they:

1. Inherit from `BaseLightningModel`
2. Implement `_create_model()`
3. Pass all base model tests
4. Add model-specific tests as needed

See [Testing Documentation](testing.md) for more details.
