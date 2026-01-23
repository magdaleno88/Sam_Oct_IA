# Configuration

The SAM-AI project uses a centralized configuration system built with Pydantic. All default values are managed in `sam_ml/config.py`, making it easy to customize behavior without modifying code.

## Overview

The configuration system provides:
- **Centralized defaults**: All default values in one place
- **Type validation**: Automatic validation using Pydantic
- **Environment variable support**: Override defaults via environment variables
- **Hierarchical structure**: Organized by domain (model, training, preprocessing, etc.)

## Quick Start

```python
from sam_ml.config import get_config, get_model_config, get_training_config

# Get full configuration
config = get_config()

# Get specific configuration sections
model_config = get_model_config()
training_config = get_training_config()
```

## Configuration Structure

The configuration is organized into four main sections:

### ModelConfig

Model architecture and optimizer settings:

```python
from sam_ml.config import get_model_config

config = get_model_config()

# Defaults:
# - num_classes: 5
# - learning_rate: 1e-4
# - optimizer: "adam"
# - weight_decay: 1e-4
# - input_shape: (3, 512, 512)
```

### TrainingConfig

Training hyperparameters and paths:

```python
from sam_ml.config import get_training_config

config = get_training_config()

# Defaults:
# - batch_size: 32
# - num_epochs: 50
# - patience: 10
# - data_dir: Path("data/processed/ddr2019")
# - output_dir: Path("outputs")
# - gpus: None
```

### SchedulerConfig

Learning rate scheduler settings:

```python
from sam_ml.config import get_scheduler_config

config = get_scheduler_config()

# Defaults:
# - factor: 0.5
# - patience: 5
# - mode: "min"
# - monitor: "val_loss"
# - verbose: True
```

### PreprocessingConfig

Preprocessing parameters and dataset paths:

```python
from sam_ml.config import get_preprocessing_config

config = get_preprocessing_config()

# Defaults:
# - min_size: 512
# - target_size: (512, 512)
# - ddr2019_raw_img_dir: Path("data/raw/ddr2019/DR_grading/DR_grading")
# - ddr2019_raw_csv_path: Path("data/raw/ddr2019/DR_grading.csv")
# - ddr2019_processed_dir: Path("data/processed/ddr2019")
```

## Using Configuration in Code

### In Model Classes

```python
from sam_ml.config import get_model_config
from sam_ml.modeling.models.base import BaseLightningModel

class MyModel(BaseLightningModel):
    def __init__(self, num_classes=None, learning_rate=None, **kwargs):
        # Get defaults from config
        model_config = get_model_config()
        
        num_classes = num_classes or model_config.num_classes
        learning_rate = learning_rate or model_config.learning_rate
        
        super().__init__(num_classes=num_classes, learning_rate=learning_rate, **kwargs)
```

### In Training Scripts

```python
from sam_ml.config import get_training_config, get_model_config

# Get defaults
training_config = get_training_config()
model_config = get_model_config()

# Use in argparse
parser.add_argument("--batch-size", type=int, default=training_config.batch_size)
parser.add_argument("--num-classes", type=int, default=model_config.num_classes)
```

### In Preprocessing

```python
from sam_ml.config import get_preprocessing_config

config = get_preprocessing_config()

# Use defaults
min_size = config.min_size  # 512
target_size = config.target_size  # (512, 512)
raw_img_dir = config.ddr2019_raw_img_dir
```

## Environment Variables

You can override configuration values using environment variables with the prefix `SAM_`:

```bash
# Model configuration
export SAM_MODEL_NUM_CLASSES=3
export SAM_MODEL_LEARNING_RATE=1e-3
export SAM_MODEL_OPTIMIZER=sgd

# Training configuration
export SAM_TRAINING_BATCH_SIZE=64
export SAM_TRAINING_NUM_EPOCHS=100

# Preprocessing configuration
export SAM_PREPROCESSING_MIN_SIZE=256
export SAM_PREPROCESSING_TARGET_SIZE="256 256"
```

**Note**: Environment variables use uppercase with underscores. The prefix indicates the section:
- `SAM_MODEL_*` for model configuration
- `SAM_TRAINING_*` for training configuration
- `SAM_SCHEDULER_*` for scheduler configuration
- `SAM_PREPROCESSING_*` for preprocessing configuration

## Customizing Configuration

### Programmatic Customization

```python
from sam_ml.config import Config, ModelConfig, TrainingConfig

# Create custom configuration
custom_config = Config(
    model=ModelConfig(
        num_classes=3,
        learning_rate=1e-3,
    ),
    training=TrainingConfig(
        batch_size=64,
        num_epochs=100,
    ),
)
```

### Using .env File

Create a `.env` file in the project root:

```env
SAM_MODEL_NUM_CLASSES=3
SAM_MODEL_LEARNING_RATE=1e-3
SAM_TRAINING_BATCH_SIZE=64
SAM_TRAINING_NUM_EPOCHS=100
```

The configuration system will automatically load these values.

## Validation

The configuration system includes automatic validation:

- **Type checking**: Ensures values are of the correct type
- **Range validation**: Checks that values are within valid ranges
- **Custom validators**: Validates complex structures (e.g., tuple shapes)

**Example validation errors:**
```python
# This will raise a validation error
ModelConfig(num_classes=0)  # num_classes must be > 0
ModelConfig(learning_rate=-1.0)  # learning_rate must be > 0
ModelConfig(input_shape=(3, 0, 512))  # dimensions must be positive
```

## Best Practices

1. **Always use config for defaults**: Don't hardcode default values in your code
2. **Override when needed**: Pass explicit values to override defaults
3. **Use environment variables for deployment**: Set different configs for dev/staging/prod
4. **Validate early**: Let Pydantic catch configuration errors at startup

## Configuration Reference

### ModelConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_classes` | `int` | `5` | Number of output classes |
| `input_shape` | `Tuple[int, int, int]` | `(3, 512, 512)` | Input image shape (channels, height, width) |
| `learning_rate` | `float` | `1e-4` | Learning rate for optimizer |
| `optimizer` | `Literal["adam", "sgd"]` | `"adam"` | Optimizer name |
| `weight_decay` | `float` | `1e-4` | Weight decay (L2 regularization) coefficient |

### TrainingConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `batch_size` | `int` | `32` | Batch size for training |
| `num_epochs` | `int` | `50` | Number of training epochs |
| `patience` | `int` | `10` | Early stopping patience |
| `data_dir` | `Path` | `"data/processed/ddr2019"` | Directory containing processed dataset |
| `output_dir` | `Path` | `"outputs"` | Directory to save checkpoints and logs |
| `gpus` | `int \| None` | `None` | Number of GPUs to use (None for CPU) |

### SchedulerConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `factor` | `float` | `0.5` | Factor by which learning rate is reduced |
| `patience` | `int` | `5` | Number of epochs to wait before reducing LR |
| `mode` | `Literal["min", "max"]` | `"min"` | Mode for ReduceLROnPlateau |
| `monitor` | `str` | `"val_loss"` | Metric to monitor |
| `verbose` | `bool` | `True` | Whether to print LR updates |

### PreprocessingConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `min_size` | `int` | `512` | Minimum image size required to process |
| `target_size` | `Tuple[int, int]` | `(512, 512)` | Target size for resizing |
| `ddr2019_raw_img_dir` | `Path` | `"data/raw/ddr2019/..."` | DDR2019 raw images directory |
| `ddr2019_raw_csv_path` | `Path` | `"data/raw/ddr2019/..."` | DDR2019 raw CSV file |
| `ddr2019_processed_dir` | `Path` | `"data/processed/ddr2019"` | DDR2019 processed output directory |

## Testing

The configuration system includes comprehensive tests. When writing tests:

```python
from sam_ml.config import reset_config

def test_with_custom_config():
    # Reset config for clean state
    reset_config()
    
    # Your test code here
    pass
```

## Migration Guide

If you have code using hardcoded defaults, migrate to use config:

**Before:**
```python
def train_model(num_classes=5, learning_rate=1e-4):
    # ...
```

**After:**
```python
from sam_ml.config import get_model_config

def train_model(num_classes=None, learning_rate=None):
    model_config = get_model_config()
    num_classes = num_classes or model_config.num_classes
    learning_rate = learning_rate or model_config.learning_rate
    # ...
```

This ensures all defaults come from the centralized configuration.
