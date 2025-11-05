# SAM - AI

A computer vision project to aid in the diagnosis of diabetic retinopathy using deep learning and dual-channel fundus scan analysis.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Testing](#testing)
- [Project Overview](#project-overview)
- [Documentation](#documentation)
- [Dataset Structure](#dataset-structure)
- [References](#references)

## Quick Start

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Verify installation:**
   ```bash
   python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
   ```

For detailed installation instructions, see [Installation Guide](docs/installation.md).

## Installation

This project uses `uv` for package management. Install with CPU support (default):

```bash
uv sync
```

For GPU support (CUDA or MPS), see [Installation Guide](docs/installation.md) for detailed setup instructions.

## Testing

This project includes comprehensive unit tests using pytest. Run tests using `uv run`:

### Install test dependencies

```bash
uv sync --extra test
```

### Run all tests

```bash
uv run pytest
```

The default configuration (from `pyproject.toml`) provides verbose output and proper test discovery, so no additional parameters are needed.

### Run specific test files

```bash
uv run pytest tests/test_fusion_layers.py
uv run pytest tests/test_dual_channel_model.py
```

### Run with coverage

```bash
uv run pytest --cov=mlops_project --cov-report=html
```

### Additional options

```bash
# Quiet mode (minimal output)
uv run pytest -q

# Run specific test by name pattern
uv run pytest -k "test_fusion_layer"

# Run only fast tests (exclude slow markers)
uv run pytest -m "not slow"
```

For more details, see [Tests Documentation](tests/README.md).

## Project Overview

SAM-AI implements a dual-channel weighted fusion deep learning model for diabetic retinopathy detection, based on the research paper listed in [References](#references).

### Key Features

- Dual-channel architecture for fundus scan analysis
- Weighted fusion mechanism for feature combination
- Pre-trained CNN backbones: Inception V3 (CLAHE) and VGG-16 (CECED) as per paper
- Simple, direct implementation matching the paper's architecture
- Object-oriented design with Keras/TensorFlow

## Documentation

- **[Installation Guide](docs/installation.md)** - Detailed setup instructions for CPU, CUDA, and MPS
- **[Project Structure](docs/project-structure.md)** - Complete directory layout and module descriptions
- **[Model Documentation](mlops_project/modeling/models/README.md)** - Model architecture and usage examples
- **[Tests Documentation](tests/README.md)** - Testing guide and pytest usage
- **[Dataset Structure](data/README.md)** - Expected dataset organization and loading instructions

## Dataset Structure

The project expects a specific dataset structure for training the dual-channel model. See [Dataset Structure Documentation](data/README.md) for:
- Required folder organization
- CLAHE and CECED preprocessing channels
- 5-class diabetic retinopathy severity levels
- TensorFlow dataset loading examples

⚠️ **Status**: Dataset structure is planned but pending implementation.

## References

- [Identification of Diabetic Retinopathy Using Weighted Fusion Deep Learning Based on Dual-Channel Fundus Scans](https://www.mdpi.com/2075-4418/12/2/540)

## Model Architecture

The model implements the dual-channel architecture from the paper:
- **Channel 1 (CLAHE)**: Uses Inception V3 for feature extraction
- **Channel 2 (CECED)**: Uses VGG-16 for feature extraction  
- Features are fused using a learnable weighted fusion mechanism
- Simple, direct implementation in a single module (`dual_channel_model.py`)

See [Model Documentation](mlops_project/modeling/models/README.md) for detailed architecture and usage examples.

## Code Style and Styling Rules

This project follows Python typing standards and modern Python best practices. All code should adhere to the following guidelines:

### Type Hints

- **All functions and methods must have type hints** for parameters and return types
- Use `typing` module for complex types (`List`, `Tuple`, `Optional`, `Union`, `Dict`, etc.)
- Use built-in types for simple cases (`int`, `str`, `bool`, `float`)
- Use `Optional[T]` for parameters that can be `None`
- Use `-> None` for functions that don't return a value

**Example:**
```python
from typing import List, Optional, Tuple
import tensorflow as tf

def process_images(
    images: List[tf.Tensor],
    batch_size: int = 32,
    normalize: bool = True
) -> tf.Tensor:
    """Process a list of images."""
    # Implementation
    pass
```

### Class Attributes

- **Type hint all class attributes** that are set in `__init__`
- Use `Optional[T]` for attributes that may be `None` initially

**Example:**
```python
class MyModel:
    def __init__(self, num_classes: int) -> None:
        self.num_classes: int = num_classes
        self.weights: Optional[tf.Variable] = None
```

### Test Functions

- **All test functions must have type hints** for fixtures and parameters
- Use `-> None` for all test functions
- Type hint pytest fixtures with return types

**Example:**
```python
@pytest.fixture
def model() -> MyModel:
    return MyModel(num_classes=5)

def test_initialization(model: MyModel) -> None:
    assert model.num_classes == 5
```

### Import Organization

- **Standard library imports first**
- **Third-party imports second** (TensorFlow, NumPy, etc.)
- **Local imports last**
- Use absolute imports for project modules

**Example:**
```python
from typing import List, Optional
from pathlib import Path

import pytest
import tensorflow as tf
import numpy as np

from mlops_project.modeling.models import MyModel
```

### Documentation

- **All public functions, classes, and methods must have docstrings**
- Use Google-style docstrings with `Args:` and `Returns:` sections
- Include type information in docstrings when helpful for clarity

### Code Formatting

- Use **4 spaces** for indentation (no tabs)
- Maximum line length: **100 characters** (soft limit, 120 hard limit)
- Use **double quotes** for strings (consistent with Python conventions)
- Add trailing commas in multi-line function calls and lists

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `DualChannelModel`)
- **Functions and methods**: `snake_case` (e.g., `process_images`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_BATCH_SIZE`)
- **Private methods**: prefix with `_` (e.g., `_create_backbone`)
- **Type variables**: `PascalCase` (e.g., `T`, `K`, `V`)

### Type Checking Tools

This project uses Python's built-in type system. For static type checking, you can use:

- `mypy` - Static type checker
- `pyright` - Fast type checker (used by Pylance in VS Code)

**Example mypy configuration:**
```bash
mypy mlops_project/ --ignore-missing-imports
```

### Additional Guidelines

- **Prefer explicit over implicit**: Use type hints even when types seem obvious
- **Use `Optional[T]` instead of `Union[T, None]`**: More readable and idiomatic
- **Use `Tuple` for fixed-size sequences**: `Tuple[int, int, int]` for (height, width, channels)
- **Use `List[T]` for variable-length sequences**: `List[tf.Tensor]` for lists of tensors
- **Type hint lambda functions** when used in complex contexts
- **Avoid `Any` type**: Use more specific types when possible

### Example: Fully Typed Function

```python
from typing import List, Optional, Tuple
import tensorflow as tf
from tensorflow import keras

def create_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 5,
    backbone: str = "resnet50"
) -> keras.Model:
    """
    Create a deep learning model.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
        backbone: Name of the backbone architecture
        
    Returns:
        Compiled Keras model
    """
    # Implementation
    pass
```

These styling rules ensure code consistency, improve readability, and enable better IDE support and static type checking.
