# Code Style Guide

This project follows Python typing standards and modern Python best practices. All code should adhere to the following guidelines.

## Type Hints

### Functions and Methods

- **All functions and methods must have type hints** for parameters and return types
- Use `typing` module for complex types (`List`, `Tuple`, `Optional`, `Union`, `Dict`, etc.)
- Use built-in types for simple cases (`int`, `str`, `bool`, `float`)
- Use `Optional[T]` for parameters that can be `None`
- Use `-> None` for functions that don't return a value

**Example:**
```python
from typing import List, Optional, Tuple
import torch

def process_images(
    images: List[torch.Tensor],
    batch_size: int = 32,
    normalize: bool = True
) -> torch.Tensor:
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
        self.weights: Optional[torch.Tensor] = None
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

## Import Organization

- **Standard library imports first**
- **Third-party imports second** (PyTorch, NumPy, etc.)
- **Local imports last**
- Use absolute imports for project modules

**Example:**
```python
from typing import List, Optional
from pathlib import Path

import pytest
import torch
import numpy as np

from sam_ml.modeling.models import MyModel
```

## Documentation

- **All public functions, classes, and methods must have docstrings**
- Use Google-style docstrings with `Args:` and `Returns:` sections
- Include type information in docstrings when helpful for clarity

**Example:**
```python
def create_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 5,
    backbone: str = "resnet50"
) -> nn.Module:
    """
    Create a deep learning model.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
        backbone: Name of the backbone architecture
        
    Returns:
        PyTorch module
    """
    # Implementation
    pass
```

## Code Formatting

- Use **4 spaces** for indentation (no tabs)
- Maximum line length: **100 characters** (soft limit, 120 hard limit)
- Use **double quotes** for strings (consistent with Python conventions)
- Add trailing commas in multi-line function calls and lists

## Naming Conventions

- **Classes**: `PascalCase` (e.g., `DualChannelModel`)
- **Functions and methods**: `snake_case` (e.g., `process_images`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_BATCH_SIZE`)
- **Private methods**: prefix with `_` (e.g., `_create_backbone`)
- **Type variables**: `PascalCase` (e.g., `T`, `K`, `V`)

## Type Checking Tools

This project uses Python's built-in type system. For static type checking, you can use:

- `mypy` - Static type checker
- `pyright` - Fast type checker (used by Pylance in VS Code)

**Example mypy configuration:**
```bash
mypy sam_ml/ --ignore-missing-imports
```

## Additional Guidelines

- **Prefer explicit over implicit**: Use type hints even when types seem obvious
- **Use `Optional[T]` instead of `Union[T, None]`**: More readable and idiomatic
- **Use `Tuple` for fixed-size sequences**: `Tuple[int, int, int]` for (height, width, channels)
- **Use `List[T]` for variable-length sequences**: `List[torch.Tensor]` for lists of tensors
- **Type hint lambda functions** when used in complex contexts
- **Avoid `Any` type**: Use more specific types when possible

## Example: Fully Typed Function

```python
from typing import List, Optional, Tuple
import torch
import torch.nn as nn

def create_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 5,
    backbone: str = "resnet50"
) -> nn.Module:
    """
    Create a deep learning model.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
        backbone: Name of the backbone architecture
        
    Returns:
        PyTorch module
    """
    # Implementation
    pass
```

These styling rules ensure code consistency, improve readability, and enable better IDE support and static type checking.
