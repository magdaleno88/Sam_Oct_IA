# SAM - AI

A computer vision project to aid in the diagnosis of diabetic retinopathy using deep learning and fundus scan analysis.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Testing](#testing)
- [Preprocessing](#preprocessing)
- [Project Structure](#project-structure)
- [Code Style](#code-style)
- [References](#references)

## Quick Start

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Verify installation:**
   ```bash
   uv run python -c "import torch; print(f'Torch: {torch.__version__}')"
   ```

## Installation

This project uses `uv` for package management. Install dependencies:

```bash
uv sync
```

**Note:** PyTorch is configured with platform-specific versions:
- macOS Intel (x86_64): PyTorch 2.2.2 (latest version with Intel support)
- Other platforms: PyTorch 2.2.2 or newer

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
uv run pytest tests/test_preprocess_ddr2019.py
uv run pytest tests/test_preprocessing_router.py
```

### Run with coverage

```bash
uv run pytest --cov=sam_ml --cov-report=html
```

### Additional options

```bash
# Quiet mode (minimal output)
uv run pytest -q

# Run specific test by name pattern
uv run pytest -k "test_resize"

# Run only fast tests (exclude slow markers)
uv run pytest -m "not slow"
```

For more details, see [Tests Documentation](tests/README.md).

## Preprocessing

The project includes a preprocessing module for preparing diabetic retinopathy datasets. Currently supports the DDR2019 dataset.

### Quick Start

Process the DDR2019 dataset using the CLI:

```bash
# Process with default settings (min-size=512, target-size=512x512)
uv run preprocess-dataset ddr2019

# Process with custom minimum size and target size
uv run preprocess-dataset ddr2019 --min-size 512 --target-size 512 512

# Process with custom paths
uv run preprocess-dataset ddr2019 \
  --raw-img-dir data/raw/ddr2019/DR_grading/DR_grading \
  --raw-csv-path data/raw/ddr2019/DR_grading.csv \
  --processed-dir data/processed/ddr2019
```

### Features

- **Minimum Size Filtering**: Only processes images with both dimensions >= 512x512
- **Automatic Padding**: Non-square images are padded to square (black padding)
- **No Upscaling**: Images are only downscaled or kept at same size (never upscaled to avoid noise)
- **Standardized Output**: All processed images are resized to 512x512
- **Label Synchronization**: CSV labels are automatically filtered to match processed images
- **Original Data Protection**: Original dataset files are never modified

### Available Datasets

- `ddr2019` - DDR2019 Diabetic Retinopathy dataset

### Preprocessing Behavior

The preprocessing pipeline:

1. **Filter by Minimum Size**: Only images with `width >= 512 AND height >= 512` are processed
2. **Pad Non-Square Images**: Asymmetric images are padded with black pixels to make them square
3. **Resize to Target Size**: All images are resized to 512x512 (downscaling only, never upscaling)
4. **Filter Labels**: CSV labels are filtered to only include processed images

**Important**: Images smaller than 512x512 are skipped to avoid upscaling, which would introduce noise.

### Output Structure

After preprocessing, the dataset will be organized as:

```
data/processed/ddr2019/
├── images/
│   ├── 20170413102628830.jpg  (all 512x512)
│   └── ...
└── labels.csv
```

The `labels.csv` file contains:
- `filename`: Image filename
- `label`: Diagnosis label (0-4)

### Processing Statistics

At the end of preprocessing, the script prints:
- **Original dataset**: Total number of images in the raw dataset
- **Processed dataset**: Number of images successfully processed
- **Images skipped**: Number of images that were too small or would require upscaling

Example output:
```
Preprocessing complete for ddr2019:
  - Original dataset: 12524 images
  - Processed dataset: 12524 images
  - Images skipped: 0 images (too small or would require upscaling)
  - Labels saved to: data/processed/ddr2019/labels.csv
```

For detailed usage instructions, see [Preprocessing Documentation](sam_ml/preprocessing/README.md).

## Project Structure

```
sam-ai/
├── data/
│   ├── raw/              # Raw, unmodified datasets
│   └── processed/        # Processed datasets ready for training
├── sam_ml/
│   ├── preprocessing/    # Dataset preprocessing scripts
│   ├── datasets/        # PyTorch Dataset classes
│   └── modeling/        # Model training and prediction
├── tests/                # Unit tests
├── notebooks/            # Jupyter notebooks for exploration
└── docs/                 # Documentation and research papers
```

## Code Style

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

### Import Organization

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
mypy sam_ml/ --ignore-missing-imports
```

### Additional Guidelines

- **Prefer explicit over implicit**: Use type hints even when types seem obvious
- **Use `Optional[T]` instead of `Union[T, None]`**: More readable and idiomatic
- **Use `Tuple` for fixed-size sequences**: `Tuple[int, int, int]` for (height, width, channels)
- **Use `List[T]` for variable-length sequences**: `List[torch.Tensor]` for lists of tensors
- **Type hint lambda functions** when used in complex contexts
- **Avoid `Any` type**: Use more specific types when possible

### Example: Fully Typed Function

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

## References

- [Identification of Diabetic Retinopathy Using Weighted Fusion Deep Learning Based on Dual-Channel Fundus Scans](https://www.mdpi.com/2075-4418/12/2/540)
