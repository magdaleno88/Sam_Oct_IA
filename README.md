# SAM - AI

A computer vision project to aid in the diagnosis of diabetic retinopathy using deep learning and fundus scan analysis.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Testing](#testing)
- [Preprocessing](#preprocessing)
- [Modeling](#modeling)
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

3. **Run tests:**
   ```bash
   uv sync --extra test
   uv run pytest
   ```

## Installation

This project uses `uv` for package management. Install dependencies:

```bash
uv sync
```

**Note:** PyTorch is configured with platform-specific versions:
- macOS Intel (x86_64): PyTorch 2.2.2 (latest version with Intel support)
- Other platforms: PyTorch 2.2.2 or newer

For detailed installation instructions, troubleshooting, and platform-specific notes, see [Installation Documentation](docs/installation.md).

## Testing

This project includes comprehensive unit tests using pytest. Run tests using `uv run`:

```bash
# Install test dependencies
uv sync --extra test

# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=sam_ml --cov-report=html
```

For detailed testing documentation, including running specific tests, using markers, and writing new tests, see [Testing Documentation](docs/testing.md).

## Preprocessing

The project includes a preprocessing module for preparing diabetic retinopathy datasets. Currently supports the DDR2019 dataset.

### Quick Start

Process the DDR2019 dataset using the CLI:

```bash
# Process with default settings (min-size=512, target-size=512x512)
uv run preprocess-dataset ddr2019

# Process with custom settings
uv run preprocess-dataset ddr2019 --min-size 512 --target-size 512 512
```

### Features

- **Minimum Size Filtering**: Only processes images with both dimensions >= 512x512
- **Automatic Padding**: Non-square images are padded to square (black padding)
- **No Upscaling**: Images are only downscaled or kept at same size (never upscaled to avoid noise)
- **Standardized Output**: All processed images are resized to 512x512
- **Label Synchronization**: CSV labels are automatically filtered to match processed images
- **Original Data Protection**: Original dataset files are never modified

For complete preprocessing documentation, including detailed usage, pipeline explanation, and troubleshooting, see [Preprocessing Documentation](docs/preprocessing.md).

## Modeling

The modeling module provides a foundation for training and evaluating deep learning models using PyTorch Lightning.

### Quick Start

```python
from sam_ml.modeling.models import BaseLightningModel
import torch.nn as nn

class MyModel(BaseLightningModel):
    def _create_model(self) -> None:
        self.model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_classes),
        )

model = MyModel(num_classes=5, learning_rate=1e-4)
```

### Features

- **PyTorch Lightning Integration**: Built on PyTorch Lightning for easy training
- **Standardized Training Loop**: Consistent training/validation/test steps
- **Automatic Logging**: Built-in metric logging with proper flags
- **Optimizer Configuration**: Easy configuration of optimizers and schedulers
- **MLFlow Ready**: Placeholder methods for future MLFlow integration

For complete modeling documentation, including API reference, examples, and best practices, see [Modeling Documentation](docs/modeling.md).

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
│       └── models/      # Model architectures
├── tests/                # Unit tests
├── notebooks/            # Jupyter notebooks for exploration
└── docs/                 # Documentation and research papers
```

For detailed project structure documentation, see [Project Structure Documentation](docs/project-structure.md).

## Code Style

This project follows Python typing standards and modern Python best practices.

### Key Guidelines

- **Type Hints**: All functions, methods, and class attributes must have type hints
- **Documentation**: All public functions, classes, and methods must have docstrings
- **Formatting**: 4 spaces indentation, 100 character line limit, double quotes
- **Naming**: PascalCase for classes, snake_case for functions, UPPER_SNAKE_CASE for constants

**Example:**
```python
from typing import List, Tuple
import torch

def process_images(
    images: List[torch.Tensor],
    batch_size: int = 32
) -> torch.Tensor:
    """Process a list of images."""
    # Implementation
    pass
```

For complete code style guidelines, including detailed examples and type checking tools, see [Code Style Documentation](docs/code-style.md).

## References

Vijayalakshmi, S., Manoharan, J. S., Nivetha, B., & Sathiya, A. (2025). Multi-task deep learning framework combining CNN, vision transformers and PSO for accurate diabetic retinopathy diagnosis and lesion localization. *Scientific Reports*, *15*(1), 35076. https://www.nature.com/articles/s41598-025-18742-z

Voxel51, Inc. (2024). *FiftyOne documentation*. https://docs.voxel51.com

Das, D., Biswas, S. K., & Bandyopadhyay, S. (2023). Detection of diabetic retinopathy using convolutional neural networks for feature extraction and classification (DRFEC). *Multimedia Tools and Applications*, *82*(19), 29943–30001. https://link.springer.com/article/10.1007/s11042-022-14165-4

Nahiduzzaman, M., Islam, M. R., Goni, M. O. F., Anower, M. S., Ahsan, M., Haider, J., & Kowalski, M. (2023). Diabetic retinopathy identification using parallel convolutional neural network based feature extractor and ELM classifier. *Expert Systems with Applications*, *217*, 119557. https://www.sciencedirect.com/science/article/pii/S0957417423000581

Usman, T. M., Saheed, Y. K., Ignace, D., & Nsang, A. (2023). Diabetic retinopathy detection using principal component analysis multi-label feature extraction and classification. *International Journal of Cognitive Computing in Engineering*, *4*, 78–88. https://www.sciencedirect.com/science/article/pii/S2666307423000050

Zaharia, M., Chen, A., Davidson, A., Ghodsi, A., Hong, S. A., Konwinski, A., ... & Zumar, C. (2018). Accelerating the machine learning lifecycle with MLflow. *IEEE Data Eng. Bull.*, *41*(4), 39-45. https://people.eecs.berkeley.edu/~alig/papers/mlflow.pdf
