# SAM - AI

A computer vision research project with two independent retinal imaging workflows: diabetic retinopathy analysis using fundus photographs, and macular OCT classification using B-scan images. The original DDR2019 workflow remains available alongside the OCT pipeline.

| Workflow | Images | Dataset | Code and configuration |
|---|---|---|---|
| Diabetic retinopathy | Color fundus photographs | DDR2019 | `sam_ml/preprocessing/`, `sam_ml/datasets/`, `sam_ml/modeling/`, `sam_ml/config.py` |
| Macular OCT | OCT B-scans | UCSD/Kermany OCT | `sam_ml/oct/`, `scripts/*oct*.py`, `configs/oct.yaml` |

> **Research use only.** Model outputs do not replace evaluation by an ophthalmologist.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Testing](#testing)
- [Preprocessing](#preprocessing)
- [Modeling](#modeling)
- [Creating Models](#creating-models)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Macular OCT Workflow](#macular-oct-workflow)
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

The project includes a preprocessing module for preparing diabetic retinopathy datasets. Currently supports the DDR2019 dataset. The pipeline uses a **preprocessor registry** (first CLI argument is a keyword, e.g. `ddr2019`) and **middleware** (per-image transforms; built-in: default, paper_dual, resize_norm). Image I/O uses OpenCV BGR; defaults live in `sam_ml/config.py`.

### Quick Start

Process the DDR2019 dataset using the CLI:

```bash
# Process with default settings (min-size=512, target-size=512x512)
uv run preprocess-dataset ddr2019

# Process with custom settings
uv run preprocess-dataset ddr2019 --min-size 512 --target-size 512 512

# Output to a different folder (e.g. second version with different size)
uv run preprocess-dataset ddr2019 --output-name ddr2019_384 --target-size 384 384
```

### Features

- **Preprocessor and middleware registries**: Extensible by keyword; add custom preprocessors and middlewares (see [Preprocessing Documentation](docs/preprocessing.md)).
- **Minimum size filtering**: Only processes images with both dimensions >= 512x512 (configurable).
- **Automatic padding**: Non-square images are padded to square (black padding).
- **No upscaling**: Images are only downscaled or kept at same size (never upscaled to avoid noise).
- **Standardized output**: All processed images are resized to the target size (default 512×512).
- **Optional middlewares**: Use `--middleware paper_dual` for CLAHE/CECED variants or `--middleware resize_norm` for resize+normalize.
- **Label synchronization**: CSV labels are automatically filtered to match processed images.
- **Original data protection**: Original dataset files are never modified.

For complete preprocessing documentation, including workflow, registry, custom preprocessors/middlewares, and troubleshooting, see [Preprocessing Documentation](docs/preprocessing.md).

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

## Creating Models

The project includes a model registry system that makes it easy to create and experiment with new model architectures. You can:

- Create your model architecture (PyTorch `nn.Module`)
- Wrap it in `BaseLightningModel` for automatic training capabilities
- Register it with a unique key
- Run experiments using the unified training pipeline

**Quick Example:**
```python
from sam_ml.modeling.models import get_model

# Get any registered model
model = get_model("simple_cnn", num_classes=5, learning_rate=1e-4)

# List all available models
from sam_ml.modeling.models import list_models
print(list_models())  # ['simple_cnn', ...]
```

**Training a Model:**
```bash
# Train using the script shortcut
uv run train-model --model simple_cnn --num-classes 5 --num-epochs 50

# List available models and options
uv run train-model --help
```

For a complete step-by-step guide on creating new models, see [Creating Models Guide](docs/creating-models.md).

## Configuration

The project uses a centralized configuration system built with Pydantic. All default values are managed in `sam_ml/config.py`.

### Quick Start

```python
from sam_ml.config import get_model_config, get_training_config

# Get configuration defaults
model_config = get_model_config()
training_config = get_training_config()

print(f"Default num_classes: {model_config.num_classes}")
print(f"Default batch_size: {training_config.batch_size}")
```

### Environment Variables

Override defaults using environment variables:

```bash
export SAM_MODEL_NUM_CLASSES=3
export SAM_TRAINING_BATCH_SIZE=64
```

For complete configuration documentation, including all available settings and customization options, see [Configuration Documentation](docs/configuration.md).

## Project Structure

```
sam-ai/
├── data/
│   ├── raw/              # Raw, unmodified datasets
│   ├── processed/        # Processed datasets ready for training
│   └── prepared/oct/     # Prepared OCT train/test data
├── sam_ml/
│   ├── preprocessing/    # Dataset preprocessing scripts
│   ├── datasets/        # DDR2019 PyTorch Dataset classes
│   ├── oct/             # Independent OCT pipeline
│   └── modeling/        # Model training and prediction
│       └── models/      # Model architectures
├── configs/oct.yaml      # OCT configuration
├── scripts/               # OCT CLI scripts
├── tests/                 # Unit tests
├── notebooks/            # Jupyter notebooks for exploration
└── docs/                 # Documentation and research papers
```

For detailed project structure documentation, see [Project Structure Documentation](docs/project-structure.md).

## Macular OCT Workflow

The OCT pipeline is separate from the original DDR2019 fundus workflow. It classifies macular B-scans into **CNV**, **DME**, **DRUSEN**, and **NORMAL**. Its code lives in `sam_ml/oct/`, with command-line scripts in `scripts/` and configuration in `configs/oct.yaml`.

### OCT data and preparation

Provide the UCSD/Kermany images under `data/raw/`, with `train/` and `test/` directories containing the four class folders. An optional `val/` directory is supported. The dataset is not downloaded automatically.

The recommended preparation command samples only training images, applies moderate balancing and preprocessing, and preserves the full test set:

```bash
uv run python scripts/prepare_oct_dataset.py --config configs/oct.yaml --input-root data/raw --output-root data/prepared/oct --train-percentage 10 --sampling-unit auto --balance-mode moderate-physical
```

Set `--train-percentage 100` to retain the full training set, or `--balance-mode none` to disable balancing. The pipeline does not overwrite original images. Patient-level sampling is used when reliable IDs are available; image-level sampling requires the explicit `--allow-image-level-sampling` option.

### OCT models and training

The OCT workflow provides `baseline_resnet50` and `improved_resnet50`. Both predict the same four classes; the improved model supports configurable dilation. The original DDR2019 model registry described above remains separate.

```bash
uv run python scripts/train_oct.py --config configs/oct.yaml --model baseline_resnet50 --experiment baseline
uv run python scripts/train_oct.py --config configs/oct.yaml --model improved_resnet50 --experiment improved
```

For a four-member ensemble:

```bash
uv run python scripts/train_ensemble.py --config configs/oct.yaml --ensemble-size 4 --experiment paper_ensemble
```

The OCT settings, including image size, model, training, and dataset preparation, are in [`configs/oct.yaml`](configs/oct.yaml). The DDR2019 defaults remain in `sam_ml/config.py`.

### OCT evaluation and inference

```bash
uv run python scripts/evaluate.py --run runs/paper_ensemble --split test --config configs/oct.yaml
uv run python scripts/predict.py --image path/to/oct.jpeg --checkpoint runs/improved/checkpoints/best.ckpt --config configs/oct.yaml
```

Evaluation includes per-class and aggregate classification metrics, a confusion matrix, and ROC-AUC. The pipeline also supports occlusion sensitivity and Grad-CAM through `scripts/explain_predictions.py`. Predictions are experimental and require clinical interpretation.

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

1. Kumar, V., Sharma, G., & Garg, D. (2023, November). Analysis of Early Detection and Prediction of Diabetic Retinopathy by Optimize Deep Learning with XG-Boosting. In Conference on Smart Generation Computing and Communication Networks (pp. 61-69). Cham: Springer Nature Switzerland. https://link.springer.com/chapter/10.1007/978-3-032-06798-2_6

2. Lalithadevi, B., & Krishnaveni, S. (2024). Diabetic retinopathy detection and severity classification
using optimized deep learning with explainable AI technique. Multimedia Tools and Applications,
83(42), 89949-90013. https://link.springer.com/article/10.1007/s11042-024-18863-z
3. Muthusamy, D., & Palani, P. (2024). Deep learning model using classification for diabetic
retinopathy detection: an overview. Artificial Intelligence Review, 57(7), 185.
https://link.springer.com/content/pdf/10.1007/s10462-024-10806-2.pdf
4. Nahiduzzaman, M., Islam, M. R., Goni, M. O. F., Anower, M. S., Ahsan, M., Haider, J., & Kowalski, M.
(2023). Diabetic retinopathy identification using parallel convolutional neural network based feature
extractor and ELM classifier. Expert Systems with Applications, 217, 119557.
https://www.sciencedirect.com/science/article/pii/S0957417423000581
5. Usman, T. M., Saheed, Y. K., Ignace, D., & Nsang, A. (2023). Diabetic retinopathy detection using
principal component analysis multi-label feature extraction and classification. International Journal
of Cognitive Computing in Engineering, 4, 78–88.
https://www.sciencedirect.com/science/article/pii/S2666307423000050

6. Vijayalakshmi, S., Manoharan, J. S., Nivetha, B., & Sathiya, A. (2025). Multi-task deep learning framework combining CNN, vision transformers and PSO for accurate diabetic retinopathy diagnosis and lesion localization. Scientific Reports, 15(1), 35076. https://www.nature.com/articles/s41598-
025-18742-z

7. Visengeriyeva, L., Kammer, A., Bär, I., Kniesz, A., y Plöd, M. (2023). CRISP-ML(Q). The ML Lifecycle Process. MLOps. INNOQ.

8. Voxel51, Inc. (2024). FiftyOne documentation. https://docs.voxel51.com

9. Zaharia, M., Chen, A., Davidson, A., Ghodsi, A., Hong, S. A., Konwinski, A., ... & Zumar, C.
(2018). Accelerating the machine learning lifecycle with MLflow. IEEE Data Eng. Bull., 41(4), 39-45. https://people.eecs.berkeley.edu/~alig/papers/mlflow.pdf

