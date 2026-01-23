# Project Structure

This document describes the organization of the SAM-AI project.

## Directory Layout

```
sam-ai/
├── data/
│   ├── raw/              # Raw, unmodified datasets
│   └── processed/        # Processed datasets ready for training
├── sam_ml/
│   ├── preprocessing/    # Dataset preprocessing scripts
│   ├── datasets/         # PyTorch Dataset classes
│   └── modeling/         # Model training and prediction
│       └── models/       # Model architectures
│           └── base.py   # BaseLightningModel class
├── tests/                # Unit tests
├── notebooks/            # Jupyter notebooks for exploration
└── docs/                 # Documentation and research papers
    ├── pdfs/            # Research papers (PDFs)
    ├── installation.md
    ├── testing.md
    ├── preprocessing.md
    ├── code-style.md
    ├── modeling.md
    └── project-structure.md
```

## Key Directories

### `data/`

Contains all dataset files:
- `raw/`: Original, unmodified datasets
- `processed/`: Preprocessed datasets ready for training

**Note**: Raw data files are never modified. All processing creates new files in `processed/`.

### `sam_ml/`

Main Python package containing all project code:

- **`preprocessing/`**: Dataset preprocessing scripts
  - `preprocess_ddr2019.py`: DDR2019 dataset preprocessing
  - `base.py`: Base preprocessing utilities
  - `utils.py`: Helper functions

- **`datasets/`**: PyTorch Dataset classes for data loading

- **`modeling/`**: Model training and prediction
  - `train.py`: Training scripts
  - `predict.py`: Prediction/inference scripts
  - `models/`: Model architectures
    - `base.py`: BaseLightningModel base class

### `tests/`

Unit tests organized by module:
- `test_base_model.py`: Tests for BaseLightningModel
- `test_preprocess_ddr2019.py`: Tests for preprocessing
- `test_preprocessing_router.py`: Tests for CLI router
- `conftest.py`: Shared fixtures and pytest configuration

### `notebooks/`

Jupyter notebooks for:
- Data exploration
- Model experimentation
- Visualization

### `docs/`

Documentation files:
- Detailed guides for each major topic
- Research papers (in `pdfs/`)
- API documentation

## Module Organization

### Preprocessing Module

Handles dataset preparation:
- Image resizing and padding
- Label file conversion
- Dataset-specific preprocessing logic

### Modeling Module

Provides model training infrastructure:
- Base model class for PyTorch Lightning
- Training utilities
- Prediction utilities

### Datasets Module

PyTorch Dataset implementations for:
- Loading processed images
- Label handling
- Data augmentation (future)

## File Naming Conventions

- **Python modules**: `snake_case.py`
- **Test files**: `test_<module_name>.py`
- **Documentation**: `kebab-case.md`
- **Notebooks**: `snake_case.ipynb`

## Import Structure

The project uses absolute imports:

```python
from sam_ml.preprocessing.preprocess_ddr2019 import preprocess_ddr2019
from sam_ml.modeling.models import BaseLightningModel
from sam_ml.datasets import MyDataset
```

## Adding New Components

When adding new components:

1. **New preprocessing script**: Add to `sam_ml/preprocessing/`
2. **New model**: Add to `sam_ml/modeling/models/`
3. **New dataset**: Add to `sam_ml/datasets/`
4. **New tests**: Add to `tests/` with `test_` prefix
5. **New documentation**: Add to `docs/` as markdown file

## Configuration Files

- `pyproject.toml`: Project configuration, dependencies, pytest settings
- `.gitignore`: Git ignore patterns
- `uv.lock`: Locked dependency versions (managed by uv)
