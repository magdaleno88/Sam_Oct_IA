# Project Structure

This document describes the organization and structure of the SAM-AI project.

## Directory Layout

```
sam-ai/
├── mlops_project/          # Main source code
│   ├── config.py           # Centralized configuration
│   ├── dataset.py          # Data loading and preparation
│   ├── features.py         # Feature engineering
│   └── modeling/           # Modeling module
│       ├── models/         # Model architectures
│       │   ├── base_model.py
│       │   ├── dual_channel_model.py
│       │   ├── fusion_layers.py
│       │   └── model_factory.py
│       ├── train.py        # Training / MLflow tracking
│       └── predict.py       # Prediction and inference
├── tests/                  # Automated tests
├── data/                   # Data (versioned with DVC)
│   ├── raw/                # Original data
│   └── processed/          # Processed data
├── models/                 # Trained models (DVC)
├── notebooks/              # Exploration notebooks
├── docs/                   # Additional documentation
├── scripts/                # Utility scripts
└── pyproject.toml          # Project configuration
```

## Module Descriptions

### `mlops_project/`

Main source code package containing all application logic.

#### `config.py`
Centralized configuration management for the project. Handles paths, model parameters, and data settings.

#### `dataset.py`
Data loading and preparation utilities. Functions for loading raw and processed data, data splitting, and preprocessing.

#### `features.py`
Feature engineering utilities. Functions for creating, selecting, and normalizing features.

#### `modeling/`

Module containing all machine learning modeling components.

##### `models/`
Model architecture implementations:
- `base_model.py`: Abstract base class for all DR models
- `dual_channel_model.py`: Dual-channel weighted fusion model implementation
- `fusion_layers.py`: Custom fusion layers (weighted and attention-based)
- `model_factory.py`: Factory pattern for model creation

##### `train.py`
Training pipeline with MLflow tracking integration.

##### `predict.py`
Prediction and inference utilities.

### `tests/`

Unit and integration tests for the project.

### `data/`

Data directory structure:
- `raw/`: Original, unprocessed data files
- `processed/`: Preprocessed and cleaned data ready for modeling

**Note:** Data is versioned using DVC (Data Version Control).

### `models/`

Directory for storing trained model artifacts. Versioned with DVC.

### `notebooks/`

Jupyter notebooks for exploratory data analysis, experimentation, and visualization.

### `docs/`

Additional documentation including:
- Installation guides
- API documentation
- Architecture documentation
- Usage examples

## Data Versioning

This project uses DVC (Data Version Control) for:
- Versioning datasets
- Tracking model artifacts
- Reproducible experiments

## Configuration

Project configuration is managed through:
- `pyproject.toml`: Python project metadata and dependencies
- `mlops_project/config.py`: Runtime configuration settings

