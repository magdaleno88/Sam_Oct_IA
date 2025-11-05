# SAM - AI
A computer vision project to aid in the diagnosis of diabetic retinopathy.

## Installation

### TensorFlow and Keras Setup

This project uses `uv` for package management. TensorFlow (which includes Keras) is installed with CPU support by default.

#### CPU (Default - All Platforms)

```bash
uv sync
```

This installs TensorFlow with CPU support by default. Works on all platforms (Linux, macOS Intel, macOS Apple Silicon, Windows).

#### NVIDIA GPU (CUDA)

```bash
uv sync --extra cuda
```

Note: Requires NVIDIA GPU, CUDA Toolkit, and cuDNN installed on your system. The `tensorflow[and-cuda]` package will replace the base TensorFlow installation.

#### Apple Silicon (MPS/Metal) - ARM64 Only

```bash
uv sync --extra mps
```

Note: **Only works on Apple Silicon Macs (M1/M2/M3)**. Requires macOS 12.0+. On Intel Macs, use `--extra cpu` instead. The platform markers will automatically prevent installation on Intel Macs.

#### Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}'); print(f'Devices: {tf.config.list_physical_devices()}')"
```

### Project Structure

```
sam-ai/
├── mlops_project/          # Main source code
│   ├── config.py           # Centralized configuration
│   ├── dataset.py          # Data loading and preparation
│   ├── features.py         # Feature engineering
│   └── modeling/           # Modeling module
│       ├── train.py        # Training / MLflow tracking
│       └── predict.py      # Prediction and inference
├── tests/                  # Automated tests
├── data/                   # Data (versioned with DVC)
│   ├── raw/                # Original data
│   └── processed/          # Processed data
├── models/                 # Trained models (DVC)
├── notebooks/              # Exploration notebooks
├── docs/                   # Additional documentation
└── pyproject.toml          # Project configuration
```

## References
- [Identification of Diabetic Retinopathy Using Weighted Fusion Deep Learning Based on Dual-Channel Fundus Scans](https://www.mdpi.com/2075-4418/12/2/540)