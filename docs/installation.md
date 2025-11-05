# Installation Guide

This guide provides detailed instructions for installing and setting up the SAM-AI project.

## TensorFlow and Keras Setup

This project uses `uv` for package management. TensorFlow (which includes Keras) is installed with CPU support by default, with optional GPU acceleration available.

### CPU (Default - All Platforms)

```bash
uv sync
```

This installs TensorFlow with CPU support by default. Works on all platforms (Linux, macOS Intel, macOS Apple Silicon, Windows).

### NVIDIA GPU (CUDA)

```bash
uv sync --extra cuda
```

**Requirements:**
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- cuDNN library installed

The `tensorflow[and-cuda]` package will replace the base TensorFlow installation.

### Apple Silicon (MPS/Metal) - ARM64 Only

```bash
uv sync --extra mps
```

**Requirements:**
- Apple Silicon Mac (M1/M2/M3)
- macOS 12.0 or higher

**Note:** Only works on Apple Silicon Macs. On Intel Macs, use `--extra cpu` instead. The platform markers will automatically prevent installation on Intel Macs.

### Verify Installation

After installation, verify that TensorFlow is correctly installed:

```bash
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}'); print(f'Devices: {tf.config.list_physical_devices()}')"
```

This will display:
- The installed TensorFlow version
- Available devices (CPU, GPU, etc.)

## Troubleshooting

### CUDA Installation Issues

If CUDA installation fails:
1. Verify NVIDIA drivers are installed: `nvidia-smi`
2. Check CUDA Toolkit version compatibility
3. Ensure cuDNN is properly installed
4. Fall back to CPU installation: `uv sync`

### MPS Installation Issues

If MPS installation fails on Apple Silicon:
1. Verify you're on macOS 12.0+
2. Check system architecture: `uname -m` (should show `arm64`)
3. Fall back to CPU installation: `uv sync`

