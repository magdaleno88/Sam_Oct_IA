# Installation

This project uses `uv` for package management and environment management.

## Quick Start

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Verify installation:**
   ```bash
   uv run python -c "import torch; print(f'Torch: {torch.__version__}')"
   ```

## Platform-Specific Notes

PyTorch is configured with platform-specific versions:

- **macOS Intel (x86_64)**: PyTorch 2.2.2 (latest version with Intel support)
- **Other platforms**: PyTorch 2.2.2 or newer

This configuration is handled automatically in `pyproject.toml` based on your platform.

## Installing Optional Dependencies

### Test Dependencies

To run tests, install test dependencies:

```bash
uv sync --extra test
```

### Jupyter Notebook Dependencies

For working with notebooks:

```bash
uv sync --extra notebook
```

## Verifying Installation

After installation, verify that key packages are available:

```bash
uv run python -c "import torch; import torchvision; import pytorch_lightning; print('All packages installed successfully')"
```

## Troubleshooting

### NumPy Compatibility Warnings

You may see warnings about NumPy compatibility with PyTorch. These are typically harmless and don't affect functionality. If you encounter issues, you can downgrade NumPy:

```bash
uv add "numpy<2"
```

### Platform-Specific Issues

If you encounter issues on macOS Intel, ensure you're using PyTorch 2.2.2. The project configuration should handle this automatically, but if you need to manually specify:

```bash
uv add "torch==2.2.2" "torchvision==0.17.2"
```
