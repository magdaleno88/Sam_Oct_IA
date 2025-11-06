# Tests

This directory contains unit tests for the SAM-AI project using pytest.

## Running Tests

### Install test dependencies

First, install the test dependencies:

```bash
uv sync --extra test
```

### Using `uv run` (Recommended)

The recommended way to run tests is using `uv run`, which ensures tests run in the correct environment.

**Note:** The project includes pytest configuration in `pyproject.toml` that sets default options, so you don't need to pass many parameters for common test runs.

#### Run all tests

```bash
uv run pytest
```

Or explicitly:

```bash
uv run pytest tests/
```

#### Run specific test file

```bash
uv run pytest tests/test_fusion_layers.py
uv run pytest tests/test_dual_channel_model.py
```

#### Run with coverage

The default configuration doesn't include coverage. To run with coverage:

```bash
uv run pytest --cov=sam_ml --cov-report=html
```

Or edit `pyproject.toml` under `[tool.pytest.ini_options]` to enable coverage by default (see configuration section).

#### Run with minimal output (quiet mode)

```bash
uv run pytest -q
```

#### Run specific test by name

```bash
uv run pytest -k "test_fusion_layer"
```

#### Run tests matching a marker

```bash
uv run pytest -m "unit"
uv run pytest -m "not slow"
```

#### View detailed test reports with warnings and errors

To see warnings and detailed error information in the test report:

```bash
# Show warnings (Python warnings, not pytest warnings summary)
uv run pytest -W default

# Show all warnings including deprecations
uv run pytest -W all

# Show very detailed output (including local variables in tracebacks)
uv run pytest -vv

# Show long traceback format (more detailed error information)
uv run pytest --tb=long

# Show all information: warnings + detailed errors + captured output
uv run pytest -W default -vv --tb=long -s

# Show warnings summary at the end (overrides --disable-warnings from config)
# Note: This shows pytest warnings, not Python warnings
uv run pytest -rw

# Show warnings summary with specific categories
uv run pytest -rw -W default
```

**Note:** The default configuration has `--disable-warnings` which hides pytest's warning summary. Use `-W` for Python warnings and `-rw` for pytest warnings summary.

### Direct pytest execution

If you prefer to run pytest directly (after installing dependencies):

```bash
pytest tests/
pytest tests/test_fusion_layers.py
pytest tests/ --cov=sam_ml --cov-report=html
pytest tests/ -v
```

**Note:** Using `uv run` is recommended as it ensures the correct Python environment and dependencies are used.

## Configuration

The project includes pytest configuration in `pyproject.toml` under `[tool.pytest.ini_options]` that configures default pytest behavior:

- **Verbose output by default** (`-v`)
- **Short traceback format** (`--tb=short`)
- **Test discovery** configured to look in `tests/` directory
- **Markers** defined for categorizing tests (slow, integration, unit, gpu, model)
- **Warning filters** to reduce noise from TensorFlow deprecation warnings
- **Warnings disabled by default** (`--disable-warnings`) to reduce noise

### Viewing Warnings and Errors

By default, warnings are suppressed to keep output clean. To see detailed information:

#### See Warnings

```bash
# Show Python warnings during test execution
uv run pytest -W default

# Show all warnings including deprecations
uv run pytest -W all

# Show pytest warnings summary (overrides --disable-warnings)
uv run pytest -rw

# Combine both: show Python warnings AND pytest warnings summary
uv run pytest -W default -rw
```

#### See Detailed Error Information

```bash
# Long traceback format (shows full stack traces)
uv run pytest --tb=long

# Very verbose output (shows local variables in tracebacks)
uv run pytest -vv --tb=long

# Show captured output (print statements, logs)
uv run pytest -s

# Complete detail: warnings + verbose + long traceback + captured output
uv run pytest -W default -vv --tb=long -s
```

#### Understanding Test Report Flags

- `-v` or `-vv`: Verbose output (one or two levels)
- `--tb=short|long|line|no`: Traceback format (short is default)
- `-W`: Control Python warnings (default, all, error::WarningType)
- `-rw`: Show pytest warnings summary (overrides --disable-warnings)
- `-s`: Show captured output (print statements, logs)
- `-r`: Show test summary info (default shows fE for failures and errors)

You can customize these defaults by editing `pyproject.toml` under `[tool.pytest.ini_options]`.

## Test Files

- `test_fusion_layers.py`: Tests for `WeightedFusionLayer`
  - Initialization
  - Weight creation and constraints
  - Feature fusion
  - Gradient flow
  - Error handling

- `test_dual_channel_model.py`: Tests for `DualChannelDiabeticRetinopathyModel`
  - Model initialization
  - Forward pass
  - Output shapes and probabilities
  - Training and inference modes
  - Model compilation
  - Integration with components

- `conftest.py`: Shared pytest fixtures and configuration

## Test Data

All tests use dummy data generated within the test fixtures:
- Random tensors for images
- One-hot encoded labels
- Proper shapes matching expected model inputs

No external data files are required for running tests.

