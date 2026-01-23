# Testing

This project includes comprehensive unit tests using pytest. All tests follow best practices with proper mocking, fixtures, and type hints.

## Quick Start

### Install test dependencies

```bash
uv sync --extra test
```

### Run all tests

```bash
uv run pytest
```

The default configuration (from `pyproject.toml`) provides verbose output and proper test discovery, so no additional parameters are needed.

## Running Tests

### Run specific test files

```bash
uv run pytest tests/test_preprocess_ddr2019.py
uv run pytest tests/test_preprocessing_router.py
uv run pytest tests/test_base_model.py
```

### Run specific test classes or functions

```bash
# Run a specific test class
uv run pytest tests/test_base_model.py::TestInitialization

# Run a specific test function
uv run pytest tests/test_base_model.py::TestInitialization::test_init_default_parameters
```

### Run with coverage

```bash
uv run pytest --cov=sam_ml --cov-report=html
```

This generates an HTML coverage report in `htmlcov/index.html`.

## Test Options

### Quiet mode (minimal output)

```bash
uv run pytest -q
```

### Run specific test by name pattern

```bash
uv run pytest -k "test_resize"
```

### Run only fast tests (exclude slow markers)

```bash
uv run pytest -m "not slow"
```

### Run with specific markers

```bash
# Run only unit tests
uv run pytest -m "unit"

# Run only integration tests
uv run pytest -m "integration"

# Run GPU tests (if available)
uv run pytest -m "gpu"
```

## Test Structure

Tests are organized in the `tests/` directory:

```
tests/
├── __init__.py
├── conftest.py          # Shared fixtures and pytest configuration
├── test_base_model.py   # Tests for BaseLightningModel
├── test_preprocess_ddr2019.py
└── test_preprocessing_router.py
```

## Test Markers

The project uses pytest markers to categorize tests:

- `@pytest.mark.slow` - Tests that take longer to run
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.gpu` - Tests that require GPU
- `@pytest.mark.model` - Tests related to model architecture

## Writing Tests

### Best Practices

1. **Type Hints**: All test functions must have type hints
2. **Fixtures**: Use pytest fixtures for reusable test data
3. **Mocking**: Use `unittest.mock` for external dependencies
4. **Organization**: Group related tests in classes
5. **Naming**: Use descriptive test names that explain what is being tested

### Example Test Structure

```python
"""Unit tests for MyModule."""

from typing import Tuple
import pytest
import torch

from sam_ml.module import MyClass


@pytest.fixture
def sample_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """Create sample test data."""
    inputs = torch.randn(4, 10)
    targets = torch.randint(0, 5, (4,))
    return (inputs, targets)


class TestMyClass:
    """Tests for MyClass."""
    
    def test_initialization(self) -> None:
        """Test basic initialization."""
        obj = MyClass(num_classes=5)
        assert obj.num_classes == 5
    
    def test_forward_pass(self, sample_data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Test forward pass."""
        inputs, _ = sample_data
        obj = MyClass(num_classes=5)
        output = obj.forward(inputs)
        assert output.shape[0] == inputs.shape[0]
```

## Continuous Integration

Tests are designed to run in CI/CD pipelines. The default pytest configuration ensures consistent behavior across environments.

## Coverage Goals

The project aims for high test coverage. Use coverage reports to identify untested code:

```bash
uv run pytest --cov=sam_ml --cov-report=term-missing
```
