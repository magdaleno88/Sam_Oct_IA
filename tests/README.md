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
