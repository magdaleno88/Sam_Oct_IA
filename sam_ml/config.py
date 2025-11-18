"""Centralized configuration for the ML project."""

from pathlib import Path
from typing import Optional


# Project root directory (parent of sam_ml package)
# This assumes config.py is in sam_ml/, so project root is parent
_PROJECT_ROOT: Optional[Path] = None


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    The project root is the directory containing the sam_ml package.
    This is determined by finding the directory containing pyproject.toml.
    
    Returns:
        Path to the project root directory
        
    Example:
        >>> root = get_project_root()
        >>> data_dir = root / "data"
    """
    global _PROJECT_ROOT
    
    if _PROJECT_ROOT is None:
        # Start from this file's location
        current_file = Path(__file__).resolve()
        # Go up to sam_ml/, then to project root
        sam_ml_dir = current_file.parent
        project_root = sam_ml_dir.parent
        
        # Verify by checking for pyproject.toml
        if (project_root / "pyproject.toml").exists():
            _PROJECT_ROOT = project_root
        else:
            # Fallback: assume we're in the project root if pyproject.toml is in current dir
            # This handles cases where the module is run from different locations
            _PROJECT_ROOT = project_root
    
    return _PROJECT_ROOT


def get_data_dir() -> Path:
    """
    Get the data directory path.
    
    Returns:
        Path to the data directory (project_root/data)
    """
    return get_project_root() / "data"


def get_raw_data_dir() -> Path:
    """
    Get the raw data directory path.
    
    Returns:
        Path to the raw data directory (project_root/data/raw)
    """
    return get_data_dir() / "raw"


def get_processed_data_dir() -> Path:
    """
    Get the processed data directory path.
    
    Returns:
        Path to the processed data directory (project_root/data/processed)
    """
    return get_data_dir() / "processed"


def get_processed_dataset_path(dataset_name: str = "eyepacs_dataset") -> Path:
    """
    Get the path to a specific processed dataset.
    
    Args:
        dataset_name: Name of the dataset (default: "eyepacs_dataset")
        
    Returns:
        Path to the processed dataset directory (project_root/data/processed/{dataset_name})
    """
    return get_processed_data_dir() / dataset_name


def get_models_dir() -> Path:
    """
    Get the models directory path.
    
    Returns:
        Path to the models directory (project_root/models)
    """
    return get_project_root() / "models"


def get_notebooks_dir() -> Path:
    """
    Get the notebooks directory path.
    
    Returns:
        Path to the notebooks directory (project_root/notebooks)
    """
    return get_project_root() / "notebooks"
