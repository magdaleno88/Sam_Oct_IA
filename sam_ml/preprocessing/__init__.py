"""Preprocessing module with polymorphic dataset processors."""

import argparse
from pathlib import Path
from typing import Dict, Optional, Type

from sam_ml.preprocessing.base import DatasetProcessor
from sam_ml.preprocessing.eyepacs_dataset import KaggleEyePACSProcessor
from sam_ml.preprocessing.utils import apply_ceced_bgr_3ch, apply_clahe_bgr

# Registry of available processors
# Maps dataset names to their processor classes
_PROCESSOR_REGISTRY: Dict[str, Type[DatasetProcessor]] = {
    "eyepacs_dataset": KaggleEyePACSProcessor,
}


def get_processor(dataset_name: str) -> Type[DatasetProcessor]:
    """
    Get processor class for a given dataset name.
    
    Args:
        dataset_name: Name of the dataset (e.g., "eyepacs_dataset")
        
    Returns:
        Processor class for the dataset
        
    Raises:
        ValueError: If no processor is found for the dataset name
    """
    processor_class = _PROCESSOR_REGISTRY.get(dataset_name)
    
    if processor_class is None:
        available = ", ".join(_PROCESSOR_REGISTRY.keys())
        raise ValueError(
            f"No processor found for dataset '{dataset_name}'. "
            f"Available datasets: {available}"
        )
    
    return processor_class


def create_processor(
    dataset_name: str,
    raw_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> DatasetProcessor:
    """
    Factory function to create a processor instance for a given dataset.
    
    This function uses polymorphism to select the appropriate processor based on
    the dataset name, providing a decoupled way to add or remove processors.
    
    Args:
        dataset_name: Name of the dataset (mandatory, used to select processor)
        raw_dir: Directory containing raw dataset files. If None, uses default
                based on dataset name: data/raw/{dataset_name}
        processed_dir: Directory for processed dataset output. If None, uses
                       default: data/processed
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Processor instance for the specified dataset
        
    Raises:
        ValueError: If no processor is found for the dataset name
        
    Example:
        >>> processor = create_processor("eyepacs_dataset")
        >>> processor.process_dataset()
    """
    # Get processor class based on dataset name
    processor_class = get_processor(dataset_name)
    
    # Set default paths if not provided
    if raw_dir is None:
        raw_dir = Path("data/raw") / dataset_name
    
    if processed_dir is None:
        processed_dir = Path("data/processed")
    
    # Create and return processor instance
    return processor_class(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        dataset_name=dataset_name,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )


def list_available_datasets() -> list[str]:
    """
    List all available dataset processors.
    
    Returns:
        List of dataset names that have processors available
    """
    return list(_PROCESSOR_REGISTRY.keys())


def register_processor(dataset_name: str, processor_class: Type[DatasetProcessor]) -> None:
    """
    Register a new dataset processor.
    
    This allows adding new processors dynamically without modifying the core module.
    
    Args:
        dataset_name: Name of the dataset
        processor_class: Processor class that implements DatasetProcessor
        
    Example:
        >>> class MyProcessor(DatasetProcessor):
        ...     pass
        >>> register_processor("my_dataset", MyProcessor)
    """
    _PROCESSOR_REGISTRY[dataset_name] = processor_class


def main() -> None:
    """Main entry point for preprocessing script."""
    parser = argparse.ArgumentParser(
        description="Preprocess diabetic retinopathy dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Process EyePACS dataset with default paths
  preprocess-dataset eyepacs_dataset
  
  # Specify custom paths
  preprocess-dataset eyepacs_dataset --raw-dir /path/to/raw --processed-dir /path/to/processed
  
  # Custom split ratios
  preprocess-dataset eyepacs_dataset --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
  
Available datasets: {', '.join(list_available_datasets())}
        """
    )
    
    parser.add_argument(
        "dataset_name",
        type=str,
        help=f"Name of the dataset to process (required). Available: {', '.join(list_available_datasets())}"
    )
    
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Directory containing raw dataset files. If not specified, uses data/raw/{dataset_name}"
    )
    
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Directory for processed dataset output (default: data/processed)"
    )
    
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Proportion of data for training (default: 0.7)"
    )
    
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of data for validation (default: 0.15)"
    )
    
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proportion of data for testing (default: 0.15)"
    )
    
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Validate ratios sum to 1.0
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        parser.error(
            f"Train, validation, and test ratios must sum to 1.0. "
            f"Got {args.train_ratio} + {args.val_ratio} + {args.test_ratio} = {total_ratio}"
        )
    
    # Validate ratios are positive
    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.test_ratio <= 0:
        parser.error("All ratios must be positive")
    
    # Create processor using factory pattern
    try:
        processor = create_processor(
            dataset_name=args.dataset_name,
            raw_dir=Path(args.raw_dir) if args.raw_dir else None,
            processed_dir=Path(args.processed_dir) if args.processed_dir else None,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.random_seed
        )
    except ValueError as e:
        parser.error(str(e))
    
    # Process the dataset
    processor.process_dataset()


# Export main functions and classes
__all__ = [
    "DatasetProcessor",
    "KaggleEyePACSProcessor",
    "create_processor",
    "get_processor",
    "list_available_datasets",
    "register_processor",
    "apply_clahe_bgr",
    "apply_ceced_bgr_3ch",
    "main",
]

