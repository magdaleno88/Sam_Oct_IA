"""Abstract base class for dataset processors."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple


class DatasetProcessor(ABC):
    """
    Abstract base class for dataset processors.
    
    Each concrete processor implements methods to extract, process, and organize
    a specific dataset format. This allows for easy extension with new dataset types
    while maintaining a consistent interface.
    """
    
    def __init__(
        self,
        raw_dir: Path,
        processed_dir: Path,
        dataset_name: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ) -> None:
        """
        Initialize the dataset processor.
        
        Args:
            raw_dir: Directory containing raw dataset files
            processed_dir: Directory for processed dataset output
            dataset_name: Name of the processed dataset
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            random_seed: Random seed for reproducibility
        """
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.dataset_name = dataset_name
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"Train, validation, and test ratios must sum to 1.0. "
                f"Got {train_ratio} + {val_ratio} + {test_ratio} = {total_ratio}"
            )
        
        if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
            raise ValueError("All ratios must be positive")
    
    @abstractmethod
    def extract_raw_data(self) -> Tuple[Path, Path]:
        """
        Extract raw data files (e.g., from zip archives).
        
        Returns:
            Tuple of (train_dir, test_dir) paths containing extracted images
        """
        pass
    
    @abstractmethod
    def load_labels(self) -> Dict[str, int]:
        """
        Load dataset labels from the raw data.
        
        Returns:
            Dictionary mapping image name (without extension) to label (0-4)
        """
        pass
    
    @abstractmethod
    def create_directory_structure(self) -> Dict[str, Path]:
        """
        Create the directory structure for processed dataset.
        
        Returns:
            Dictionary with paths to created directories
        """
        pass
    
    @abstractmethod
    def split_dataset(
        self,
        image_paths: List[Path],
        labels: Dict[str, int]
    ) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]]]:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            image_paths: List of image file paths
            labels: Dictionary mapping image names to labels
            
        Returns:
            Tuple of (train, val, test) lists, each containing (path, label) tuples
        """
        pass
    
    @abstractmethod
    def process_dataset(self) -> None:
        """
        Main method to process the dataset from raw to processed format.
        
        This method orchestrates the full processing pipeline:
        1. Extract raw data
        2. Load labels
        3. Create directory structure
        4. Split dataset
        5. Process images with CLAHE and CECED
        6. Save processed images
        """
        pass
    
    @property
    @abstractmethod
    def supported_dataset_name(self) -> str:
        """
        Return the dataset name this processor supports.
        
        Returns:
            Dataset name identifier (e.g., "eyepacs_dataset")
        """
        pass

