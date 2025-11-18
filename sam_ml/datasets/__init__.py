"""Dataset loading module for TensorFlow data pipelines."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import tensorflow as tf

from sam_ml.datasets.eyepacs import load_eyepacs_datasets, load_eyepacs_dual_channel

__all__ = [
    "load_eyepacs_datasets",
    "load_eyepacs_dual_channel",
    "DualChannelDatasets",
    "create_eyepacs_datasets",
]


class DualChannelDatasets:
    """
    Container for dual-channel datasets (CLAHE and CECED).
    
    This class holds train, validation, and test datasets for both preprocessing
    channels, along with metadata about the datasets.
    """
    
    def __init__(
        self,
        clahe_train: tf.data.Dataset,
        clahe_val: tf.data.Dataset,
        clahe_test: tf.data.Dataset,
        ceced_train: tf.data.Dataset,
        ceced_val: tf.data.Dataset,
        ceced_test: tf.data.Dataset,
        num_classes: int = 5,
        class_names: Optional[Dict[int, str]] = None
    ) -> None:
        """
        Initialize dual-channel datasets container.
        
        Args:
            clahe_train: Training dataset for CLAHE channel
            clahe_val: Validation dataset for CLAHE channel
            clahe_test: Test dataset for CLAHE channel
            ceced_train: Training dataset for CECED channel
            ceced_val: Validation dataset for CECED channel
            ceced_test: Test dataset for CECED channel
            num_classes: Number of classes (default: 5)
            class_names: Optional mapping of class index to class name
        """
        self.clahe_train = clahe_train
        self.clahe_val = clahe_val
        self.clahe_test = clahe_test
        self.ceced_train = ceced_train
        self.ceced_val = ceced_val
        self.ceced_test = ceced_test
        self.num_classes = num_classes
        self.class_names = class_names or {
            0: "No Diabetic Retinopathy",
            1: "Mild Retinopathy",
            2: "Moderate Retinopathy",
            3: "Severe Retinopathy",
            4: "Proliferative Retinopathy",
        }
    
    def get_combined_train(self) -> tf.data.Dataset:
        """
        Get combined training dataset for dual-channel model.
        
        Returns:
            Zipped dataset containing (CLAHE, CECED) image pairs with labels
        """
        return tf.data.Dataset.zip((self.clahe_train, self.ceced_train))
    
    def get_combined_val(self) -> tf.data.Dataset:
        """
        Get combined validation dataset for dual-channel model.
        
        Returns:
            Zipped dataset containing (CLAHE, CECED) image pairs with labels
        """
        return tf.data.Dataset.zip((self.clahe_val, self.ceced_val))
    
    def get_combined_test(self) -> tf.data.Dataset:
        """
        Get combined test dataset for dual-channel model.
        
        Returns:
            Zipped dataset containing (CLAHE, CECED) image pairs with labels
        """
        return tf.data.Dataset.zip((self.clahe_test, self.ceced_test))
    
    def get_train_split(
        self,
        split: str = "both"
    ) -> Tuple[tf.data.Dataset, ...]:
        """
        Get training datasets for specified channel(s).
        
        Args:
            split: One of "clahe", "ceced", or "both"
            
        Returns:
            Tuple of datasets. For "both", returns (clahe_train, ceced_train).
            For single channel, returns single dataset.
        """
        if split == "clahe":
            return (self.clahe_train,)
        elif split == "ceced":
            return (self.ceced_train,)
        elif split == "both":
            return (self.clahe_train, self.ceced_train)
        else:
            raise ValueError(f"split must be 'clahe', 'ceced', or 'both', got {split}")
    
    def get_val_split(
        self,
        split: str = "both"
    ) -> Tuple[tf.data.Dataset, ...]:
        """
        Get validation datasets for specified channel(s).
        
        Args:
            split: One of "clahe", "ceced", or "both"
            
        Returns:
            Tuple of datasets. For "both", returns (clahe_val, ceced_val).
            For single channel, returns single dataset.
        """
        if split == "clahe":
            return (self.clahe_val,)
        elif split == "ceced":
            return (self.ceced_val,)
        elif split == "both":
            return (self.clahe_val, self.ceced_val)
        else:
            raise ValueError(f"split must be 'clahe', 'ceced', or 'both', got {split}")
    
    def get_test_split(
        self,
        split: str = "both"
    ) -> Tuple[tf.data.Dataset, ...]:
        """
        Get test datasets for specified channel(s).
        
        Args:
            split: One of "clahe", "ceced", or "both"
            
        Returns:
            Tuple of datasets. For "both", returns (clahe_test, ceced_test).
            For single channel, returns single dataset.
        """
        if split == "clahe":
            return (self.clahe_test,)
        elif split == "ceced":
            return (self.ceced_test,)
        elif split == "both":
            return (self.clahe_test, self.ceced_test)
        else:
            raise ValueError(f"split must be 'clahe', 'ceced', or 'both', got {split}")


def create_eyepacs_datasets(
    base_path: Optional[Path] = None,
    dataset_name: str = "eyepacs_dataset",
    batch_size: int = 32,
    image_size_clahe: Tuple[int, int] = (299, 299),
    image_size_ceced: Tuple[int, int] = (224, 224),
    label_mode: str = "categorical",
    shuffle: bool = True,
    seed: Optional[int] = 42,
    cache: bool = True,
    prefetch: bool = True,
) -> DualChannelDatasets:
    """
    Create a DualChannelDatasets container for EyePACS dataset.
    
    This is a convenience function that loads datasets and wraps them in
    a DualChannelDatasets container for easier access.
    
    Args:
        base_path: Base path to processed datasets. If None, uses default:
                   data/processed/{dataset_name}
        dataset_name: Name of the dataset directory (default: "eyepacs_dataset")
        batch_size: Batch size for datasets (default: 32)
        image_size_clahe: Image size for CLAHE channel (default: (299, 299))
        image_size_ceced: Image size for CECED channel (default: (224, 224))
        label_mode: Label format - "categorical", "int", or "binary" (default: "categorical")
        shuffle: Whether to shuffle the training data (default: True)
        seed: Random seed for shuffling (default: 42)
        cache: Whether to cache datasets in memory (default: True)
        prefetch: Whether to prefetch batches (default: True)
        
    Returns:
        DualChannelDatasets container with all datasets
        
    Example:
        >>> from sam_ml.datasets import create_eyepacs_datasets
        >>> datasets = create_eyepacs_datasets(batch_size=64)
        >>> clahe_train, ceced_train = datasets.get_train_split("both")
        >>> model.fit([clahe_train, ceced_train], validation_data=([datasets.clahe_val, datasets.ceced_val]))
    """
    (
        clahe_train,
        clahe_val,
        clahe_test,
        ceced_train,
        ceced_val,
        ceced_test,
    ) = load_eyepacs_datasets(
        base_path=base_path,
        dataset_name=dataset_name,
        batch_size=batch_size,
        image_size_clahe=image_size_clahe,
        image_size_ceced=image_size_ceced,
        label_mode=label_mode,
        shuffle=shuffle,
        seed=seed,
        cache=cache,
        prefetch=prefetch,
    )
    
    return DualChannelDatasets(
        clahe_train=clahe_train,
        clahe_val=clahe_val,
        clahe_test=clahe_test,
        ceced_train=ceced_train,
        ceced_val=ceced_val,
        ceced_test=ceced_test,
        num_classes=5,
    )

