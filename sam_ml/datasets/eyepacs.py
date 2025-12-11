"""EyePACS dataset loader for TensorFlow."""

from pathlib import Path
from typing import Optional, Tuple

import tensorflow as tf


def load_eyepacs_datasets(
    base_path: Optional[Path] = None,
    dataset_name: str = "eyepacs_dataset",
    batch_size: int = 32,
    image_size_clahe: Tuple[int, int] = (299, 299),
    image_size_ceced: Tuple[int, int] = (224, 224),
    label_mode: str = "categorical",
    shuffle: bool = True,
    seed: Optional[int] = 42,
    validation_split: Optional[float] = None,
    subset: Optional[str] = None,
    interpolation: str = "bilinear",
    follow_links: bool = False,
    crop_to_aspect_ratio: bool = False,
    cache: bool = True,
    prefetch: bool = True,
    num_parallel_calls: Optional[int] = None,
) -> Tuple[
    tf.data.Dataset,  # clahe_train
    tf.data.Dataset,  # clahe_val
    tf.data.Dataset,  # clahe_test
    tf.data.Dataset,  # ceced_train
    tf.data.Dataset,  # ceced_val
    tf.data.Dataset,  # ceced_test
]:
    """
    Load EyePACS datasets for both CLAHE and CECED preprocessing channels.
    
    This function loads train, validation, and test splits for both preprocessing
    channels from the processed dataset directory structure.
    
    Args:
        base_path: Base path to processed datasets. If None, uses default:
                   data/processed/{dataset_name}
        dataset_name: Name of the dataset directory (default: "eyepacs_dataset")
        batch_size: Batch size for datasets (default: 32)
        image_size_clahe: Image size for CLAHE channel (default: (299, 299) for InceptionV3)
        image_size_ceced: Image size for CECED channel (default: (224, 224) for VGG-16)
        label_mode: Label format - "categorical", "int", or "binary" (default: "categorical")
        shuffle: Whether to shuffle the data (default: True)
        seed: Random seed for shuffling and transformations (default: 42)
        validation_split: Not used (splits are pre-defined in directory structure)
        subset: Not used (splits are pre-defined in directory structure)
        interpolation: Interpolation method for resizing (default: "bilinear")
        follow_links: Whether to follow symlinks (default: False)
        crop_to_aspect_ratio: Whether to crop to aspect ratio (default: False)
        cache: Whether to cache datasets in memory (default: True)
        prefetch: Whether to prefetch batches (default: True)
        num_parallel_calls: Number of parallel calls for data loading (default: None = auto)
        
    Returns:
        Tuple of 6 datasets:
        - clahe_train: Training dataset for CLAHE channel
        - clahe_val: Validation dataset for CLAHE channel
        - clahe_test: Test dataset for CLAHE channel
        - ceced_train: Training dataset for CECED channel
        - ceced_val: Validation dataset for CECED channel
        - ceced_test: Test dataset for CECED channel
        
    Example:
        >>> from sam_ml.datasets import load_eyepacs_datasets
        >>> clahe_train, clahe_val, clahe_test, ceced_train, ceced_val, ceced_test = \\
        ...     load_eyepacs_datasets(batch_size=64)
        >>> # Use datasets for training
        >>> model.fit([clahe_train, ceced_train], validation_data=([clahe_val, ceced_val]))
    """
    # Set default base path if not provided
    if base_path is None:
        base_path = Path("data/processed") / dataset_name
    else:
        base_path = Path(base_path)
    
    # Define paths for each channel and split
    clahe_train_path = base_path / "CLAHE" / "train"
    clahe_val_path = base_path / "CLAHE" / "val"
    clahe_test_path = base_path / "CLAHE" / "test"
    ceced_train_path = base_path / "CECED" / "train"
    ceced_val_path = base_path / "CECED" / "val"
    ceced_test_path = base_path / "CECED" / "test"
    
    # Validate paths exist
    for path in [clahe_train_path, clahe_val_path, clahe_test_path,
                 ceced_train_path, ceced_val_path, ceced_test_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {path}\n"
                f"Please run the preprocessing pipeline first to generate the dataset."
            )
    
    # Load CLAHE datasets
    clahe_train = tf.keras.utils.image_dataset_from_directory(
        directory=str(clahe_train_path),
        labels="inferred",
        label_mode=label_mode,
        class_names=None,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size_clahe,
        shuffle=shuffle,
        seed=seed,
        validation_split=validation_split,
        subset=subset,
        interpolation=interpolation,
        follow_links=follow_links,
        crop_to_aspect_ratio=crop_to_aspect_ratio,
    )
    
    clahe_val = tf.keras.utils.image_dataset_from_directory(
        directory=str(clahe_val_path),
        labels="inferred",
        label_mode=label_mode,
        class_names=None,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size_clahe,
        shuffle=False,  # Don't shuffle validation
        seed=seed,
        validation_split=validation_split,
        subset=subset,
        interpolation=interpolation,
        follow_links=follow_links,
        crop_to_aspect_ratio=crop_to_aspect_ratio,
    )
    
    clahe_test = tf.keras.utils.image_dataset_from_directory(
        directory=str(clahe_test_path),
        labels="inferred",
        label_mode=label_mode,
        class_names=None,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size_clahe,
        shuffle=False,  # Don't shuffle test
        seed=seed,
        validation_split=validation_split,
        subset=subset,
        interpolation=interpolation,
        follow_links=follow_links,
        crop_to_aspect_ratio=crop_to_aspect_ratio,
    )
    
    # Load CECED datasets
    ceced_train = tf.keras.utils.image_dataset_from_directory(
        directory=str(ceced_train_path),
        labels="inferred",
        label_mode=label_mode,
        class_names=None,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size_ceced,
        shuffle=shuffle,
        seed=seed,
        validation_split=validation_split,
        subset=subset,
        interpolation=interpolation,
        follow_links=follow_links,
        crop_to_aspect_ratio=crop_to_aspect_ratio,
    )
    
    ceced_val = tf.keras.utils.image_dataset_from_directory(
        directory=str(ceced_val_path),
        labels="inferred",
        label_mode=label_mode,
        class_names=None,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size_ceced,
        shuffle=False,  # Don't shuffle validation
        seed=seed,
        validation_split=validation_split,
        subset=subset,
        interpolation=interpolation,
        follow_links=follow_links,
        crop_to_aspect_ratio=crop_to_aspect_ratio,
    )
    
    ceced_test = tf.keras.utils.image_dataset_from_directory(
        directory=str(ceced_test_path),
        labels="inferred",
        label_mode=label_mode,
        class_names=None,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size_ceced,
        shuffle=False,  # Don't shuffle test
        seed=seed,
        validation_split=validation_split,
        subset=subset,
        interpolation=interpolation,
        follow_links=follow_links,
        crop_to_aspect_ratio=crop_to_aspect_ratio,
    )
    
    # Apply TensorFlow data loading optimizations
    def optimize_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Apply optimizations to a dataset."""
        # Cache dataset in memory for faster access
        if cache:
            dataset = dataset.cache()
        
        # Prefetch batches to overlap data preprocessing and model execution
        if prefetch:
            dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return dataset
    
    clahe_train = optimize_dataset(clahe_train)
    clahe_val = optimize_dataset(clahe_val)
    clahe_test = optimize_dataset(clahe_test)
    ceced_train = optimize_dataset(ceced_train)
    ceced_val = optimize_dataset(ceced_val)
    ceced_test = optimize_dataset(ceced_test)
    
    return (
        clahe_train,
        clahe_val,
        clahe_test,
        ceced_train,
        ceced_val,
        ceced_test,
    )


def _create_paired_dataset(
    clahe_dir: Path,
    ceced_dir: Path,
    image_size_clahe: Tuple[int, int],
    image_size_ceced: Tuple[int, int],
    batch_size: int,
    label_mode: str,
    shuffle: bool,
    seed: Optional[int],
) -> tf.data.Dataset:
    """
    Create a dataset that pairs images from CLAHE and CECED directories by filename.
    
    This ensures that the same original image (processed with different filters)
    is correctly paired together.
    
    Args:
        clahe_dir: Directory containing CLAHE images
        ceced_dir: Directory containing CECED images
        image_size_clahe: Target size for CLAHE images
        image_size_ceced: Target size for CECED images
        batch_size: Batch size
        label_mode: Label format ("categorical", "int", or "binary")
        shuffle: Whether to shuffle
        seed: Random seed
        
    Returns:
        Dataset yielding ((clahe_image, ceced_image), label) tuples
    """
    # Get all image files from CLAHE directory, sorted by filename
    # This ensures consistent ordering
    clahe_files = []
    clahe_labels = []
    
    # Walk through class directories
    for class_dir in sorted(clahe_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        
        label = int(class_dir.name)  # Class folder is numeric (0-4)
        
        # Get all image files in this class directory, sorted
        for img_file in sorted(class_dir.glob("*.jpg")):
            clahe_files.append(img_file)
            clahe_labels.append(label)
    
    # Verify corresponding CECED files exist
    ceced_files = []
    for clahe_file in clahe_files:
        # Construct corresponding CECED file path
        # Structure: {split}/{label}/img_{idx:05d}.jpg
        relative_path = clahe_file.relative_to(clahe_dir)
        ceced_file = ceced_dir / relative_path
        
        if not ceced_file.exists():
            raise FileNotFoundError(
                f"Corresponding CECED file not found: {ceced_file}\n"
                f"Expected for CLAHE file: {clahe_file}"
            )
        
        ceced_files.append(ceced_file)
    
    # Convert Path objects to strings for TensorFlow
    clahe_file_paths = [str(f) for f in clahe_files]
    ceced_file_paths = [str(f) for f in ceced_files]
    
    # Create datasets from file paths (as strings)
    def load_and_preprocess_image(file_path: str, target_size: Tuple[int, int]) -> tf.Tensor:
        """Load and preprocess a single image."""
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, target_size, method="bilinear")
        img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0, 1]
        return img
    
    # Create dataset from file paths
    clahe_ds = tf.data.Dataset.from_tensor_slices(clahe_file_paths)
    ceced_ds = tf.data.Dataset.from_tensor_slices(ceced_file_paths)
    labels_ds = tf.data.Dataset.from_tensor_slices(clahe_labels)
    
    # Load and preprocess images
    clahe_ds = clahe_ds.map(
        lambda x: load_and_preprocess_image(x, image_size_clahe),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ceced_ds = ceced_ds.map(
        lambda x: load_and_preprocess_image(x, image_size_ceced),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Convert labels to categorical if needed
    if label_mode == "categorical":
        num_classes = 5
        labels_ds = labels_ds.map(
            lambda x: tf.one_hot(x, depth=num_classes),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Zip datasets together: ((clahe_image, ceced_image), label)
    # TensorFlow will handle the tuple conversion for multi-input models
    # The model's call() method receives inputs as a list, but TensorFlow
    # automatically converts tuples to lists when needed
    paired_ds = tf.data.Dataset.zip((tf.data.Dataset.zip((clahe_ds, ceced_ds)), labels_ds))
    
    # Shuffle if requested
    # Use a balanced buffer size for good shuffling quality while maintaining memory efficiency
    # A buffer of 2,000-3,000 provides excellent shuffling without excessive memory usage
    # This is a good compromise: better randomization than 1,000, but still memory-efficient
    if shuffle:
        # Balanced buffer size: 2000 samples
        # - Provides good randomization quality (better than 1,000)
        # - Still memory-efficient (much better than 10,000+)
        # - For datasets smaller than 2,000, uses the full dataset size
        buffer_size = min(2000, len(clahe_files))
        paired_ds = paired_ds.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=True)
    
    # Batch before other optimizations
    # Batching reduces the number of elements and makes subsequent operations more efficient
    paired_ds = paired_ds.batch(batch_size)
    
    # Prefetch for better performance (overlaps data loading with model execution)
    # Note: We don't cache here because:
    # 1. Image datasets are too large to cache in memory efficiently
    # 2. Cache should be applied after batching if needed, but for large datasets
    #    it's better to rely on prefetch and file system caching
    paired_ds = paired_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return paired_ds


def load_eyepacs_dual_channel(
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
) -> Tuple[
    tf.data.Dataset,  # train (combined)
    tf.data.Dataset,  # val (combined)
    tf.data.Dataset,  # test (combined)
]:
    """
    Load EyePACS datasets combined for dual-channel model training.
    
    This function ensures that images from the same original sample are correctly
    paired together. Images are matched by filename (e.g., img_00001.jpg in both
    CLAHE and CECED directories represent the same original image).
    
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
        Tuple of 3 combined datasets:
        - train: Combined training dataset yielding ((clahe_images, ceced_images), labels)
        - val: Combined validation dataset yielding ((clahe_images, ceced_images), labels)
        - test: Combined test dataset yielding ((clahe_images, ceced_images), labels)
        
    Note:
        The returned datasets have a structure:
        - Each batch: ((clahe_batch, ceced_batch), labels_batch)
        - clahe_batch: shape (batch_size, 299, 299, 3)
        - ceced_batch: shape (batch_size, 224, 224, 3)
        - labels_batch: shape (batch_size, 5) if categorical, else (batch_size,)
        
        For model.fit(), the model should accept two inputs:
        ```python
        model.fit(train, validation_data=val, epochs=50)
        ```
        The model's call method receives: [clahe_images, ceced_images]
        
    Example:
        >>> from sam_ml.datasets import load_eyepacs_dual_channel
        >>> train, val, test = load_eyepacs_dual_channel(batch_size=64)
        >>> model.fit(train, validation_data=val, epochs=50)
    """
    # Set default base path if not provided
    if base_path is None:
        base_path = Path("data/processed") / dataset_name
    else:
        base_path = Path(base_path)
    
    # Define paths for each channel and split
    clahe_train_path = base_path / "CLAHE" / "train"
    clahe_val_path = base_path / "CLAHE" / "val"
    clahe_test_path = base_path / "CLAHE" / "test"
    ceced_train_path = base_path / "CECED" / "train"
    ceced_val_path = base_path / "CECED" / "val"
    ceced_test_path = base_path / "CECED" / "test"
    
    # Validate paths exist
    for path in [clahe_train_path, clahe_val_path, clahe_test_path,
                 ceced_train_path, ceced_val_path, ceced_test_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {path}\n"
                f"Please run the preprocessing pipeline first to generate the dataset."
            )
    
    # Create paired datasets for each split
    train = _create_paired_dataset(
        clahe_dir=clahe_train_path,
        ceced_dir=ceced_train_path,
        image_size_clahe=image_size_clahe,
        image_size_ceced=image_size_ceced,
        batch_size=batch_size,
        label_mode=label_mode,
        shuffle=shuffle,
        seed=seed,
    )
    
    val = _create_paired_dataset(
        clahe_dir=clahe_val_path,
        ceced_dir=ceced_val_path,
        image_size_clahe=image_size_clahe,
        image_size_ceced=image_size_ceced,
        batch_size=batch_size,
        label_mode=label_mode,
        shuffle=False,  # Don't shuffle validation
        seed=seed,
    )
    
    test = _create_paired_dataset(
        clahe_dir=clahe_test_path,
        ceced_dir=ceced_test_path,
        image_size_clahe=image_size_clahe,
        image_size_ceced=image_size_ceced,
        batch_size=batch_size,
        label_mode=label_mode,
        shuffle=False,  # Don't shuffle test
        seed=seed,
    )
    
    # Apply optimizations
    # Simple, effective strategy based on dataset characteristics:
    # 
    # 1. Training set: NO cache - Large dataset (24K+ images), shuffled each epoch.
    #    - Prefetch is sufficient for efficient I/O
    #    - Caching doesn't help because: first epoch still reads from disk,
    #      shuffling changes order each epoch, and adds complexity/lockfile issues
    # 
    # 2. Validation/Test sets: Memory cache - Small datasets (~5K samples), reused frequently
    #    - Significant performance benefit with minimal memory cost
    #    - Read multiple times per epoch (validation) or multiple evaluations (test)
    # 
    # 3. Prefetch: Always enabled - Overlaps I/O with computation for all datasets
    if cache:
        # Only cache validation and test sets (small, reused frequently)
        val = val.cache()
        test = test.cache()
        # Training set: No cache - rely on prefetch and OS file system caching
    
    if prefetch:
        train = train.prefetch(buffer_size=tf.data.AUTOTUNE)
        val = val.prefetch(buffer_size=tf.data.AUTOTUNE)
        test = test.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train, val, test

