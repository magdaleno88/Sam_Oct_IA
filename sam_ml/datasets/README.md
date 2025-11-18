# Datasets Module

The datasets module provides TensorFlow data loading functionality for the EyePACS diabetic retinopathy dataset. It ensures that images from the same original sample are correctly paired together when loading dual-channel (CLAHE and CECED) preprocessed images.

## Overview

The dual-channel model requires two preprocessed versions of the same original image:
- **CLAHE images**: Contrast-Limited Adaptive Histogram Equalization (299×299 for InceptionV3)
- **CECED images**: Contrast-Enhanced Canny Edge Detection (224×224 for VGG-16)

This module ensures that when loading datasets, images with the same filename (e.g., `img_00001.jpg`) from both CLAHE and CECED directories are correctly paired together, representing the same original fundus image processed with different filters.

## Key Features

- **Guaranteed Image Pairing**: Images are paired by filename to ensure the same original image is used for both channels
- **TensorFlow Integration**: Uses `tf.data.Dataset` for efficient data loading
- **Automatic Preprocessing**: Images are automatically resized and normalized
- **Performance Optimizations**: Includes caching and prefetching for faster training
- **Flexible Configuration**: Supports different batch sizes, image sizes, and label formats

## Quick Start

### Basic Usage

Load datasets for dual-channel model training:

```python
from sam_ml.datasets import load_eyepacs_dual_channel

# Load train, validation, and test datasets
train, val, test = load_eyepacs_dual_channel(
    batch_size=32,
    shuffle=True,
    seed=42
)

# Train the model
model.fit(train, validation_data=val, epochs=50)
```

### Using the Container Class

For more control, use the `DualChannelDatasets` container:

```python
from sam_ml.datasets import create_eyepacs_datasets

# Create dataset container
datasets = create_eyepacs_datasets(
    batch_size=64,
    image_size_clahe=(299, 299),
    image_size_ceced=(224, 224),
)

# Access individual datasets
clahe_train, ceced_train = datasets.get_train_split("both")
clahe_val, ceced_val = datasets.get_val_split("both")

# Or use combined datasets
train_combined = datasets.get_combined_train()
val_combined = datasets.get_combined_val()
```

## How Image Pairing Works

The dataset loader ensures correct pairing by:

1. **Scanning CLAHE directory**: Collects all image files sorted by filename
2. **Matching CECED files**: For each CLAHE file, finds the corresponding CECED file with the same relative path
3. **Verification**: Raises an error if a matching CECED file is not found
4. **Consistent ordering**: Files are processed in sorted order to ensure consistent pairing

### Example Directory Structure

```
data/processed/eyepacs_dataset/
├── CLAHE/
│   ├── train/
│   │   ├── 0/
│   │   │   ├── img_00001.jpg  ← Same original image
│   │   │   └── img_00002.jpg
│   │   └── 1/
│   │       └── img_00003.jpg
│   └── val/
│       └── ...
└── CECED/
    ├── train/
    │   ├── 0/
    │   │   ├── img_00001.jpg  ← Same original image (different filter)
    │   │   └── img_00002.jpg
    │   └── 1/
    │       └── img_00003.jpg
    └── val/
        └── ...
```

When loading, `img_00001.jpg` from CLAHE/train/0/ is automatically paired with `img_00001.jpg` from CECED/train/0/, ensuring they represent the same original image.

## Dataset Structure

### Input Format

The datasets yield batches with the following structure:

```python
((clahe_images, ceced_images), labels)
```

Where:
- `clahe_images`: Tensor of shape `(batch_size, 299, 299, 3)` - CLAHE preprocessed images
- `ceced_images`: Tensor of shape `(batch_size, 224, 224, 3)` - CECED preprocessed images
- `labels`: Tensor of shape `(batch_size, 5)` if categorical, else `(batch_size,)` - Class labels

### Image Preprocessing

Images are automatically:
- **Loaded**: Read from disk as JPEG files
- **Resized**: CLAHE to 299×299, CECED to 224×224
- **Normalized**: Pixel values scaled to [0, 1] range
- **Batched**: Grouped into batches of specified size

### Label Format

Labels can be in three formats:
- **"categorical"**: One-hot encoded vectors (default) - shape `(batch_size, 5)`
- **"int"**: Integer labels - shape `(batch_size,)`
- **"binary"**: Binary labels (not commonly used for 5-class problem)

## API Reference

### `load_eyepacs_dual_channel()`

Load EyePACS datasets combined for dual-channel model training.

**Parameters:**
- `base_path` (Path, optional): Base path to processed datasets. Default: `data/processed/{dataset_name}`
- `dataset_name` (str): Name of the dataset directory. Default: `"eyepacs_dataset"`
- `batch_size` (int): Batch size for datasets. Default: `32`
- `image_size_clahe` (Tuple[int, int]): Image size for CLAHE channel. Default: `(299, 299)`
- `image_size_ceced` (Tuple[int, int]): Image size for CECED channel. Default: `(224, 224)`
- `label_mode` (str): Label format. Default: `"categorical"`
- `shuffle` (bool): Whether to shuffle training data. Default: `True`
- `seed` (int, optional): Random seed for shuffling. Default: `42`
- `cache` (bool): Whether to cache datasets in memory. Default: `True`
- `prefetch` (bool): Whether to prefetch batches. Default: `True`

**Returns:**
- `train` (tf.data.Dataset): Training dataset
- `val` (tf.data.Dataset): Validation dataset
- `test` (tf.data.Dataset): Test dataset

**Example:**
```python
train, val, test = load_eyepacs_dual_channel(
    batch_size=64,
    shuffle=True,
    seed=42
)
```

### `load_eyepacs_datasets()`

Load individual CLAHE and CECED datasets separately.

**Returns:**
- `clahe_train`, `clahe_val`, `clahe_test`: CLAHE datasets
- `ceced_train`, `ceced_val`, `ceced_test`: CECED datasets

**Example:**
```python
(
    clahe_train, clahe_val, clahe_test,
    ceced_train, ceced_val, ceced_test
) = load_eyepacs_datasets(batch_size=32)
```

### `create_eyepacs_datasets()`

Factory function to create a `DualChannelDatasets` container.

**Returns:**
- `DualChannelDatasets`: Container with all datasets and convenience methods

**Example:**
```python
datasets = create_eyepacs_datasets(batch_size=64)
train_combined = datasets.get_combined_train()
```

### `DualChannelDatasets`

Container class for dual-channel datasets.

**Methods:**
- `get_combined_train()`: Get combined training dataset
- `get_combined_val()`: Get combined validation dataset
- `get_combined_test()`: Get combined test dataset
- `get_train_split(split)`: Get training datasets for specified channel(s)
- `get_val_split(split)`: Get validation datasets for specified channel(s)
- `get_test_split(split)`: Get test datasets for specified channel(s)

**Properties:**
- `num_classes`: Number of classes (5)
- `class_names`: Dictionary mapping class index to class name

## Integration with Model

The datasets are designed to work seamlessly with the dual-channel model:

```python
from sam_ml.datasets import load_eyepacs_dual_channel
from sam_ml.modeling.models import DualChannelDiabeticRetinopathyModel

# Load datasets
train, val, test = load_eyepacs_dual_channel(batch_size=32)

# Create model
model = DualChannelDiabeticRetinopathyModel(num_classes=5)

# Compile
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
history = model.fit(
    train,
    validation_data=val,
    epochs=50
)

# Evaluate
test_loss, test_acc = model.evaluate(test)
```

The model's `call()` method receives `[clahe_images, ceced_images]` as input, which matches the dataset structure.

## Performance Considerations

### Caching

Caching datasets in memory can significantly speed up training:

```python
train, val, test = load_eyepacs_dual_channel(
    batch_size=32,
    cache=True  # Cache in memory
)
```

**Note**: Only enable caching if you have sufficient RAM. For large datasets, consider caching to disk or disabling caching.

### Prefetching

Prefetching overlaps data preprocessing and model execution:

```python
train, val, test = load_eyepacs_dual_channel(
    batch_size=32,
    prefetch=True  # Prefetch batches
)
```

This is enabled by default and uses `tf.data.AUTOTUNE` to automatically determine the optimal buffer size.

### Shuffling

Shuffling is important for training but should be disabled for validation and test sets:

```python
train, val, test = load_eyepacs_dual_channel(
    shuffle=True,  # Shuffle training data
    seed=42        # Reproducible shuffling
)
```

Validation and test sets are automatically not shuffled.

## Error Handling

### Missing Directory

If the dataset directory doesn't exist:

```python
try:
    train, val, test = load_eyepacs_dual_channel()
except FileNotFoundError as e:
    print("Dataset not found. Run preprocessing first:")
    print("uv run preprocess-dataset eyepacs_dataset")
```

### Missing Corresponding File

If a CLAHE file doesn't have a corresponding CECED file:

```python
# Raises FileNotFoundError with detailed message
train, val, test = load_eyepacs_dual_channel()
```

This indicates an issue with the preprocessing pipeline - all images should have both CLAHE and CECED versions.

## Testing

The module includes comprehensive unit tests:

```bash
uv run pytest tests/test_datasets.py -v
```

Tests cover:
- Image pairing correctness
- Dataset structure validation
- Error handling
- Shuffling and reproducibility
- Different label formats

## Troubleshooting

### Images Not Pairing Correctly

**Problem**: Model performance is poor, suggesting images aren't paired correctly.

**Solution**: 
1. Verify that preprocessing created matching filenames in both directories
2. Check that both CLAHE and CECED directories have the same structure
3. Ensure files are sorted consistently (they should be by default)

### Out of Memory Errors

**Problem**: Running out of memory when loading datasets.

**Solution**:
1. Disable caching: `cache=False`
2. Reduce batch size: `batch_size=16` or smaller
3. Use smaller image sizes (not recommended, as it affects model performance)

### Slow Data Loading

**Problem**: Data loading is a bottleneck during training.

**Solution**:
1. Enable prefetching: `prefetch=True` (default)
2. Increase `num_parallel_calls` if using custom loading
3. Use SSD storage for faster file I/O
4. Consider caching to disk instead of memory

## See Also

- [Preprocessing Documentation](../preprocessing/README.md) - How datasets are created
- [Model Documentation](../modeling/models/README.md) - How to use the dual-channel model
- [Data Structure Documentation](../../data/README.md) - Expected directory structure

