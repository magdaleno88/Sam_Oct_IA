# Preprocessing Module

The preprocessing module provides a polymorphic, extensible system for processing diabetic retinopathy datasets. It uses an object-oriented design pattern that allows easy addition of new dataset processors while maintaining a consistent interface.

## Overview

The preprocessing module implements a factory pattern with an abstract base class (`DatasetProcessor`) that defines the interface for all dataset processors. Each concrete processor handles a specific dataset format and implements the required methods for extraction, labeling, splitting, and processing.

## Architecture

### Components

- **`base.py`**: Abstract base class `DatasetProcessor` defining the interface
- **`utils.py`**: Shared utility functions for image preprocessing (CLAHE and CECED)
- **`eyepacs_dataset.py`**: Concrete processor for EyePACS dataset
- **`__init__.py`**: Factory functions and CLI entry point

### Design Pattern

The module uses **polymorphism** and **factory pattern**:
- All processors inherit from `DatasetProcessor` (abstract base class)
- Factory function `create_processor()` selects the appropriate processor based on dataset name
- Processors are registered in a registry and can be added/removed dynamically
- Each processor is decoupled and independent

## Quick Start

### CLI Usage

The simplest way to use the preprocessor is via the command-line interface:

```bash
# Process EyePACS dataset with default paths
uv run preprocess-dataset eyepacs_dataset

# With custom paths
uv run preprocess-dataset eyepacs_dataset \
    --raw-dir /path/to/raw \
    --processed-dir /path/to/processed

# Custom split ratios
uv run preprocess-dataset eyepacs_dataset \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1
```

### Programmatic Usage

```python
from sam_ml.preprocessing import create_processor

# Create processor for EyePACS dataset
processor = create_processor("eyepacs_dataset")

# Process the dataset
processor.process_dataset()
```

## Available Datasets

To see all available dataset processors:

```python
from sam_ml.preprocessing import list_available_datasets

datasets = list_available_datasets()
print(datasets)  # ['eyepacs_dataset']
```

## Dataset-Specific Documentation

### EyePACS Dataset

**Dataset Name**: `eyepacs_dataset`

**Data Source**: Hugging Face dataset `bumbledeep/eyepacs`

The processor automatically loads the dataset from Hugging Face - no local raw data files required!

**Dataset Information**:
- **Source**: Hugging Face (`bumbledeep/eyepacs`)
- **Split**: `train` (approximately 35,000 samples)
- **Features**:
  - `image`: PIL.Image object in RGB format
  - `label`: Integer in {0, 1, 2, 3, 4}
- **Image Properties**: RGB fundus photographs with resolutions ranging from 720×720 to 2048×2048

**Class Mapping**:
The dataset uses numeric labels (0-4) for TensorFlow/Keras compatibility. The mapping is:

| Numeric Label | Class Name | Description |
|--------------|------------|-------------|
| `0` | No Diabetic Retinopathy | No signs of diabetic retinopathy |
| `1` | Mild Retinopathy | Mild non-proliferative diabetic retinopathy |
| `2` | Moderate Retinopathy | Moderate non-proliferative diabetic retinopathy |
| `3` | Severe Retinopathy | Severe non-proliferative diabetic retinopathy |
| `4` | Proliferative Retinopathy | Proliferative diabetic retinopathy |

**Note**: The processor creates only numeric label folders (`0`, `1`, `2`, `3`, `4`) for compatibility with `tf.keras.utils.image_dataset_from_directory`. If you see named folders (e.g., `mild_retinopathy`, `no_diabetic_retinopathy`) from a previous run, you can safely delete them as they are not used by the current processor.

**Processing Steps**:
1. Loads dataset from Hugging Face automatically
2. Extracts labels from dataset samples
3. Creates directory structure for processed data
4. Splits dataset into train/val/test (default: 70/15/15)
5. Converts PIL Images to OpenCV BGR format
6. Applies CLAHE preprocessing to all images
7. Applies CECED preprocessing to all images
8. Saves processed images to organized directory structure

**Output Structure**:
```
data/processed/eyepacs_dataset/
├── CLAHE/
│   ├── train/
│   │   ├── 0/          # No Diabetic Retinopathy
│   │   ├── 1/          # Mild Retinopathy
│   │   ├── 2/          # Moderate Retinopathy
│   │   ├── 3/          # Severe Retinopathy
│   │   └── 4/          # Proliferative Retinopathy
│   ├── val/
│   │   ├── 0/
│   │   ├── 1/
│   │   ├── 2/
│   │   ├── 3/
│   │   └── 4/
│   └── test/
│       ├── 0/
│       ├── 1/
│       ├── 2/
│       ├── 3/
│       └── 4/
└── CECED/
    ├── train/
    │   ├── 0/
    │   ├── 1/
    │   ├── 2/
    │   ├── 3/
    │   └── 4/
    ├── val/
    │   ├── 0/
    │   ├── 1/
    │   ├── 2/
    │   ├── 3/
    │   └── 4/
    └── test/
        ├── 0/
        ├── 1/
        ├── 2/
        ├── 3/
        └── 4/
```

**Important**: Only numeric folders (`0`, `1`, `2`, `3`, `4`) are created. Any named folders (e.g., `mild_retinopathy`, `no_diabetic_retinopathy`) are from old runs and should be removed.

**TensorFlow Loading**:
The output structure is compatible with `tf.keras.utils.image_dataset_from_directory`:

```python
import tensorflow as tf

# Load CLAHE images
train_ds_clahe = tf.keras.utils.image_dataset_from_directory(
    "data/processed/eyepacs_dataset/CLAHE/train",
    image_size=(299, 299),  # For InceptionV3
    batch_size=32,
    label_mode="categorical"
)

# Load CECED images
train_ds_ceced = tf.keras.utils.image_dataset_from_directory(
    "data/processed/eyepacs_dataset/CECED/train",
    image_size=(224, 224),  # For VGG-16
    batch_size=32,
    label_mode="categorical"
)
```

## Preprocessing Techniques

### CLAHE (Contrast-Limited Adaptive Histogram Equalization)

CLAHE enhances image contrast by applying histogram equalization locally. It's applied to the L channel of the LAB color space to preserve color information while improving contrast.

**Usage**:
```python
from sam_ml.preprocessing import apply_clahe_bgr
import cv2

img_bgr = cv2.imread("image.jpg")
processed = apply_clahe_bgr(img_bgr)
cv2.imwrite("processed.jpg", processed)
```

### CECED (Contrast-Enhanced Canny Edge Detection)

CECED extracts edges from images using:
1. Conversion to grayscale
2. CLAHE enhancement
3. Normalization
4. Gaussian blur for noise reduction
5. Canny edge detection
6. Conversion to 3-channel format

**Usage**:
```python
from sam_ml.preprocessing import apply_ceced_bgr_3ch
import cv2

img_bgr = cv2.imread("image.jpg")
edges = apply_ceced_bgr_3ch(img_bgr)
cv2.imwrite("edges.jpg", edges)
```

## Advanced Usage

### Custom Paths

```python
from pathlib import Path
from sam_ml.preprocessing import create_processor

processor = create_processor(
    dataset_name="eyepacs_dataset",
    raw_dir=Path("/custom/path/to/raw"),
    processed_dir=Path("/custom/path/to/processed"),
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_seed=123
)

processor.process_dataset()
```

### Creating a Custom Processor

To add support for a new dataset, create a processor class:

```python
from pathlib import Path
from typing import Dict, List, Tuple

from sam_ml.preprocessing.base import DatasetProcessor
from sam_ml.preprocessing import register_processor

class MyDatasetProcessor(DatasetProcessor):
    """Processor for my custom dataset."""
    
    @property
    def supported_dataset_name(self) -> str:
        return "my_dataset"
    
    def extract_raw_data(self) -> Tuple[Path, Path]:
        # Implement extraction logic
        train_dir = self.raw_dir / "train"
        test_dir = self.raw_dir / "test"
        return train_dir, test_dir
    
    def load_labels(self) -> Dict[str, int]:
        # Implement label loading logic
        return {}
    
    def create_directory_structure(self) -> Dict[str, Path]:
        # Implement directory creation
        return {}
    
    def split_dataset(self, image_paths: List[Path], labels: Dict[str, int]):
        # Implement splitting logic
        return ([], [], [])
    
    def process_dataset(self) -> None:
        # Implement full processing pipeline
        pass

# Register the processor
register_processor("my_dataset", MyDatasetProcessor)
```

## CLI Reference

### Command: `preprocess-dataset`

**Synopsis**:
```bash
preprocess-dataset DATASET_NAME [OPTIONS]
```

**Positional Arguments**:
- `DATASET_NAME`: Name of the dataset to process (required)

**Options**:
- `--raw-dir RAW_DIR`: Directory containing raw dataset files. Default: `data/raw/{dataset_name}`
- `--processed-dir PROCESSED_DIR`: Directory for processed dataset output. Default: `data/processed`
- `--train-ratio TRAIN_RATIO`: Proportion of data for training (default: 0.7)
- `--val-ratio VAL_RATIO`: Proportion of data for validation (default: 0.15)
- `--test-ratio TEST_RATIO`: Proportion of data for testing (default: 0.15)
- `--random-seed RANDOM_SEED`: Random seed for reproducibility (default: 42)
- `-h, --help`: Show help message

**Examples**:
```bash
# Basic usage
preprocess-dataset eyepacs_dataset

# Custom paths and ratios
preprocess-dataset eyepacs_dataset \
    --raw-dir /data/raw \
    --processed-dir /data/processed \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1

# Different random seed
preprocess-dataset eyepacs_dataset --random-seed 123
```

## API Reference

### Factory Functions

#### `create_processor(dataset_name, ...)`

Creates a processor instance for the specified dataset.

**Parameters**:
- `dataset_name` (str): Name of the dataset (required)
- `raw_dir` (Path, optional): Raw data directory. Default: `data/raw/{dataset_name}`
- `processed_dir` (Path, optional): Processed data directory. Default: `data/processed`
- `train_ratio` (float): Training data proportion (default: 0.7)
- `val_ratio` (float): Validation data proportion (default: 0.15)
- `test_ratio` (float): Test data proportion (default: 0.15)
- `random_seed` (int): Random seed (default: 42)

**Returns**: `DatasetProcessor` instance

**Raises**: `ValueError` if dataset name is not found

#### `get_processor(dataset_name)`

Gets the processor class for a dataset name.

**Parameters**:
- `dataset_name` (str): Name of the dataset

**Returns**: `Type[DatasetProcessor]`

**Raises**: `ValueError` if dataset name is not found

#### `list_available_datasets()`

Lists all available dataset processors.

**Returns**: `List[str]` of dataset names

#### `register_processor(dataset_name, processor_class)`

Registers a new processor dynamically.

**Parameters**:
- `dataset_name` (str): Name of the dataset
- `processor_class` (Type[DatasetProcessor]): Processor class

### Abstract Base Class

#### `DatasetProcessor`

Abstract base class for all dataset processors.

**Methods** (must be implemented by subclasses):
- `extract_raw_data() -> Tuple[Path, Path]`: Extract raw data files
- `load_labels() -> Dict[str, int]`: Load dataset labels
- `create_directory_structure() -> Dict[str, Path]`: Create output directories
- `split_dataset(image_paths, labels) -> Tuple[List, List, List]`: Split dataset
- `process_dataset() -> None`: Main processing pipeline

**Properties**:
- `supported_dataset_name: str`: Dataset name this processor supports

## Troubleshooting

### Common Issues

**Issue**: `No processor found for dataset 'X'`
- **Solution**: Check available datasets with `list_available_datasets()`. Ensure the dataset name matches exactly.

**Issue**: `Ratios must sum to 1.0`
- **Solution**: Ensure `train_ratio + val_ratio + test_ratio == 1.0`

**Issue**: `ConnectionError` or `DatasetNotFoundError` when loading from Hugging Face
- **Solution**: Ensure you have internet connectivity. The dataset will be downloaded automatically on first use. Check that the dataset name `bumbledeep/eyepacs` is correct.

**Issue**: `ModuleNotFoundError: No module named 'datasets'`
- **Solution**: Install the datasets library: `uv sync` (it's included in project dependencies)

### Performance Tips

- The preprocessing can be time-consuming for large datasets (~35,000 images). Processing typically takes several hours.
- The Hugging Face dataset is downloaded automatically on first use and cached locally.
- Progress bars (using `tqdm`) show real-time processing status for label extraction and image processing.
- Use `--random-seed` for reproducible dataset splits.
- The dataset is loaded into memory - ensure sufficient RAM (~8GB+ recommended).

## See Also

- [Dataset Structure Documentation](../data/README.md) - Expected dataset organization
- [Model Documentation](../modeling/models/README.md) - Model architecture and usage
- [Main README](../../README.md) - Project overview

