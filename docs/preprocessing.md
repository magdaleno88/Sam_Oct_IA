# Preprocessing

The project includes a preprocessing module for preparing diabetic retinopathy datasets. Currently supports the DDR2019 dataset.

## Quick Start

Process the DDR2019 dataset using the CLI:

```bash
# Process with default settings (min-size=512, target-size=512x512)
uv run preprocess-dataset ddr2019

# Process with custom minimum size and target size
uv run preprocess-dataset ddr2019 --min-size 512 --target-size 512 512

# Process with custom paths
uv run preprocess-dataset ddr2019 \
  --raw-img-dir data/raw/ddr2019/DR_grading/DR_grading \
  --raw-csv-path data/raw/ddr2019/DR_grading.csv \
  --processed-dir data/processed/ddr2019
```

## Features

- **Minimum Size Filtering**: Only processes images with both dimensions >= 512x512
- **Automatic Padding**: Non-square images are padded to square (black padding)
- **No Upscaling**: Images are only downscaled or kept at same size (never upscaled to avoid noise)
- **Standardized Output**: All processed images are resized to 512x512
- **Label Synchronization**: CSV labels are automatically filtered to match processed images
- **Original Data Protection**: Original dataset files are never modified

## Available Datasets

- `ddr2019` - DDR2019 Diabetic Retinopathy dataset

## Preprocessing Pipeline

The preprocessing pipeline performs the following steps:

1. **Filter by Minimum Size**: Only images with `width >= 512 AND height >= 512` are processed
2. **Pad Non-Square Images**: Asymmetric images are padded with black pixels to make them square
3. **Resize to Target Size**: All images are resized to 512x512 (downscaling only, never upscaling)
4. **Filter Labels**: CSV labels are filtered to only include processed images

**Important**: Images smaller than 512x512 are skipped to avoid upscaling, which would introduce noise.

## Command-Line Interface

### Basic Usage

```bash
uv run preprocess-dataset <dataset_name> [options]
```

### Arguments

- `dataset_name`: Name of the dataset to process (currently only `ddr2019`)

### Options

- `--raw-img-dir PATH`: Path to raw images directory (default: dataset-specific)
- `--raw-csv-path PATH`: Path to raw CSV labels file (default: dataset-specific)
- `--processed-dir PATH`: Output directory for processed data (default: `data/processed/<dataset_name>`)
- `--min-size SIZE`: Minimum image size in pixels (default: 512)
- `--target-size WIDTH HEIGHT`: Target image size after processing (default: 512 512)

### Examples

```bash
# Use default paths and settings
uv run preprocess-dataset ddr2019

# Custom minimum size
uv run preprocess-dataset ddr2019 --min-size 600

# Custom target size
uv run preprocess-dataset ddr2019 --target-size 256 256

# Full custom configuration
uv run preprocess-dataset ddr2019 \
  --raw-img-dir /path/to/raw/images \
  --raw-csv-path /path/to/labels.csv \
  --processed-dir /path/to/output \
  --min-size 512 \
  --target-size 512 512
```

## Output Structure

After preprocessing, the dataset will be organized as:

```
data/processed/ddr2019/
├── images/
│   ├── 20170413102628830.jpg  (all 512x512)
│   └── ...
└── labels.csv
```

The `labels.csv` file contains:
- `filename`: Image filename
- `label`: Diagnosis label (0-4)

## Processing Statistics

At the end of preprocessing, the script prints:
- **Original dataset**: Total number of images in the raw dataset
- **Processed dataset**: Number of images successfully processed
- **Images skipped**: Number of images that were too small or would require upscaling

Example output:
```
Preprocessing complete for ddr2019:
  - Original dataset: 12524 images
  - Processed dataset: 12524 images
  - Images skipped: 0 images (too small or would require upscaling)
  - Labels saved to: data/processed/ddr2019/labels.csv
```

## Image Processing Details

### Padding

Non-square images are padded with black pixels (RGB: 0, 0, 0) to make them square before resizing. The padding is added symmetrically:

- **Wide images** (width > height): Padding added to top and bottom
- **Tall images** (height > width): Padding added to left and right

### Resizing

Images are resized using high-quality resampling. The resizing algorithm:
- Only downscales or maintains size (never upscales)
- Preserves aspect ratio after padding
- Uses bilinear interpolation for smooth results

### Size Filtering

Images smaller than the minimum size are skipped entirely. This prevents:
- Upscaling artifacts
- Loss of image quality
- Introduction of noise

## Programmatic Usage

You can also use the preprocessing functions programmatically:

```python
from sam_ml.preprocessing.preprocess_ddr2019 import preprocess_ddr2019

results = preprocess_ddr2019(
    raw_img_dir="data/raw/ddr2019/DR_grading/DR_grading",
    raw_csv_path="data/raw/ddr2019/DR_grading.csv",
    processed_dir="data/processed/ddr2019",
    min_size=512,
    target_size=(512, 512),
)

print(f"Processed {results['images_processed']} images")
print(f"Labels saved to: {results['labels_path']}")
```

## Troubleshooting

### Common Issues

**Issue**: "No JPG files found"
- **Solution**: Check that the raw image directory path is correct and contains JPG files

**Issue**: "Missing required columns"
- **Solution**: Ensure the CSV file has `id_code` and `diagnosis` columns

**Issue**: All images skipped
- **Solution**: Check that images meet the minimum size requirement (default: 512x512)

### Performance Tips

- Processing large datasets can take time. Monitor progress in the console output.
- Ensure sufficient disk space for processed images.
- Process datasets in batches if memory is limited.
