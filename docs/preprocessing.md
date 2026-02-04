# Preprocessing

The project includes a preprocessing module for preparing diabetic retinopathy datasets. Currently supports the DDR2019 dataset. The pipeline uses **OpenCV** for image I/O and transformations; images are loaded and saved in **BGR** (OpenCV convention). Middlewares receive and return BGR; convert to RGB or grayscale only when needed for a specific step.

## How the preprocess-dataset workflow works

The CLI command is `preprocess-dataset` (e.g. `uv run preprocess-dataset ddr2019`). The first argument is a **preprocessor keyword** (e.g. `ddr2019`) that selects a registered preprocessor; that preprocessor adds its own CLI options and runs the pipeline (core loop + middleware + label conversion).

### High-level workflow (text diagram)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  User runs:  uv run preprocess-dataset ddr2019 [options]                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  sam_ml.preprocessing.main()  (__init__.py)                                 │
│  • Parse CLI args (dataset, --raw-img-dir, --raw-csv-path, --processed-dir, │
│    --output-name, --min-size, --target-size)                                 │
│  • Load PreprocessingConfig (get_preprocessing_config()) for defaults       │
│  • Resolve processed_dir: --processed-dir wins, else --output-name under     │
│    data/processed, else config default                                       │
│  • Build kwargs and route by dataset name                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                    dataset == "ddr2019" │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  preprocess_ddr2019(**kwargs)  (preprocess_ddr2019.py)                       │
│  • Apply config defaults for any None path/size                              │
│  • resized_img_dir = processed_dir / "images"                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
┌───────────────────────────────────────┐   ┌───────────────────────────────────────┐
│  resize_and_copy_images(...)          │   │  convert_labels_csv(...)               │
│  • raw_img_dir → resized_img_dir      │   │  • raw_csv_path, processed_dir         │
│  • min_size, target_size from args    │   │  • processed_filenames (from step 1)   │
└───────────────────────────────────────┘   └───────────────────────────────────────┘
                    │                                       │
                    │  For each *.jpg in raw_img_dir:        │  • Read CSV (id_code, diagnosis)
                    │    • Open image, convert to RGB        │  • Rename → filename, label
                    │    • Skip if width or height < min_size│  • Filter rows to processed_filenames
                    │    • If not square: add_padding_to_    │  • Write processed_dir/labels.csv
                    │      square(img)  [black padding]      │
                    │    • Skip if after padding still       │
                    │      smaller than target_size         │
                    │    • Resize to target_size (LANCZOS)   │
                    │    • Save to resized_img_dir/filename  │
                    │  Returns: (images_processed,          │
                    │            processed_filenames)        │
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  preprocess_ddr2019() returns dict:                                          │
│  { images_processed, labels_path, processed_filenames }                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  main() prints summary (original count, processed count, skipped, labels     │
│  path) and returns 0.                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data flow summary

| Stage | Inputs | Outputs |
|-------|--------|--------|
| **CLI** | `preprocess-dataset ddr2019` + options | Parsed args, config-backed defaults |
| **main()** | Parsed args | Resolved paths, kwargs for dataset processor |
| **preprocess_ddr2019()** | raw_img_dir, raw_csv_path, processed_dir, min_size, target_size | Orchestrates image + label steps |
| **resize_and_copy_images()** | raw dir, output dir, min_size, target_size | `processed_dir/images/*.jpg`, returns (count, set of filenames) |
| **convert_labels_csv()** | raw CSV path, processed_dir, processed_filenames | `processed_dir/labels.csv` (filename, label) |

### Key files and responsibilities

- **`sam_ml/preprocessing/__init__.py`**: CLI entry (`main()`), two-phase parse (keyword → preprocessor lookup), preprocessor `add_arguments` and `run(config)`.
- **`sam_ml/preprocessing/base.py`**: `BasePreprocessor`, `BasePreprocessorConfig` (Pydantic), preprocessor registry, shared `run_core_loop()` (load BGR → middleware → write).
- **`sam_ml/preprocessing/middleware.py`**: `BaseMiddleware`, middleware registry, built-in middlewares (`default`, `paper_dual`, `resize_norm`). Contract: BGR in, list of (output_key, BGR) out.
- **`sam_ml/preprocessing/preprocess_ddr2019.py`**: `Ddr2019Preprocessor` (registered as `ddr2019`), `convert_labels_csv()`, `preprocess_ddr2019()` for backward compatibility.
- **`sam_ml/preprocessing/utils.py`**: `load_image_bgr`, `save_image_bgr`, `resize_bgr`, `add_padding_to_square_bgr` (BGR; OpenCV or PIL fallback).
- **`sam_ml/config.py`**: `PreprocessingConfig` and `get_preprocessing_config()` supply default paths, `min_size`, `target_size`, `default_middleware`, `default_output_subdir`.

### Dependencies between steps

1. **Image processing must run first**: `convert_labels_csv()` needs the `processed_filenames` set from `resize_and_copy_images()` so the output CSV only includes rows for images that were actually written.
2. **Config is used for defaults only**: Explicit CLI args and kwargs override config; paths not passed use `PreprocessingConfig` (and env vars via Pydantic).
3. **Single dataset so far**: Routing is a single `if parsed_args.dataset == "ddr2019"`; adding another dataset would require another branch and a new module similar to `preprocess_ddr2019.py`.

This workflow document reflects the current behavior for refactoring analysis (e.g. extracting a generic pipeline, splitting I/O from transforms, or supporting multiple datasets).

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

# Output to a named folder under data/processed (e.g. second version with different size)
uv run preprocess-dataset ddr2019 --output-name ddr2019_384 --target-size 384 384
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

For a full workflow diagram and data flow (for refactoring analysis), see [How the preprocess-dataset workflow works](#how-the-preprocess-dataset-workflow-works) above.

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

### Options (DDR2019 preprocessor)

- `--raw-img-dir PATH`: Path to raw images directory (default: from config)
- `--raw-csv-path PATH`: Path to raw CSV labels file (default: from config)
- `--processed-dir PATH`: Full path to output directory (overrides default and `--output-name`)
- `--output-name FOLDER`: Output folder name under `data/processed` (e.g. `ddr2019_384`). Ignored if `--processed-dir` is set.
- `--min-size SIZE`: Minimum image size in pixels (default: 512)
- `--target-size WIDTH HEIGHT`: Target image size after processing (default: 512 512)
- `--middleware NAME`: Middleware key for image processing: `default` (resize only), `paper_dual` (resized + CLAHE + CECED), `resize_norm` (default: from config)

### Examples

```bash
# Use default paths and settings
uv run preprocess-dataset ddr2019

# Custom minimum size
uv run preprocess-dataset ddr2019 --min-size 600

# Custom target size
uv run preprocess-dataset ddr2019 --target-size 256 256

# Second version with different size (output to data/processed/ddr2019_384)
uv run preprocess-dataset ddr2019 --output-name ddr2019_384 --target-size 384 384

# Full custom configuration (full path overrides --output-name)
uv run preprocess-dataset ddr2019 \
  --raw-img-dir /path/to/raw/images \
  --raw-csv-path /path/to/labels.csv \
  --processed-dir /path/to/output \
  --min-size 512 \
  --target-size 512 512
```

## Output Structure

After preprocessing, the dataset will be organized under the output directory. By default (or with `--output-name ddr2019`) the path is `data/processed/ddr2019/`. With `--output-name FOLDER` the path is `data/processed/<FOLDER>/`.

```
data/processed/ddr2019/          # or data/processed/<output-name>/
├── images/
│   ├── 20170413102628830.jpg    (resized to target size)
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

### OpenCV and BGR

- **Load**: Images are loaded as BGR numpy arrays (e.g. via `cv2.imread` or PIL fallback).
- **Middleware contract**: Each middleware receives BGR and returns a list of `(output_key, BGR array)` pairs. For color outputs, arrays must be BGR so the core can save with `cv2.imwrite` (or fallback) without conversion.
- **Save**: Core loop writes each `(output_key, img_bgr)` to `processed_dir/output_key/filename`. Convert to RGB or grayscale only inside a middleware step when needed; return BGR for color images.

### Resizing

Images are resized using high-quality resampling. The resizing algorithm:
- Only downscales or maintains size (never upscales)
- Preserves aspect ratio after padding
- Uses LANCZOS-style interpolation for smooth results

### Size Filtering

Images smaller than the minimum size are skipped entirely. This prevents:
- Upscaling artifacts
- Loss of image quality
- Introduction of noise

## Registry and middleware

The pipeline is extensible via a **preprocessor registry** (keyword → class) and a **middleware registry** (name → class). Defaults are in `sam_ml/config.py` (`PreprocessingConfig`).

- **Preprocessor registry**: The first CLI argument is a **keyword** (e.g. `ddr2019`) that maps to a preprocessor class.
- **Middleware registry**: Each preprocessor selects a **middleware** by name (e.g. `default`, `paper_dual`). Built-in middlewares:
  - **default**: Min-size filter, pad to square, resize to target (single output `images/`).
  - **paper_dual**: Same as default plus CLAHE and CECED variants; writes `images/`, `images_clahe/`, `images_ceced/` (requires OpenCV for filters).
  - **resize_norm**: Resize and normalize; single output (BGR uint8).

### Adding a custom preprocessor

1. Subclass `BasePreprocessor` and define a config model (e.g. extending `BasePreprocessorConfig`).
2. Implement `add_arguments(parser)` to add CLI options and `run(config)` to run the pipeline (e.g. via `run_core_loop()` and dataset-specific label conversion).
3. Use `@register_preprocessor("my_dataset")` and ensure the module is imported at CLI startup (e.g. in `sam_ml/preprocessing/__init__.py`).

### Adding a custom middleware

1. Subclass `BaseMiddleware` and implement `process(self, img_bgr, filename, context) -> list[tuple[str, np.ndarray]]`.
2. Contract: input and returned arrays are **BGR** for color images; each `(output_key, img)` is written to `processed_dir/output_key/filename`.
3. Use `@register_middleware("my_middleware")` and pass `--middleware my_middleware` when running the CLI (or set as default in config).

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
