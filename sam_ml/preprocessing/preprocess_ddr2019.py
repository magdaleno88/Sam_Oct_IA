"""Preprocessing script for DDR2019 dataset.

This script processes the raw DDR2019 dataset by:
1. Filtering images by minimum size (>= 512x512)
2. Adding padding to non-square images to make them square
3. Resizing all images to 512x512
4. Converting the label CSV to the standard format (filename, label)

Image processing uses the middleware pipeline (OpenCV BGR) when running
via run_middleware_pipeline / resize_and_copy_images. The PIL-based
add_padding_to_square is kept for backward compatibility where needed.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from PIL import Image
from tqdm import tqdm

from sam_ml.config import get_preprocessing_config
from sam_ml.preprocessing.base import (
    BasePreprocessor,
    BasePreprocessorConfig,
    register_preprocessor,
    run_core_loop,
)
from sam_ml.preprocessing.middleware import (
    MiddlewareContext,
    get_middleware,
)
from sam_ml.preprocessing.utils import load_image_bgr, save_image_bgr


def _get_default_paths() -> tuple[Path, Path, Path]:
    """Get default paths from config."""
    config = get_preprocessing_config()
    return (
        config.ddr2019_raw_img_dir,
        config.ddr2019_raw_csv_path,
        config.ddr2019_processed_dir,
    )


def _get_default_sizes() -> tuple[int, tuple[int, int]]:
    """Get default sizes from config."""
    config = get_preprocessing_config()
    return config.min_size, config.target_size


def run_middleware_pipeline(
    raw_img_dir: str,
    processed_dir: str,
    middleware_key: str = "default",
    min_size: int = 512,
    target_size: tuple[int, int] = (512, 512),
    output_key: str = "images",
) -> tuple[int, set[str]]:
    """Run the middleware pipeline on all JPGs in raw_img_dir (OpenCV BGR).

    Loads each image with cv2.imread (BGR), runs the registered middleware,
    and writes each (output_key, img) to processed_dir/output_key/filename.
    Returns (number of images that produced at least one output, set of filenames).

    Args:
        raw_img_dir: Path to raw images directory.
        processed_dir: Base output directory (subdirs per output_key).
        middleware_key: Key for middleware registry (e.g. "default").
        min_size: Minimum size passed in context to middleware.
        target_size: Target size passed in context to middleware.
        output_key: Subdir name for single-output middleware (e.g. "images").

    Returns:
        Tuple of (processed_count, processed_filenames).
    """
    raw_path = Path(raw_img_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw images directory not found: {raw_img_dir}")
    image_files = list(raw_path.glob("*.jpg"))
    if not image_files:
        raise ValueError(f"No JPG files found in {raw_img_dir}")

    config = get_preprocessing_config()
    middleware = get_middleware(
        middleware_key,
        min_size=min_size,
        target_size=target_size,
        output_key=output_key,
    )
    context: MiddlewareContext = {
        "min_size": min_size,
        "target_size": target_size,
    }
    processed_dir_path = Path(processed_dir)
    processed_count = 0
    processed_filenames: set[str] = set()
    skipped_small_count = 0

    for img_file in tqdm(image_files, desc="Processing images"):
        try:
            img_bgr = load_image_bgr(img_file)
            if img_bgr is None:
                print(f"Warning: Could not read {img_file.name}")
                continue
            results = middleware.process(img_bgr, img_file.name, context)
            if not results:
                skipped_small_count += 1
                continue
            for out_key, out_img in results:
                out_dir = processed_dir_path / out_key
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / img_file.name
                if not save_image_bgr(out_path, out_img):
                    print(f"Warning: Could not write {out_path}")
            processed_count += 1
            processed_filenames.add(img_file.name)
        except Exception as e:
            print(f"Warning: Failed to process {img_file.name}: {e}")
            continue

    if skipped_small_count > 0:
        print(
            f"Info: Skipped {skipped_small_count} images smaller than {min_size}x{min_size}"
        )
    return processed_count, processed_filenames


def add_padding_to_square(img: Image.Image) -> Image.Image:
    """Add padding to make an image square.
    
    Pads the smaller dimension (width or height) to match the larger one.
    Uses black padding (RGB 0, 0, 0).
    
    Args:
        img: PIL Image to pad.
    
    Returns:
        Square PIL Image with padding added.
    """
    width, height = img.size
    
    # If already square, return as is
    if width == height:
        return img
    
    # Determine the target square size (max of width and height)
    target_size = max(width, height)
    
    # Create a new square image with black background
    square_img = Image.new("RGB", (target_size, target_size), color=(0, 0, 0))
    
    # Calculate padding offsets to center the image
    if width < height:
        # Pad horizontally (left and right)
        x_offset = (target_size - width) // 2
        y_offset = 0
    else:
        # Pad vertically (top and bottom)
        x_offset = 0
        y_offset = (target_size - height) // 2
    
    # Paste the original image onto the square canvas
    square_img.paste(img, (x_offset, y_offset))
    
    return square_img


def resize_and_copy_images(
    raw_img_dir: Optional[str] = None,
    resized_img_dir: Optional[str] = None,
    min_size: int | None = None,
    target_size: tuple[int, int] | None = None,
) -> tuple[int, set[str]]:
    """Resize and copy images from raw directory to processed directory.

    Uses the default middleware (OpenCV BGR): min-size filter, pad to square, resize.
    Only processes images with both dimensions >= min_size. Smaller images are skipped.
    Non-square images are padded to make them square, then all images are resized to target_size.

    Args:
        raw_img_dir: Path to raw images directory. Defaults to config value.
        resized_img_dir: Path to output directory for resized images. Defaults to config value.
        min_size: Minimum size (width and height) required to process an image. Defaults to config value.
        target_size: Target size (width, height) for resizing. Defaults to config value.

    Returns:
        Tuple of (number of images processed, set of processed filenames).

    Raises:
        FileNotFoundError: If raw_img_dir doesn't exist.
        OSError: If unable to create output directory or write images.
    """
    default_min_size, default_target_size = _get_default_sizes()
    default_raw_img_dir, _, default_processed_dir = _get_default_paths()
    config = get_preprocessing_config()

    if min_size is None:
        min_size = default_min_size
    if target_size is None:
        target_size = default_target_size
    if raw_img_dir is None:
        raw_img_dir = str(default_raw_img_dir)
    if resized_img_dir is None:
        resized_img_dir = str(default_processed_dir / config.default_output_subdir)

    # resized_img_dir is processed_dir/images; pipeline expects processed_dir
    processed_dir = str(Path(resized_img_dir).parent)
    return run_middleware_pipeline(
        raw_img_dir=raw_img_dir,
        processed_dir=processed_dir,
        middleware_key=config.default_middleware,
        min_size=min_size,
        target_size=target_size,
        output_key=config.default_output_subdir,
    )


def convert_labels_csv(
    raw_csv_path: Optional[str] = None,
    processed_dir: Optional[str] = None,
    output_filename: str = "labels.csv",
    processed_filenames: Optional[set[str]] = None,
) -> Path:
    """Convert DDR2019 label CSV to standard format.
    
    Converts the CSV from format (id_code, diagnosis) to (filename, label).
    Only includes labels for images that were actually processed (size >= 512x512).
    
    Args:
        raw_csv_path: Path to input CSV file. Defaults to config value.
        processed_dir: Directory to save output CSV. Defaults to config value.
        output_filename: Name of output CSV file. Defaults to "labels.csv".
        processed_filenames: Set of filenames that were successfully processed.
                            If None, includes all rows from CSV.
    
    Returns:
        Path to the created labels.csv file.
    
    Raises:
        FileNotFoundError: If raw_csv_path doesn't exist.
        ValueError: If required columns are missing in the CSV.
    """
    # Get defaults from config
    _, default_raw_csv_path, default_processed_dir = _get_default_paths()
    
    if raw_csv_path is None:
        raw_csv_path = str(default_raw_csv_path)
    if processed_dir is None:
        processed_dir = str(default_processed_dir)
    
    raw_path = Path(raw_csv_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw CSV file not found: {raw_csv_path}")
    
    # Read the CSV
    df = pd.read_csv(raw_path)
    
    # Validate required columns
    required_columns = {"id_code", "diagnosis"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"CSV missing required columns: {missing}")
    
    # Rename columns to standard format
    df = df.rename(columns={"id_code": "filename", "diagnosis": "label"})
    
    # Ensure filename column contains strings (in case of numeric IDs)
    df["filename"] = df["filename"].astype(str)
    
    # Filter to only include processed images
    if processed_filenames is not None:
        original_count = len(df)
        df = df[df["filename"].isin(processed_filenames)].copy()
        filtered_count = original_count - len(df)
        if filtered_count > 0:
            print(f"Info: Removed {filtered_count} labels for non-processed images")
    
    # Create output directory if it doesn't exist
    output_dir = Path(processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to output directory
    output_path = output_dir / output_filename
    df.to_csv(output_path, index=False)
    
    return output_path


class Ddr2019PreprocessorConfig(BasePreprocessorConfig):
    """Configuration for DDR2019 preprocessor; uses same fields as base."""

    pass


@register_preprocessor("ddr2019")
class Ddr2019Preprocessor(BasePreprocessor):
    """Preprocessor for DDR2019 dataset: middleware pipeline + label CSV conversion."""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add common and DDR2019-specific CLI arguments."""
        config = get_preprocessing_config()
        parser.add_argument(
            "--raw-img-dir",
            type=str,
            default=str(config.ddr2019_raw_img_dir),
            help="Path to raw images directory",
        )
        parser.add_argument(
            "--raw-csv-path",
            type=str,
            default=str(config.ddr2019_raw_csv_path),
            help="Path to raw CSV file",
        )
        parser.add_argument(
            "--processed-dir",
            type=str,
            default=None,
            help="Full path to processed output directory (overrides --output-name)",
        )
        parser.add_argument(
            "--output-name",
            type=str,
            default=None,
            metavar="FOLDER",
            help="Output folder name under data/processed (e.g. 'ddr2019_512'). Ignored if --processed-dir is set.",
        )
        parser.add_argument(
            "--min-size",
            type=int,
            default=config.min_size,
            help=f"Minimum size (width and height) to process. Default: {config.min_size}",
        )
        parser.add_argument(
            "--target-size",
            type=int,
            nargs=2,
            metavar=("WIDTH", "HEIGHT"),
            default=list(config.target_size),
            help=f"Target size for resizing. Default: {config.target_size[0]} {config.target_size[1]}",
        )
        parser.add_argument(
            "--middleware",
            type=str,
            default=config.default_middleware,
            help=f"Middleware key for image processing. Default: {config.default_middleware}",
        )

    def run(self, config: BasePreprocessorConfig) -> dict[str, Any]:
        """Run DDR2019 preprocessing: core loop + convert_labels_csv."""
        context: MiddlewareContext = {
            "min_size": config.min_size,
            "target_size": config.target_size,
        }
        middleware = get_middleware(
            config.middleware,
            min_size=config.min_size,
            target_size=config.target_size,
            output_key=config.output_subdir,
        )
        images_processed, processed_filenames = run_core_loop(
            raw_img_dir=config.raw_img_dir,
            processed_dir=config.processed_dir,
            middleware=middleware,
            context=context,
        )
        labels_path = convert_labels_csv(
            raw_csv_path=config.raw_csv_path,
            processed_dir=config.processed_dir,
            processed_filenames=processed_filenames,
        )
        return {
            "images_processed": images_processed,
            "labels_path": labels_path,
            "processed_filenames": processed_filenames,
        }


def preprocess_ddr2019(
    raw_img_dir: Optional[str] = None,
    raw_csv_path: Optional[str] = None,
    processed_dir: Optional[str] = None,
    min_size: int | None = None,
    target_size: tuple[int, int] | None = None,
) -> dict:
    """Run complete preprocessing pipeline for DDR2019 dataset.

    Only processes images with both dimensions >= min_size. Smaller images are skipped.
    Non-square images are padded to make them square, then all images are resized to target_size.
    Labels for non-processed images are removed from the output CSV.

    Args:
        raw_img_dir: Path to raw images directory. Defaults to config value.
        raw_csv_path: Path to raw CSV file. Defaults to config value.
        processed_dir: Output directory for processed data. Defaults to config value.
        min_size: Minimum size (width and height) required to process an image. Defaults to config value.
        target_size: Target size for image resizing. Defaults to config value.

    Returns:
        Dictionary with processing results:
        - images_processed: Number of images processed (size >= min_size)
        - labels_path: Path to created labels.csv (only for processed images)
        - processed_filenames: Set of filenames that were processed
    """
    cfg = get_preprocessing_config()
    if raw_img_dir is None:
        raw_img_dir = str(cfg.ddr2019_raw_img_dir)
    if raw_csv_path is None:
        raw_csv_path = str(cfg.ddr2019_raw_csv_path)
    if processed_dir is None:
        processed_dir = str(cfg.ddr2019_processed_dir)
    if min_size is None:
        min_size = cfg.min_size
    if target_size is None:
        target_size = tuple(cfg.target_size)

    preprocessor_config = Ddr2019PreprocessorConfig(
        raw_img_dir=raw_img_dir,
        raw_csv_path=raw_csv_path,
        processed_dir=processed_dir,
        min_size=min_size,
        target_size=target_size,
        middleware=cfg.default_middleware,
        output_subdir=cfg.default_output_subdir,
    )
    return Ddr2019Preprocessor().run(preprocessor_config)


if __name__ == "__main__":
    print("Starting DDR2019 preprocessing...")
    min_size, target_size = _get_default_sizes()
    raw_img_dir, raw_csv_path, processed_dir = _get_default_paths()
    
    print(f"  - Minimum size: {min_size}x{min_size}")
    print(f"  - Target size: {target_size[0]}x{target_size[1]}")
    
    # Count original images before processing
    raw_img_dir = str(raw_img_dir)
    raw_path = Path(raw_img_dir)
    if raw_path.exists():
        original_image_count = len(list(raw_path.glob("*.jpg")))
    else:
        original_image_count = 0
    
    results = preprocess_ddr2019()
    
    print(f"\nDDR2019 preprocessing complete.")
    print(f"  - Original dataset: {original_image_count} images")
    print(f"  - Processed dataset: {results['images_processed']} images")
    print(f"  - Images skipped: {original_image_count - results['images_processed']} images (too small or would require upscaling)")
    print(f"  - Labels saved to: {results['labels_path']}")
