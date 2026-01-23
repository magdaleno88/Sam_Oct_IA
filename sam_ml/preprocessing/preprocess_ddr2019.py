"""Preprocessing script for DDR2019 dataset.

This script processes the raw DDR2019 dataset by:
1. Filtering images by minimum size (>= 512x512)
2. Adding padding to non-square images to make them square
3. Resizing all images to 512x512
4. Converting the label CSV to the standard format (filename, label)
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image
from tqdm import tqdm

from sam_ml.config import get_preprocessing_config


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
    # Get defaults from config
    default_min_size, default_target_size = _get_default_sizes()
    default_raw_img_dir, _, default_processed_dir = _get_default_paths()
    
    if min_size is None:
        min_size = default_min_size
    if target_size is None:
        target_size = default_target_size
    if raw_img_dir is None:
        raw_img_dir = str(default_raw_img_dir)
    if resized_img_dir is None:
        resized_img_dir = str(default_processed_dir / "images")
    
    raw_path = Path(raw_img_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw images directory not found: {raw_img_dir}")
    
    # Create output directory
    output_path = Path(resized_img_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all JPG files
    image_files = list(raw_path.glob("*.jpg"))
    
    if not image_files:
        raise ValueError(f"No JPG files found in {raw_img_dir}")
    
    processed_count = 0
    processed_filenames = set()
    skipped_small_count = 0
    
    # Process each image with progress bar
    for img_file in tqdm(image_files, desc="Processing images"):
        try:
            # Open and convert to RGB (handles RGBA, grayscale, etc.)
            img = Image.open(img_file).convert("RGB")
            
            # Check minimum size requirement - only process images >= min_size
            # This ensures we never upscale (which would add noise)
            # We only downscale or keep same size, never upscale
            width, height = img.size
            if width < min_size or height < min_size:
                # Skip images smaller than minimum size to avoid upscaling noise
                skipped_small_count += 1
                continue
            
            # Add padding if image is not square
            # After padding, the image will be square with size = max(width, height)
            if width != height:
                img = add_padding_to_square(img)
            
            # Verify that after padding, we won't need to upscale
            # The image (after padding) must be >= target_size to avoid upscaling
            current_width, current_height = img.size
            if current_width < target_size[0] or current_height < target_size[1]:
                # Skip this image - it would require upscaling which adds noise
                skipped_small_count += 1
                continue
            
            # Resize to target size
            # At this point, we're guaranteed to be downscaling or keeping same size (never upscaling)
            img_processed = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Save to output directory
            output_file = output_path / img_file.name
            img_processed.save(output_file, "JPEG", quality=95)
            
            processed_count += 1
            processed_filenames.add(img_file.name)
        except Exception as e:
            print(f"Warning: Failed to process {img_file.name}: {e}")
            continue
    
    if skipped_small_count > 0:
        print(f"Info: Skipped {skipped_small_count} images smaller than {min_size}x{min_size}")
    
    return processed_count, processed_filenames


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
    # Get defaults from config
    default_min_size, default_target_size = _get_default_sizes()
    default_raw_img_dir, default_raw_csv_path, default_processed_dir = _get_default_paths()
    
    if min_size is None:
        min_size = default_min_size
    if target_size is None:
        target_size = default_target_size
    if raw_img_dir is None:
        raw_img_dir = str(default_raw_img_dir)
    if raw_csv_path is None:
        raw_csv_path = str(default_raw_csv_path)
    if processed_dir is None:
        processed_dir = str(default_processed_dir)
    
    resized_img_dir = os.path.join(processed_dir, "images")
    
    # Process images (filter by min_size, pad non-square, resize to target_size)
    images_processed, processed_filenames = resize_and_copy_images(
        raw_img_dir=raw_img_dir,
        resized_img_dir=resized_img_dir,
        min_size=min_size,
        target_size=target_size,
    )
    
    # Process labels (only for processed images)
    labels_path = convert_labels_csv(
        raw_csv_path=raw_csv_path,
        processed_dir=processed_dir,
        processed_filenames=processed_filenames,
    )
    
    return {
        "images_processed": images_processed,
        "labels_path": labels_path,
        "processed_filenames": processed_filenames,
    }


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
