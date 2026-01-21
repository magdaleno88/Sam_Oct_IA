"""Preprocessing script for DDR2019 dataset.

This script processes the raw DDR2019 dataset by:
1. Resizing and copying images from raw to processed directory
2. Converting the label CSV to the standard format (filename, label)
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image
from tqdm import tqdm


# Default paths (can be overridden)
RAW_IMG_DIR = "data/raw/ddr2019/DR_grading/DR_grading"
RAW_CSV_PATH = "data/raw/ddr2019/DR_grading.csv"
PROCESSED_DIR = "data/processed/ddr2019"
RESIZED_IMG_DIR = os.path.join(PROCESSED_DIR, "images")


def resize_and_copy_images(
    raw_img_dir: Optional[str] = None,
    resized_img_dir: Optional[str] = None,
    resize_shape: Optional[tuple[int, int]] = None,
) -> tuple[int, set[str]]:
    """Resize and copy images from raw directory to processed directory.
    
    Only processes square images (width == height). Asymmetric images are skipped.
    By default, images keep their original size unless resize_shape is specified.
    
    Args:
        raw_img_dir: Path to raw images directory. Defaults to RAW_IMG_DIR.
        resized_img_dir: Path to output directory for resized images. Defaults to RESIZED_IMG_DIR.
        resize_shape: Target size (width, height) for resizing. If None, keeps original size.
    
    Returns:
        Tuple of (number of images processed, set of processed filenames).
    
    Raises:
        FileNotFoundError: If raw_img_dir doesn't exist.
        OSError: If unable to create output directory or write images.
    """
    if raw_img_dir is None:
        raw_img_dir = RAW_IMG_DIR
    if resized_img_dir is None:
        resized_img_dir = RESIZED_IMG_DIR
    
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
    skipped_count = 0
    
    # Process each image with progress bar
    desc = "Resizing images" if resize_shape else "Copying images"
    for img_file in tqdm(image_files, desc=desc):
        try:
            # Open and convert to RGB (handles RGBA, grayscale, etc.)
            img = Image.open(img_file).convert("RGB")
            
            # Check if image is square (width == height)
            width, height = img.size
            if width != height:
                # Skip asymmetric images
                skipped_count += 1
                continue
            
            # Resize image if resize_shape is provided, otherwise keep original size
            if resize_shape is not None:
                img_processed = img.resize(resize_shape, Image.Resampling.LANCZOS)
            else:
                img_processed = img
            
            # Save to output directory
            output_file = output_path / img_file.name
            img_processed.save(output_file, "JPEG", quality=95)
            
            processed_count += 1
            processed_filenames.add(img_file.name)
        except Exception as e:
            print(f"Warning: Failed to process {img_file.name}: {e}")
            continue
    
    if skipped_count > 0:
        print(f"Info: Skipped {skipped_count} asymmetric (non-square) images")
    
    return processed_count, processed_filenames


def convert_labels_csv(
    raw_csv_path: Optional[str] = None,
    processed_dir: Optional[str] = None,
    output_filename: str = "labels.csv",
    processed_filenames: Optional[set[str]] = None,
) -> Path:
    """Convert DDR2019 label CSV to standard format.
    
    Converts the CSV from format (id_code, diagnosis) to (filename, label).
    Only includes labels for images that were actually processed (square images).
    
    Args:
        raw_csv_path: Path to input CSV file. Defaults to RAW_CSV_PATH.
        processed_dir: Directory to save output CSV. Defaults to PROCESSED_DIR.
        output_filename: Name of output CSV file. Defaults to "labels.csv".
        processed_filenames: Set of filenames that were successfully processed.
                            If None, includes all rows from CSV.
    
    Returns:
        Path to the created labels.csv file.
    
    Raises:
        FileNotFoundError: If raw_csv_path doesn't exist.
        ValueError: If required columns are missing in the CSV.
    """
    if raw_csv_path is None:
        raw_csv_path = RAW_CSV_PATH
    if processed_dir is None:
        processed_dir = PROCESSED_DIR
    
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
    
    # Filter to only include processed images (square images)
    if processed_filenames is not None:
        original_count = len(df)
        df = df[df["filename"].isin(processed_filenames)].copy()
        filtered_count = original_count - len(df)
        if filtered_count > 0:
            print(f"Info: Removed {filtered_count} labels for asymmetric/non-processed images")
    
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
    resize_shape: Optional[tuple[int, int]] = None,
) -> dict:
    """Run complete preprocessing pipeline for DDR2019 dataset.
    
    Only processes square images (width == height). Asymmetric images are skipped
    and their labels are removed from the output CSV.
    By default, images keep their original size unless resize_shape is specified.
    
    Args:
        raw_img_dir: Path to raw images directory. Defaults to RAW_IMG_DIR.
        raw_csv_path: Path to raw CSV file. Defaults to RAW_CSV_PATH.
        processed_dir: Output directory for processed data. Defaults to PROCESSED_DIR.
        resize_shape: Target size for image resizing. If None, keeps original size.
    
    Returns:
        Dictionary with processing results:
        - images_processed: Number of images processed (square images only)
        - labels_path: Path to created labels.csv (only for processed images)
        - processed_filenames: Set of filenames that were processed
    """
    if processed_dir is None:
        processed_dir = PROCESSED_DIR
    
    resized_img_dir = os.path.join(processed_dir, "images")
    
    # Process images (only square images)
    images_processed, processed_filenames = resize_and_copy_images(
        raw_img_dir=raw_img_dir,
        resized_img_dir=resized_img_dir,
        resize_shape=resize_shape,
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
    results = preprocess_ddr2019()
    print(f"DDR2019 preprocessing complete.")
    print(f"  - Images processed: {results['images_processed']}")
    print(f"  - Labels saved to: {results['labels_path']}")
