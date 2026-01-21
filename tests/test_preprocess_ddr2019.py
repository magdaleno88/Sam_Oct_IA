"""Unit tests for DDR2019 preprocessing module."""

import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from sam_ml.preprocessing.preprocess_ddr2019 import (
    convert_labels_csv,
    preprocess_ddr2019,
    resize_and_copy_images,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def sample_images_dir(temp_dir):
    """Create a directory with sample test images (all square)."""
    img_dir = Path(temp_dir) / "raw_images"
    img_dir.mkdir(parents=True)
    
    # Create 5 sample images with different square sizes
    for i in range(5):
        # Create square images
        sizes = [(800, 800), (1024, 1024), (640, 640), (512, 512), (400, 400)]
        img = Image.new("RGB", sizes[i], color=(i * 50, i * 50, i * 50))
        img.save(img_dir / f"test_image_{i:03d}.jpg", "JPEG")
    
    return str(img_dir)


@pytest.fixture
def mixed_images_dir(temp_dir):
    """Create a directory with mixed square and non-square images."""
    img_dir = Path(temp_dir) / "mixed_images"
    img_dir.mkdir(parents=True)
    
    # Create 3 square images
    for i in range(3):
        img = Image.new("RGB", (512, 512), color=(i * 50, i * 50, i * 50))
        img.save(img_dir / f"square_{i:03d}.jpg", "JPEG")
    
    # Create 2 non-square (asymmetric) images
    for i in range(2):
        img = Image.new("RGB", (800, 600), color=(i * 50, i * 50, i * 50))
        img.save(img_dir / f"asymmetric_{i:03d}.jpg", "JPEG")
    
    return str(img_dir)


@pytest.fixture
def sample_csv_file(temp_dir):
    """Create a sample CSV file with labels."""
    csv_path = Path(temp_dir) / "DR_grading.csv"
    
    # Create CSV matching the sample images
    data = {
        "id_code": [f"test_image_{i:03d}.jpg" for i in range(5)],
        "diagnosis": [0, 1, 2, 0, 1],
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    
    return str(csv_path)


@pytest.fixture
def sample_csv_file_mismatch(temp_dir):
    """Create a CSV file with mismatched filenames."""
    csv_path = Path(temp_dir) / "DR_grading_mismatch.csv"
    
    data = {
        "id_code": [f"nonexistent_{i:03d}.jpg" for i in range(3)],
        "diagnosis": [0, 1, 2],
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    
    return str(csv_path)


class TestResizeAndCopyImages:
    """Tests for resize_and_copy_images function."""
    
    def test_resize_and_copy_images_basic(self, temp_dir, sample_images_dir):
        """Test basic image resizing and copying."""
        output_dir = Path(temp_dir) / "processed" / "images"
        
        count, filenames = resize_and_copy_images(
            raw_img_dir=sample_images_dir,
            resized_img_dir=str(output_dir),
            resize_shape=(512, 512),
        )
        
        # Check that correct number of images were processed
        assert count == 5
        assert len(filenames) == 5
        
        # Check that output directory exists
        assert output_dir.exists()
        
        # Check that all images were created
        output_files = list(output_dir.glob("*.jpg"))
        assert len(output_files) == 5
        
        # Check that images are correctly resized
        for img_file in output_files:
            img = Image.open(img_file)
            assert img.size == (512, 512)
            assert img.mode == "RGB"
            assert img_file.name in filenames
    
    def test_resize_and_copy_images_keeps_original_size(self, temp_dir, sample_images_dir):
        """Test that images keep their original size when resize_shape is None."""
        output_dir = Path(temp_dir) / "processed" / "images"
        
        # Get original image sizes
        raw_path = Path(sample_images_dir)
        original_sizes = {}
        for img_file in raw_path.glob("*.jpg"):
            img = Image.open(img_file)
            original_sizes[img_file.name] = img.size
        
        count, filenames = resize_and_copy_images(
            raw_img_dir=sample_images_dir,
            resized_img_dir=str(output_dir),
            resize_shape=None,  # Keep original size
        )
        
        # Check that correct number of images were processed
        assert count == 5
        assert len(filenames) == 5
        
        # Check that images keep their original sizes
        for img_file in output_dir.glob("*.jpg"):
            img = Image.open(img_file)
            original_size = original_sizes[img_file.name]
            assert img.size == original_size, f"Image {img_file.name} size changed"
            assert img.mode == "RGB"
    
    def test_resize_and_copy_images_custom_size(self, temp_dir, sample_images_dir):
        """Test resizing to custom dimensions."""
        output_dir = Path(temp_dir) / "processed" / "images"
        
        count, filenames = resize_and_copy_images(
            raw_img_dir=sample_images_dir,
            resized_img_dir=str(output_dir),
            resize_shape=(256, 256),
        )
        
        assert count == 5
        assert len(filenames) == 5
        
        # Verify all images are 256x256
        for img_file in output_dir.glob("*.jpg"):
            img = Image.open(img_file)
            assert img.size == (256, 256)
    
    def test_resize_and_copy_images_nonexistent_dir(self, temp_dir):
        """Test error handling for nonexistent input directory."""
        output_dir = Path(temp_dir) / "processed" / "images"
        
        with pytest.raises(FileNotFoundError):
            resize_and_copy_images(
                raw_img_dir=str(Path(temp_dir) / "nonexistent"),
                resized_img_dir=str(output_dir),
            )
    
    def test_resize_and_copy_images_empty_dir(self, temp_dir):
        """Test error handling for empty directory."""
        empty_dir = Path(temp_dir) / "empty"
        empty_dir.mkdir()
        output_dir = Path(temp_dir) / "processed" / "images"
        
        with pytest.raises(ValueError, match="No JPG files found"):
            resize_and_copy_images(
                raw_img_dir=str(empty_dir),
                resized_img_dir=str(output_dir),
            )
    
    def test_resize_and_copy_images_creates_output_dir(self, temp_dir, sample_images_dir):
        """Test that output directory is created if it doesn't exist."""
        output_dir = Path(temp_dir) / "nested" / "deep" / "output" / "images"
        
        assert not output_dir.exists()
        
        count, filenames = resize_and_copy_images(
            raw_img_dir=sample_images_dir,
            resized_img_dir=str(output_dir),
        )
        
        assert output_dir.exists()
        assert len(list(output_dir.glob("*.jpg"))) == 5
        assert count == 5
        assert len(filenames) == 5
    
    def test_resize_and_copy_images_filters_asymmetric(self, temp_dir, mixed_images_dir):
        """Test that asymmetric (non-square) images are filtered out."""
        output_dir = Path(temp_dir) / "processed" / "images"
        
        count, filenames = resize_and_copy_images(
            raw_img_dir=mixed_images_dir,
            resized_img_dir=str(output_dir),
            resize_shape=(512, 512),
        )
        
        # Should only process 3 square images, skip 2 asymmetric ones
        assert count == 3
        assert len(filenames) == 3
        
        # Check that only square images were processed
        output_files = list(output_dir.glob("*.jpg"))
        assert len(output_files) == 3
        
        # Verify all processed images are square and correctly sized
        for img_file in output_files:
            img = Image.open(img_file)
            assert img.size == (512, 512)
            # Should only have square_*.jpg files
            assert img_file.name.startswith("square_")
        
        # Verify asymmetric images are not in the processed set
        assert "asymmetric_000.jpg" not in filenames
        assert "asymmetric_001.jpg" not in filenames


class TestConvertLabelsCsv:
    """Tests for convert_labels_csv function."""
    
    def test_convert_labels_csv_basic(self, temp_dir, sample_csv_file):
        """Test basic CSV conversion."""
        output_dir = Path(temp_dir) / "processed"
        
        output_path = convert_labels_csv(
            raw_csv_path=sample_csv_file,
            processed_dir=str(output_dir),
        )
        
        # Check that file was created
        assert output_path.exists()
        assert output_path.name == "labels.csv"
        
        # Read and validate the converted CSV
        df = pd.read_csv(output_path)
        
        # Check column names
        assert list(df.columns) == ["filename", "label"]
        
        # Check data integrity
        assert len(df) == 5
        assert df["filename"].dtype == object  # String type
        assert df["label"].dtype in [int, "int64"]
        
        # Check specific values
        assert df.iloc[0]["filename"] == "test_image_000.jpg"
        assert df.iloc[0]["label"] == 0
    
    def test_convert_labels_csv_filters_unprocessed(self, temp_dir, sample_csv_file):
        """Test that CSV only includes labels for processed images."""
        output_dir = Path(temp_dir) / "processed"
        
        # Simulate that only 3 out of 5 images were processed
        processed_filenames = {
            "test_image_000.jpg",
            "test_image_001.jpg",
            "test_image_002.jpg",
        }
        
        output_path = convert_labels_csv(
            raw_csv_path=sample_csv_file,
            processed_dir=str(output_dir),
            processed_filenames=processed_filenames,
        )
        
        # Read and validate the filtered CSV
        df = pd.read_csv(output_path)
        
        # Should only have 3 rows (for processed images)
        assert len(df) == 3
        
        # Check that only processed filenames are present
        csv_filenames = set(df["filename"])
        assert csv_filenames == processed_filenames
        
        # Verify excluded filenames are not present
        assert "test_image_003.jpg" not in csv_filenames
        assert "test_image_004.jpg" not in csv_filenames
    
    def test_convert_labels_csv_custom_output_name(self, temp_dir, sample_csv_file):
        """Test CSV conversion with custom output filename."""
        output_dir = Path(temp_dir) / "processed"
        
        output_path = convert_labels_csv(
            raw_csv_path=sample_csv_file,
            processed_dir=str(output_dir),
            output_filename="custom_labels.csv",
        )
        
        assert output_path.name == "custom_labels.csv"
        assert output_path.exists()
    
    def test_convert_labels_csv_creates_output_dir(self, temp_dir, sample_csv_file):
        """Test that output directory is created if it doesn't exist."""
        output_dir = Path(temp_dir) / "nested" / "deep" / "output"
        
        assert not output_dir.exists()
        
        convert_labels_csv(
            raw_csv_path=sample_csv_file,
            processed_dir=str(output_dir),
        )
        
        assert output_dir.exists()
        assert (output_dir / "labels.csv").exists()
    
    def test_convert_labels_csv_nonexistent_file(self, temp_dir):
        """Test error handling for nonexistent CSV file."""
        output_dir = Path(temp_dir) / "processed"
        
        with pytest.raises(FileNotFoundError):
            convert_labels_csv(
                raw_csv_path=str(Path(temp_dir) / "nonexistent.csv"),
                processed_dir=str(output_dir),
            )
    
    def test_convert_labels_csv_missing_columns(self, temp_dir):
        """Test error handling for CSV with missing columns."""
        csv_path = Path(temp_dir) / "bad.csv"
        
        # Create CSV with wrong column names
        df = pd.DataFrame({"wrong_col": [1, 2, 3], "another_col": [4, 5, 6]})
        df.to_csv(csv_path, index=False)
        
        output_dir = Path(temp_dir) / "processed"
        
        with pytest.raises(ValueError, match="missing required columns"):
            convert_labels_csv(
                raw_csv_path=str(csv_path),
                processed_dir=str(output_dir),
            )


class TestPreprocessDdr2019:
    """Tests for the complete preprocessing pipeline."""
    
    def test_preprocess_ddr2019_complete(self, temp_dir, sample_images_dir, sample_csv_file):
        """Test the complete preprocessing pipeline with resizing."""
        processed_dir = Path(temp_dir) / "processed"
        
        results = preprocess_ddr2019(
            raw_img_dir=sample_images_dir,
            raw_csv_path=sample_csv_file,
            processed_dir=str(processed_dir),
            resize_shape=(512, 512),
        )
        
        # Check return values
        assert results["images_processed"] == 5
        assert results["labels_path"].exists()
        assert len(results["processed_filenames"]) == 5
        
        # Check directory structure
        images_dir = processed_dir / "images"
        assert images_dir.exists()
        assert (processed_dir / "labels.csv").exists()
        
        # Check images
        image_files = list(images_dir.glob("*.jpg"))
        assert len(image_files) == 5
        
        # Check labels CSV
        labels_df = pd.read_csv(results["labels_path"])
        assert len(labels_df) == 5
        assert list(labels_df.columns) == ["filename", "label"]
        
        # Verify that all filenames in CSV match processed images
        csv_filenames = set(labels_df["filename"])
        image_filenames = {f.name for f in image_files}
        assert csv_filenames == image_filenames
        assert csv_filenames == results["processed_filenames"]
    
    def test_preprocess_ddr2019_keeps_original_size(self, temp_dir, sample_images_dir, sample_csv_file):
        """Test preprocessing pipeline keeping original image sizes."""
        processed_dir = Path(temp_dir) / "processed"
        
        # Get original image sizes
        raw_path = Path(sample_images_dir)
        original_sizes = {}
        for img_file in raw_path.glob("*.jpg"):
            img = Image.open(img_file)
            original_sizes[img_file.name] = img.size
        
        results = preprocess_ddr2019(
            raw_img_dir=sample_images_dir,
            raw_csv_path=sample_csv_file,
            processed_dir=str(processed_dir),
            resize_shape=None,  # Keep original size
        )
        
        # Check return values
        assert results["images_processed"] == 5
        assert results["labels_path"].exists()
        assert len(results["processed_filenames"]) == 5
        
        # Check that images keep their original sizes
        images_dir = processed_dir / "images"
        for img_file in images_dir.glob("*.jpg"):
            img = Image.open(img_file)
            original_size = original_sizes[img_file.name]
            assert img.size == original_size, f"Image {img_file.name} size changed"
    
    def test_preprocess_ddr2019_filters_asymmetric(self, temp_dir, mixed_images_dir):
        """Test that preprocessing filters out asymmetric images."""
        # Create CSV for mixed images
        csv_path = Path(temp_dir) / "DR_grading.csv"
        data = {
            "id_code": [
                "square_000.jpg",
                "square_001.jpg",
                "square_002.jpg",
                "asymmetric_000.jpg",
                "asymmetric_001.jpg",
            ],
            "diagnosis": [0, 1, 2, 0, 1],
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        
        processed_dir = Path(temp_dir) / "processed"
        
        results = preprocess_ddr2019(
            raw_img_dir=mixed_images_dir,
            raw_csv_path=str(csv_path),
            processed_dir=str(processed_dir),
            resize_shape=(512, 512),
        )
        
        # Should only process 3 square images
        assert results["images_processed"] == 3
        assert len(results["processed_filenames"]) == 3
        
        # Check that only square images were processed
        images_dir = processed_dir / "images"
        image_files = list(images_dir.glob("*.jpg"))
        assert len(image_files) == 3
        
        # Check labels CSV - should only have 3 rows
        labels_df = pd.read_csv(results["labels_path"])
        assert len(labels_df) == 3
        
        # Verify CSV only contains square image labels
        csv_filenames = set(labels_df["filename"])
        assert csv_filenames == {"square_000.jpg", "square_001.jpg", "square_002.jpg"}
        assert "asymmetric_000.jpg" not in csv_filenames
        assert "asymmetric_001.jpg" not in csv_filenames
    
    def test_preprocess_ddr2019_default_paths(self, temp_dir, sample_images_dir, sample_csv_file):
        """Test preprocessing with default paths and default behavior (no resizing)."""
        # This test would require mocking the module constants or setting up
        # the actual directory structure. For now, we'll test with explicit paths.
        # In a real scenario, you might use monkeypatch to set the constants.
        processed_dir = Path(temp_dir) / "processed"
        
        # Get original image sizes
        raw_path = Path(sample_images_dir)
        original_sizes = {}
        for img_file in raw_path.glob("*.jpg"):
            img = Image.open(img_file)
            original_sizes[img_file.name] = img.size
        
        results = preprocess_ddr2019(
            raw_img_dir=sample_images_dir,
            raw_csv_path=sample_csv_file,
            processed_dir=str(processed_dir),
            # resize_shape=None by default - should keep original sizes
        )
        
        assert results["images_processed"] == 5
        assert results["labels_path"].exists()
        assert len(results["processed_filenames"]) == 5
        
        # Verify images keep original sizes (default behavior)
        images_dir = processed_dir / "images"
        for img_file in images_dir.glob("*.jpg"):
            img = Image.open(img_file)
            original_size = original_sizes[img_file.name]
            assert img.size == original_size, f"Image {img_file.name} size changed (should keep original)"


class TestIntegration:
    """Integration tests validating the full preprocessing workflow."""
    
    def test_labels_match_images(self, temp_dir, sample_images_dir, sample_csv_file):
        """Test that all labels in CSV correspond to processed images."""
        processed_dir = Path(temp_dir) / "processed"
        
        preprocess_ddr2019(
            raw_img_dir=sample_images_dir,
            raw_csv_path=sample_csv_file,
            processed_dir=str(processed_dir),
        )
        
        # Load labels
        labels_df = pd.read_csv(processed_dir / "labels.csv")
        
        # Get processed images
        images_dir = processed_dir / "images"
        image_files = {f.name for f in images_dir.glob("*.jpg")}
        
        # Every filename in CSV should have a corresponding image
        csv_filenames = set(labels_df["filename"])
        assert csv_filenames.issubset(image_files)
        
        # Every image should have a corresponding label (if CSV is complete)
        # Note: This assumes CSV contains all images, which may not always be true
        # In practice, you might have images without labels or vice versa
    
    def test_image_shapes_consistent(self, temp_dir, sample_images_dir, sample_csv_file):
        """Test that all processed images have consistent shapes when resized."""
        processed_dir = Path(temp_dir) / "processed"
        
        preprocess_ddr2019(
            raw_img_dir=sample_images_dir,
            raw_csv_path=sample_csv_file,
            processed_dir=str(processed_dir),
            resize_shape=(512, 512),
        )
        
        images_dir = processed_dir / "images"
        image_files = list(images_dir.glob("*.jpg"))
        
        # All images should have the same size when resized
        sizes = {Image.open(f).size for f in image_files}
        assert len(sizes) == 1
        assert (512, 512) in sizes
    
    def test_file_counts_match(self, temp_dir, sample_images_dir, sample_csv_file):
        """Test that the number of processed images matches expectations."""
        processed_dir = Path(temp_dir) / "processed"
        
        results = preprocess_ddr2019(
            raw_img_dir=sample_images_dir,
            raw_csv_path=sample_csv_file,
            processed_dir=str(processed_dir),
        )
        
        # Count files
        images_dir = processed_dir / "images"
        image_count = len(list(images_dir.glob("*.jpg")))
        labels_df = pd.read_csv(processed_dir / "labels.csv")
        label_count = len(labels_df)
        
        # All counts should match
        assert results["images_processed"] == image_count
        assert image_count == label_count == 5
    
    def test_original_files_not_modified(self, temp_dir, sample_images_dir, sample_csv_file):
        """Test that original dataset files are not modified during preprocessing."""
        # Store original file states
        raw_img_path = Path(sample_images_dir)
        raw_csv_path = Path(sample_csv_file)
        
        # Get original image file sizes and modification times
        original_image_states = {}
        for img_file in raw_img_path.glob("*.jpg"):
            original_image_states[img_file.name] = {
                "size": img_file.stat().st_size,
                "mtime": img_file.stat().st_mtime,
            }
        
        # Get original CSV state
        original_csv_state = {
            "size": raw_csv_path.stat().st_size,
            "mtime": raw_csv_path.stat().st_mtime,
            "content": raw_csv_path.read_text(),
        }
        
        # Run preprocessing
        processed_dir = Path(temp_dir) / "processed"
        preprocess_ddr2019(
            raw_img_dir=sample_images_dir,
            raw_csv_path=sample_csv_file,
            processed_dir=str(processed_dir),
        )
        
        # Verify original images are unchanged
        for img_file in raw_img_path.glob("*.jpg"):
            current_state = {
                "size": img_file.stat().st_size,
                "mtime": img_file.stat().st_mtime,
            }
            original_state = original_image_states[img_file.name]
            assert current_state["size"] == original_state["size"], \
                f"Original image {img_file.name} size was modified"
            assert current_state["mtime"] == original_state["mtime"], \
                f"Original image {img_file.name} modification time changed"
        
        # Verify original CSV is unchanged
        current_csv_state = {
            "size": raw_csv_path.stat().st_size,
            "mtime": raw_csv_path.stat().st_mtime,
            "content": raw_csv_path.read_text(),
        }
        assert current_csv_state["size"] == original_csv_state["size"], \
            "Original CSV file size was modified"
        assert current_csv_state["mtime"] == original_csv_state["mtime"], \
            "Original CSV file modification time changed"
        assert current_csv_state["content"] == original_csv_state["content"], \
            "Original CSV file content was modified"
        
        # Verify processed files exist in separate location
        processed_images_dir = processed_dir / "images"
        assert processed_images_dir.exists(), "Processed images directory should exist"
        assert processed_images_dir != raw_img_path, "Processed directory should be different from raw"
        
        processed_csv = processed_dir / "labels.csv"
        assert processed_csv.exists(), "Processed CSV should exist"
        assert processed_csv != raw_csv_path, "Processed CSV should be different from raw CSV"
