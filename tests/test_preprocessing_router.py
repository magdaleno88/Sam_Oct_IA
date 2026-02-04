"""Unit tests for preprocessing router script."""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from PIL import Image

from sam_ml.preprocessing import main
from sam_ml.preprocessing.base import get_preprocessor, list_preprocessors
from sam_ml.preprocessing.middleware import list_middlewares


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def sample_images_dir(temp_dir):
    """Create a directory with sample test images."""
    img_dir = Path(temp_dir) / "raw_images"
    img_dir.mkdir(parents=True)
    
    # Create 3 sample images
    for i in range(3):
        img = Image.new("RGB", (800, 600), color=(i * 50, i * 50, i * 50))
        img.save(img_dir / f"test_image_{i:03d}.jpg", "JPEG")
    
    return str(img_dir)


@pytest.fixture
def sample_csv_file(temp_dir):
    """Create a sample CSV file with labels."""
    csv_path = Path(temp_dir) / "DR_grading.csv"
    
    data = {
        "id_code": [f"test_image_{i:03d}.jpg" for i in range(3)],
        "diagnosis": [0, 1, 0],
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    
    return str(csv_path)


class TestPreprocessingRouter:
    """Tests for the preprocessing router main function."""
    
    def test_main_ddr2019_basic(self, temp_dir, sample_images_dir, sample_csv_file):
        """Test basic routing to DDR2019 preprocessing."""
        processed_dir = Path(temp_dir) / "processed"
        
        exit_code = main([
            "ddr2019",
            "--raw-img-dir", sample_images_dir,
            "--raw-csv-path", sample_csv_file,
            "--processed-dir", str(processed_dir),
        ])
        
        assert exit_code == 0
        
        # Verify output was created
        assert (processed_dir / "images").exists()
        assert (processed_dir / "labels.csv").exists()
    
    def test_main_ddr2019_with_target_size(self, temp_dir, sample_images_dir, sample_csv_file):
        """Test DDR2019 preprocessing with custom target size."""
        processed_dir = Path(temp_dir) / "processed"
        
        exit_code = main([
            "ddr2019",
            "--raw-img-dir", sample_images_dir,
            "--raw-csv-path", sample_csv_file,
            "--processed-dir", str(processed_dir),
            "--target-size", "256", "256",
        ])
        
        assert exit_code == 0
        
        # Verify images are resized correctly
        images_dir = processed_dir / "images"
        for img_file in images_dir.glob("*.jpg"):
            img = Image.open(img_file)
            assert img.size == (256, 256)
    
    def test_main_ddr2019_with_min_size(self, temp_dir, sample_images_dir, sample_csv_file):
        """Test DDR2019 preprocessing with custom minimum size."""
        processed_dir = Path(temp_dir) / "processed"
        
        exit_code = main([
            "ddr2019",
            "--raw-img-dir", sample_images_dir,
            "--raw-csv-path", sample_csv_file,
            "--processed-dir", str(processed_dir),
            "--min-size", "600",
            "--target-size", "512", "512",
        ])
        
        assert exit_code == 0
        
        # Verify images are processed (800x600 >= 600x600)
        images_dir = processed_dir / "images"
        image_files = list(images_dir.glob("*.jpg"))
        assert len(image_files) == 3  # All 3 images are >= 600
        
        # Verify all are resized to 512x512
        for img_file in image_files:
            img = Image.open(img_file)
            assert img.size == (512, 512)
    
    def test_main_ddr2019_default_paths(self, temp_dir):
        """Test DDR2019 preprocessing with default paths (requires actual data)."""
        # Mock the preprocessor's run to avoid requiring actual data
        with patch(
            "sam_ml.preprocessing.preprocess_ddr2019.Ddr2019Preprocessor.run"
        ) as mock_run:
            mock_run.side_effect = FileNotFoundError("Raw images directory not found")

            exit_code = main(["ddr2019"])

            assert exit_code == 1
    
    def test_main_invalid_dataset(self, capsys):
        """Test that invalid dataset name is rejected."""
        exit_code = main(["invalid_dataset"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err.lower() or "invalid" in captured.err.lower()
    
    def test_main_missing_dataset_argument(self, capsys):
        """Test that missing dataset argument returns error."""
        exit_code = main([])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "missing" in captured.err.lower() or "dataset" in captured.err.lower()
    
    def test_main_error_handling(self, temp_dir):
        """Test error handling when preprocessing fails."""
        processed_dir = Path(temp_dir) / "processed"
        
        # Use non-existent directory to trigger error
        exit_code = main([
            "ddr2019",
            "--raw-img-dir", str(Path(temp_dir) / "nonexistent"),
            "--raw-csv-path", str(Path(temp_dir) / "nonexistent.csv"),
            "--processed-dir", str(processed_dir),
        ])
        
        assert exit_code == 1
    
    def test_main_help_message(self, capsys):
        """Test that help message is displayed correctly."""
        with pytest.raises(SystemExit):
            main(["--help"])
        
        captured = capsys.readouterr()
        assert "Preprocess datasets for SAM-AI project" in captured.out
        assert "ddr2019" in captured.out
        assert "--raw-img-dir" in captured.out
        assert "--min-size" in captured.out
        assert "--target-size" in captured.out
    
    def test_main_example_usage(self, temp_dir, sample_images_dir, sample_csv_file):
        """Test example usage from help message."""
        processed_dir = Path(temp_dir) / "processed"
        
        # Test example: preprocess-dataset ddr2019
        exit_code = main([
            "ddr2019",
            "--raw-img-dir", sample_images_dir,
            "--raw-csv-path", sample_csv_file,
            "--processed-dir", str(processed_dir),
        ])
        
        assert exit_code == 0
        assert (processed_dir / "images").exists()
        assert (processed_dir / "labels.csv").exists()
        
        # Test example: preprocess-dataset ddr2019 --target-size 256 256
        processed_dir2 = Path(temp_dir) / "processed2"
        exit_code = main([
            "ddr2019",
            "--raw-img-dir", sample_images_dir,
            "--raw-csv-path", sample_csv_file,
            "--processed-dir", str(processed_dir2),
            "--target-size", "256", "256",
        ])
        
        assert exit_code == 0
        images_dir = processed_dir2 / "images"
        for img_file in images_dir.glob("*.jpg"):
            img = Image.open(img_file)
            assert img.size == (256, 256)

    def test_main_ddr2019_output_name(self, temp_dir, sample_images_dir, sample_csv_file):
        """Test that --output-name sets folder name under data/processed (default base)."""
        # Run from temp_dir so output goes to temp_dir/data/processed/<output-name>
        orig_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            exit_code = main([
                "ddr2019",
                "--raw-img-dir", sample_images_dir,
                "--raw-csv-path", sample_csv_file,
                "--output-name", "ddr2019_384",
            ])
            assert exit_code == 0
            out_dir = Path("data/processed/ddr2019_384")
            assert (out_dir / "images").exists()
            assert (out_dir / "labels.csv").exists()
        finally:
            os.chdir(orig_cwd)
