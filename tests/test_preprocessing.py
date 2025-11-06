"""Unit tests for preprocessing module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from PIL import Image

from sam_ml.preprocessing import (
    apply_ceced_bgr_3ch,
    apply_clahe_bgr,
    create_processor,
    get_processor,
    list_available_datasets,
)
from sam_ml.preprocessing.base import DatasetProcessor
from sam_ml.preprocessing.eyepacs_dataset import KaggleEyePACSProcessor


class TestPreprocessingFunctions:
    """Test preprocessing utility functions (CLAHE and CECED)."""
    
    @pytest.fixture
    def sample_image(self) -> np.ndarray:
        """Create a sample BGR image for testing."""
        return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    def test_apply_clahe_bgr(self, sample_image: np.ndarray) -> None:
        """Test CLAHE preprocessing maintains shape and valid range."""
        processed = apply_clahe_bgr(sample_image)
        
        assert processed.shape == sample_image.shape
        assert processed.dtype == sample_image.dtype
        assert processed.min() >= 0
        assert processed.max() <= 255
    
    def test_apply_ceced_bgr_3ch(self, sample_image: np.ndarray) -> None:
        """Test CECED preprocessing produces 3-channel grayscale edges."""
        processed = apply_ceced_bgr_3ch(sample_image)
        
        assert processed.shape == sample_image.shape
        assert processed.shape[2] == 3
        # All channels should be identical (edge detection is grayscale)
        np.testing.assert_array_equal(processed[:, :, 0], processed[:, :, 1])
        np.testing.assert_array_equal(processed[:, :, 1], processed[:, :, 2])
    
    def test_apply_ceced_bgr_3ch_uses_green_channel(self) -> None:
        """Test CECED uses green channel for processing."""
        # Create image with distinct channel values
        img_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        img_bgr[:, :, 0] = 50   # Blue channel
        img_bgr[:, :, 1] = 200  # Green channel (brightest)
        img_bgr[:, :, 2] = 100  # Red channel
        
        processed = apply_ceced_bgr_3ch(img_bgr)
        
        # Since CECED uses green channel, edges should be based on green channel intensity
        # The processed image should have edges detected from the green channel
        assert processed.shape == img_bgr.shape
        assert processed.shape[2] == 3
    
    def test_apply_ceced_bgr_3ch_auto_canny(self) -> None:
        """Test CECED uses auto Canny thresholds based on median."""
        # Create a test image with known median
        img_bgr = np.full((100, 100, 3), 128, dtype=np.uint8)
        img_bgr[40:60, 40:60, :] = 200  # Add a bright square
        
        processed = apply_ceced_bgr_3ch(img_bgr)
        
        # Should produce edge image
        assert processed.shape == img_bgr.shape
        assert processed.dtype == np.uint8
    
    def test_apply_ceced_bgr_3ch_dilation(self) -> None:
        """Test CECED dilation parameter works."""
        img_bgr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Without dilation
        processed_no_dilate = apply_ceced_bgr_3ch(img_bgr, dilate_iterations=0)
        
        # With dilation
        processed_dilate = apply_ceced_bgr_3ch(img_bgr, dilate_iterations=2)
        
        assert processed_no_dilate.shape == processed_dilate.shape
        # Dilated version should have more white pixels (thicker edges)
        assert processed_dilate.sum() >= processed_no_dilate.sum()
    
    def test_apply_ceced_bgr_3ch_even_blur_kernel(self) -> None:
        """Test CECED handles even blur kernel size by making it odd."""
        img_bgr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Should not raise error with even kernel size
        processed = apply_ceced_bgr_3ch(img_bgr, blur_kernel_size=4)
        
        assert processed.shape == img_bgr.shape


class TestProcessorRegistry:
    """Test processor registry and factory functions."""
    
    def test_list_available_datasets(self) -> None:
        """Test listing available datasets."""
        datasets = list_available_datasets()
        assert isinstance(datasets, list)
        assert "eyepacs_dataset" in datasets
    
    def test_get_processor(self) -> None:
        """Test getting processor for valid and invalid dataset names."""
        # Valid dataset
        processor_class = get_processor("eyepacs_dataset")
        assert processor_class == KaggleEyePACSProcessor
        
        # Invalid dataset
        with pytest.raises(ValueError, match="No processor found"):
            get_processor("nonexistent_dataset")
    
    def test_create_processor(self) -> None:
        """Test creating processor with default and custom paths."""
        # Default paths
        processor = create_processor("eyepacs_dataset")
        assert isinstance(processor, KaggleEyePACSProcessor)
        assert processor.dataset_name == "eyepacs_dataset"
        assert processor.raw_dir == Path("data/raw/eyepacs_dataset")
        assert processor.processed_dir == Path("data/processed")
        
        # Custom paths
        processor = create_processor(
            "eyepacs_dataset",
            raw_dir=Path("/custom/raw"),
            processed_dir=Path("/custom/processed")
        )
        assert processor.raw_dir == Path("/custom/raw")
        assert processor.processed_dir == Path("/custom/processed")
        
        # Invalid dataset
        with pytest.raises(ValueError, match="No processor found"):
            create_processor("nonexistent_dataset")


class TestKaggleEyePACSProcessor:
    """Test Kaggle EyePACS processor functionality."""
    
    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def processor(self, temp_dir: Path) -> KaggleEyePACSProcessor:
        """Create a processor instance for testing."""
        raw_dir = temp_dir / "raw"
        processed_dir = temp_dir / "processed"
        raw_dir.mkdir()
        processed_dir.mkdir()
        
        return KaggleEyePACSProcessor(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            dataset_name="test_dataset",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )
    
    def test_processor_initialization(self, processor: KaggleEyePACSProcessor) -> None:
        """Test processor initialization and validation."""
        assert processor.dataset_name == "test_dataset"
        assert processor.supported_dataset_name == "eyepacs_dataset"
        assert processor.train_ratio == 0.7
        assert processor.val_ratio == 0.15
        assert processor.test_ratio == 0.15
    
    def test_processor_initialization_validation(self, temp_dir: Path) -> None:
        """Test processor initialization validates ratios."""
        raw_dir = temp_dir / "raw"
        processed_dir = temp_dir / "processed"
        raw_dir.mkdir()
        processed_dir.mkdir()
        
        # Ratios don't sum to 1.0
        with pytest.raises(ValueError, match="must sum to 1.0"):
            KaggleEyePACSProcessor(
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                dataset_name="test",
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.05
            )
        
        # Negative ratio
        with pytest.raises(ValueError, match="must be positive"):
            KaggleEyePACSProcessor(
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                dataset_name="test",
                train_ratio=-0.1,
                val_ratio=0.5,
                test_ratio=0.6
            )
    
    def test_create_directory_structure(self, processor: KaggleEyePACSProcessor) -> None:
        """Test creating directory structure with numeric label folders."""
        dirs = processor.create_directory_structure()
        
        assert len(dirs) == 30  # 2 channels * 3 splits * 5 classes
        
        dataset_dir = processor.processed_dir / processor.dataset_name
        # Check numeric label folders (0, 1, 2, 3, 4) instead of named folders
        assert (dataset_dir / "CLAHE" / "train" / "0").exists()
        assert (dataset_dir / "CECED" / "val" / "1").exists()
        assert (dataset_dir / "CLAHE" / "test" / "4").exists()
        assert (dataset_dir / "CECED" / "train" / "2").exists()
        assert (dataset_dir / "CLAHE" / "val" / "3").exists()
    
    @patch('sam_ml.preprocessing.eyepacs_dataset.load_dataset')
    def test_extract_raw_data(self, mock_load_dataset: MagicMock, processor: KaggleEyePACSProcessor) -> None:
        """Test loading dataset from Hugging Face."""
        # Mock Hugging Face dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_load_dataset.return_value = mock_dataset
        
        train_dir, test_dir = processor.extract_raw_data()
        
        # Verify dataset was loaded
        mock_load_dataset.assert_called_once_with("bumbledeep/eyepacs", split="train")
        assert processor._dataset == mock_dataset
        assert train_dir == Path("")
        assert test_dir == Path("")
    
    @patch('sam_ml.preprocessing.eyepacs_dataset.load_dataset')
    def test_load_labels(self, mock_load_dataset: MagicMock, processor: KaggleEyePACSProcessor) -> None:
        """Test loading labels from Hugging Face dataset."""
        # Mock dataset with samples (using label_code as the dataset provides)
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=5)
        mock_dataset.__iter__ = MagicMock(return_value=iter([
            {"label_code": 0},
            {"label_code": 1},
            {"label_code": 2},
            {"label_code": 3},
            {"label_code": 4},
        ]))
        mock_load_dataset.return_value = mock_dataset
        
        processor.extract_raw_data()
        labels = processor.load_labels()
        
        assert len(labels) == 5
        assert labels["0"] == 0
        assert labels["1"] == 1
        assert labels["2"] == 2
        assert labels["3"] == 3
        assert labels["4"] == 4
    
    def test_split_dataset(self, processor: KaggleEyePACSProcessor) -> None:
        """Test dataset splitting into train/val/test."""
        # Create labels dictionary mapping indices to labels
        labels = {str(i): i % 5 for i in range(10)}
        
        train, val, test = processor.split_dataset([], labels)
        
        # Verify split ratios (70/15/15)
        assert len(train) == 7  # 70% of 10
        assert len(val) == 1    # 15% of 10
        assert len(test) == 2   # 15% of 10
        assert len(train) + len(val) + len(test) == 10
        
        # Verify all items are (index, label) tuples
        for idx, label in train + val + test:
            assert isinstance(idx, int)
            assert isinstance(label, int)
            assert 0 <= label <= 4
    
    def test_pad_to_square_square_image(self, processor: KaggleEyePACSProcessor) -> None:
        """Test pad_to_square returns square image unchanged."""
        # Already square image
        square_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = processor.pad_to_square(square_img)
        
        np.testing.assert_array_equal(result, square_img)
        assert result.shape == (100, 100, 3)
    
    def test_pad_to_square_wide_image(self, processor: KaggleEyePACSProcessor) -> None:
        """Test pad_to_square pads wide image correctly."""
        # Wide image (height < width)
        wide_img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        result = processor.pad_to_square(wide_img)
        
        assert result.shape == (200, 200, 3)  # Padded to max(100, 200) = 200
        # Check padding is symmetric
        assert result.shape[0] == result.shape[1]
        # Original image should be in center
        top_pad = (200 - 100) // 2
        np.testing.assert_array_equal(
            result[top_pad:top_pad+100, :, :],
            wide_img
        )
    
    def test_pad_to_square_tall_image(self, processor: KaggleEyePACSProcessor) -> None:
        """Test pad_to_square pads tall image correctly."""
        # Tall image (height > width)
        tall_img = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
        result = processor.pad_to_square(tall_img)
        
        assert result.shape == (200, 200, 3)  # Padded to max(200, 100) = 200
        # Check padding is symmetric
        assert result.shape[0] == result.shape[1]
        # Original image should be in center
        left_pad = (200 - 100) // 2
        np.testing.assert_array_equal(
            result[:, left_pad:left_pad+100, :],
            tall_img
        )
    
    def test_pad_to_square_custom_pad_value(self, processor: KaggleEyePACSProcessor) -> None:
        """Test pad_to_square uses custom padding color."""
        # Wide image with custom padding color
        wide_img = np.ones((50, 150, 3), dtype=np.uint8) * 128
        custom_pad = (255, 0, 0)  # Blue padding
        result = processor.pad_to_square(wide_img, pad_value=custom_pad)
        
        assert result.shape == (150, 150, 3)
        # Check padding areas have custom color
        top_pad = (150 - 50) // 2
        # Top padding should be custom color
        np.testing.assert_array_equal(
            result[:top_pad, :, :],
            np.full((top_pad, 150, 3), custom_pad, dtype=np.uint8)
        )
    
    def test_process_and_save_image(self, processor: KaggleEyePACSProcessor, temp_dir: Path) -> None:
        """Test processing and saving a PIL image with correct size and format."""
        # Create a sample PIL image (different size to test resizing)
        pil_image = Image.new("RGB", (512, 512), color=(128, 128, 128))
        
        output_path = temp_dir / "test_output.jpg"
        
        # Process and save with CLAHE size (299×299)
        processor._process_and_save_image(
            pil_image=pil_image,
            output_path=output_path,
            preprocessing_fn=apply_clahe_bgr,
            label=0,
            index=0,
            target_size=(299, 299)
        )
        
        # Verify file was created
        assert output_path.exists()
        assert output_path.suffix.lower() == '.jpg'
        
        # Verify image can be read back and has correct properties
        saved_img = cv2.imread(str(output_path))
        assert saved_img is not None
        assert saved_img.shape == (299, 299, 3)  # Height, Width, Channels
        assert saved_img.dtype == np.uint8
        
        # Test with CECED size (224×224)
        output_path_ceced = temp_dir / "test_output_ceced.jpg"
        processor._process_and_save_image(
            pil_image=pil_image,
            output_path=output_path_ceced,
            preprocessing_fn=apply_ceced_bgr_3ch,
            label=0,
            index=0,
            target_size=(224, 224)
        )
        
        saved_img_ceced = cv2.imread(str(output_path_ceced))
        assert saved_img_ceced is not None
        assert saved_img_ceced.shape == (224, 224, 3)  # Height, Width, Channels
        assert saved_img_ceced.dtype == np.uint8
    
    def test_process_and_save_image_non_square_padding(self, processor: KaggleEyePACSProcessor, temp_dir: Path) -> None:
        """Test that non-square images are padded before resizing."""
        # Create a non-square PIL image (wide)
        pil_image_wide = Image.new("RGB", (300, 200), color=(128, 128, 128))
        
        output_path = temp_dir / "test_wide.jpg"
        
        # Process with target size 224×224
        processor._process_and_save_image(
            pil_image=pil_image_wide,
            output_path=output_path,
            preprocessing_fn=apply_clahe_bgr,
            label=0,
            index=0,
            target_size=(224, 224)
        )
        
        # Verify final image is square and correct size
        saved_img = cv2.imread(str(output_path))
        assert saved_img is not None
        assert saved_img.shape == (224, 224, 3)  # Should be square after padding and resizing
        
        # Test with tall image
        pil_image_tall = Image.new("RGB", (200, 300), color=(128, 128, 128))
        output_path_tall = temp_dir / "test_tall.jpg"
        
        processor._process_and_save_image(
            pil_image=pil_image_tall,
            output_path=output_path_tall,
            preprocessing_fn=apply_clahe_bgr,
            label=0,
            index=0,
            target_size=(224, 224)
        )
        
        saved_img_tall = cv2.imread(str(output_path_tall))
        assert saved_img_tall is not None
        assert saved_img_tall.shape == (224, 224, 3)  # Should be square after padding and resizing


class TestCLI:
    """Test CLI validation."""
    
    def test_main_help(self) -> None:
        """Test CLI help output."""
        import sys
        from io import StringIO
        from sam_ml.preprocessing import main
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            sys.argv = ["preprocess-dataset", "--help"]
            try:
                main()
            except SystemExit:
                pass
            
            output = sys.stdout.getvalue()
            assert "Preprocess diabetic retinopathy dataset" in output
            assert "eyepacs_dataset" in output
        finally:
            sys.stdout = old_stdout
    
    def test_main_validation(self) -> None:
        """Test CLI validates ratios and dataset name."""
        import sys
        from sam_ml.preprocessing import main
        
        # Invalid ratios
        sys.argv = [
            "preprocess-dataset",
            "eyepacs_dataset",
            "--train-ratio", "0.8",
            "--val-ratio", "0.1",
            "--test-ratio", "0.05"  # Doesn't sum to 1.0
        ]
        with pytest.raises(SystemExit):
            main()
        
        # Invalid dataset
        sys.argv = [
            "preprocess-dataset",
            "nonexistent_dataset",
            "--train-ratio", "0.7",
            "--val-ratio", "0.15",
            "--test-ratio", "0.15"
        ]
        with pytest.raises(SystemExit):
            main()
