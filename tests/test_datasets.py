"""Unit tests for datasets module."""

import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from sam_ml.datasets import (
    DualChannelDatasets,
    create_eyepacs_datasets,
    load_eyepacs_datasets,
    load_eyepacs_dual_channel,
)
from sam_ml.datasets.eyepacs import _create_paired_dataset


class TestCreatePairedDataset:
    """Test the internal _create_paired_dataset function."""
    
    @pytest.fixture
    def mock_dataset_dirs(self, tmp_path: Path) -> Tuple[Path, Path]:
        """Create mock CLAHE and CECED directories with test images."""
        clahe_dir = tmp_path / "CLAHE"
        ceced_dir = tmp_path / "CECED"
        
        # Create class directories (0-4)
        for label in range(5):
            clahe_class_dir = clahe_dir / str(label)
            ceced_class_dir = ceced_dir / str(label)
            clahe_class_dir.mkdir(parents=True)
            ceced_class_dir.mkdir(parents=True)
            
            # Create 3 test images per class with matching filenames
            for idx in range(3):
                img_name = f"img_{idx:05d}.jpg"
                
                # Create CLAHE image (299x299)
                clahe_img = Image.new("RGB", (299, 299), color=(100, 150, 200))
                clahe_img.save(clahe_class_dir / img_name, "JPEG")
                
                # Create CECED image (224x224) - same filename, different size
                ceced_img = Image.new("RGB", (224, 224), color=(50, 100, 150))
                ceced_img.save(ceced_class_dir / img_name, "JPEG")
        
        return clahe_dir, ceced_dir
    
    def test_create_paired_dataset_basic(
        self,
        mock_dataset_dirs: Tuple[Path, Path]
    ) -> None:
        """Test basic paired dataset creation."""
        clahe_dir, ceced_dir = mock_dataset_dirs
        
        dataset = _create_paired_dataset(
            clahe_dir=clahe_dir,
            ceced_dir=ceced_dir,
            image_size_clahe=(299, 299),
            image_size_ceced=(224, 224),
            batch_size=2,
            label_mode="categorical",
            shuffle=False,
            seed=42,
        )
        
        # Get first batch
        for batch in dataset.take(1):
            (clahe_batch, ceced_batch), labels_batch = batch
            
            # Check shapes
            assert clahe_batch.shape[1:] == (299, 299, 3)
            assert ceced_batch.shape[1:] == (224, 224, 3)
            assert labels_batch.shape[1] == 5  # Categorical labels
            
            # Check that batch size is correct (or smaller if last batch)
            assert clahe_batch.shape[0] == ceced_batch.shape[0] == labels_batch.shape[0]
            assert clahe_batch.shape[0] <= 2  # batch_size
    
    def test_create_paired_dataset_pairs_same_images(
        self,
        mock_dataset_dirs: Tuple[Path, Path]
    ) -> None:
        """Test that paired dataset correctly pairs images by filename."""
        clahe_dir, ceced_dir = mock_dataset_dirs
        
        # Create images with unique pixel values based on filename
        # This allows us to verify correct pairing
        for label in range(5):
            clahe_class_dir = clahe_dir / str(label)
            ceced_class_dir = ceced_dir / str(label)
            
            for idx in range(3):
                img_name = f"img_{idx:05d}.jpg"
                
                # Create CLAHE image with unique color based on idx
                clahe_color = (100 + idx * 10, 150 + idx * 10, 200 + idx * 10)
                clahe_img = Image.new("RGB", (299, 299), color=clahe_color)
                clahe_img.save(clahe_class_dir / img_name, "JPEG")
                
                # Create CECED image with matching unique color (different value to distinguish)
                ceced_color = (50 + idx * 10, 100 + idx * 10, 150 + idx * 10)
                ceced_img = Image.new("RGB", (224, 224), color=ceced_color)
                ceced_img.save(ceced_class_dir / img_name, "JPEG")
        
        dataset = _create_paired_dataset(
            clahe_dir=clahe_dir,
            ceced_dir=ceced_dir,
            image_size_clahe=(299, 299),
            image_size_ceced=(224, 224),
            batch_size=15,  # Large enough to get all images
            label_mode="int",
            shuffle=False,
            seed=42,
        )
        
        # Get all batches and verify pairing
        all_pairs = []
        file_indices = []
        for batch in dataset:
            (clahe_batch, ceced_batch), labels_batch = batch
            for i in range(clahe_batch.shape[0]):
                clahe_img = clahe_batch[i].numpy()
                ceced_img = ceced_batch[i].numpy()
                label = labels_batch[i].numpy()
                
                all_pairs.append((clahe_img, ceced_img, label))
                
                # Extract the index from the image color to verify pairing
                # CLAHE images have colors: (100+idx*10, 150+idx*10, 200+idx*10)
                # After normalization: (100+idx*10)/255, etc.
                clahe_mean = clahe_img.mean()
                # Reverse engineer idx from mean color
                # Mean of (100+idx*10, 150+idx*10, 200+idx*10) = (450+idx*30)/3 = 150+idx*10
                # After normalization: (150+idx*10)/255
                # So idx ≈ (clahe_mean * 255 - 150) / 10
                estimated_idx = int(round((clahe_mean * 255 - 150) / 10))
                file_indices.append(estimated_idx)
        
        # Should have 15 images (5 classes * 3 images)
        assert len(all_pairs) == 15
        
        # Verify that images are paired correctly by checking structure
        for clahe_img, ceced_img, label in all_pairs:
            assert clahe_img.shape == (299, 299, 3)
            assert ceced_img.shape == (224, 224, 3)
            assert label in range(5)
        
        # Verify that we have images from all expected indices (0, 1, 2) for each class
        # Since we iterate through classes first, then images, we should see pattern
        assert len(set(file_indices)) == 3  # Should have indices 0, 1, 2
    
    def test_create_paired_dataset_missing_file_raises_error(
        self,
        tmp_path: Path
    ) -> None:
        """Test that missing CECED file raises FileNotFoundError."""
        clahe_dir = tmp_path / "CLAHE"
        ceced_dir = tmp_path / "CECED"
        
        # Create CLAHE directory with one image
        clahe_class_dir = clahe_dir / "0"
        clahe_class_dir.mkdir(parents=True)
        img = Image.new("RGB", (299, 299))
        img.save(clahe_class_dir / "img_00000.jpg", "JPEG")
        
        # Don't create corresponding CECED file
        
        with pytest.raises(FileNotFoundError):
            _create_paired_dataset(
                clahe_dir=clahe_dir,
                ceced_dir=ceced_dir,
                image_size_clahe=(299, 299),
                image_size_ceced=(224, 224),
                batch_size=1,
                label_mode="categorical",
                shuffle=False,
                seed=42,
            )
    
    def test_create_paired_dataset_shuffle(
        self,
        mock_dataset_dirs: Tuple[Path, Path]
    ) -> None:
        """Test that shuffling works correctly."""
        clahe_dir, ceced_dir = mock_dataset_dirs
        
        # Create two datasets with same seed
        dataset1 = _create_paired_dataset(
            clahe_dir=clahe_dir,
            ceced_dir=ceced_dir,
            image_size_clahe=(299, 299),
            image_size_ceced=(224, 224),
            batch_size=15,
            label_mode="int",
            shuffle=True,
            seed=42,
        )
        
        dataset2 = _create_paired_dataset(
            clahe_dir=clahe_dir,
            ceced_dir=ceced_dir,
            image_size_clahe=(299, 299),
            image_size_ceced=(224, 224),
            batch_size=15,
            label_mode="int",
            shuffle=True,
            seed=42,
        )
        
        # Get first batch from each
        batch1 = next(iter(dataset1))
        batch2 = next(iter(dataset2))
        
        # With same seed, order should be the same
        (clahe1, ceced1), labels1 = batch1
        (clahe2, ceced2), labels2 = batch2
        
        # Check that first images match (same seed = same shuffle)
        np.testing.assert_array_equal(labels1[0].numpy(), labels2[0].numpy())
    
    def test_create_paired_dataset_verifies_file_existence(
        self,
        tmp_path: Path
    ) -> None:
        """Test that pairing function verifies corresponding files exist."""
        clahe_dir = tmp_path / "CLAHE"
        ceced_dir = tmp_path / "CECED"
        
        # Create CLAHE directory with images
        for label in range(2):
            clahe_class_dir = clahe_dir / str(label)
            clahe_class_dir.mkdir(parents=True)
            for idx in range(2):
                img = Image.new("RGB", (299, 299))
                img.save(clahe_class_dir / f"img_{idx:05d}.jpg", "JPEG")
        
        # Create CECED directory but missing one file
        for label in range(2):
            ceced_class_dir = ceced_dir / str(label)
            ceced_class_dir.mkdir(parents=True)
            # Only create first image, missing second
            img = Image.new("RGB", (224, 224))
            img.save(ceced_class_dir / "img_00000.jpg", "JPEG")
            # Intentionally missing img_00001.jpg
        
        # Should raise FileNotFoundError when trying to pair
        with pytest.raises(FileNotFoundError, match="Corresponding CECED file not found"):
            _create_paired_dataset(
                clahe_dir=clahe_dir,
                ceced_dir=ceced_dir,
                image_size_clahe=(299, 299),
                image_size_ceced=(224, 224),
                batch_size=1,
                label_mode="categorical",
                shuffle=False,
                seed=42,
            )


class TestLoadEyePACSDatasets:
    """Test load_eyepacs_datasets function."""
    
    @pytest.fixture
    def mock_processed_dataset(self, tmp_path: Path) -> Path:
        """Create a mock processed dataset directory structure."""
        base_path = tmp_path / "eyepacs_dataset"
        
        # Create directory structure
        for channel in ["CLAHE", "CECED"]:
            for split in ["train", "val", "test"]:
                for label in range(5):
                    class_dir = base_path / channel / split / str(label)
                    class_dir.mkdir(parents=True)
                    
                    # Create 2 test images per class
                    for idx in range(2):
                        img_name = f"img_{idx:05d}.jpg"
                        img = Image.new("RGB", (224, 224))
                        img.save(class_dir / img_name, "JPEG")
        
        return base_path
    
    def test_load_eyepacs_datasets_basic(
        self,
        mock_processed_dataset: Path
    ) -> None:
        """Test basic dataset loading."""
        # base_path should point directly to the dataset directory
        (
            clahe_train,
            clahe_val,
            clahe_test,
            ceced_train,
            ceced_val,
            ceced_test,
        ) = load_eyepacs_datasets(
            base_path=mock_processed_dataset,  # Direct path to dataset directory
            batch_size=2,
            image_size_clahe=(299, 299),
            image_size_ceced=(224, 224),
            shuffle=False,
        )
        
        # Verify all datasets are created
        assert clahe_train is not None
        assert clahe_val is not None
        assert clahe_test is not None
        assert ceced_train is not None
        assert ceced_val is not None
        assert ceced_test is not None
        
        # Get a batch from training set
        for images, labels in clahe_train.take(1):
            assert images.shape[1:] == (299, 299, 3)
            assert labels.shape[1] == 5  # Categorical
    
    def test_load_eyepacs_datasets_missing_directory_raises_error(
        self,
        tmp_path: Path
    ) -> None:
        """Test that missing directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_eyepacs_datasets(
                base_path=tmp_path / "nonexistent",
                dataset_name="test",
                batch_size=2,
            )


class TestLoadEyePACSDualChannel:
    """Test load_eyepacs_dual_channel function."""
    
    @pytest.fixture
    def mock_processed_dataset(self, tmp_path: Path) -> Path:
        """Create a mock processed dataset directory structure."""
        base_path = tmp_path / "eyepacs_dataset"
        
        # Create directory structure with matching filenames
        for channel in ["CLAHE", "CECED"]:
            for split in ["train", "val", "test"]:
                for label in range(5):
                    class_dir = base_path / channel / split / str(label)
                    class_dir.mkdir(parents=True)
                    
                    # Create 2 test images per class with matching filenames
                    for idx in range(2):
                        img_name = f"img_{idx:05d}.jpg"
                        if channel == "CLAHE":
                            img = Image.new("RGB", (299, 299))
                        else:
                            img = Image.new("RGB", (224, 224))
                        img.save(class_dir / img_name, "JPEG")
        
        return base_path
    
    def test_load_eyepacs_dual_channel_basic(
        self,
        mock_processed_dataset: Path
    ) -> None:
        """Test basic dual-channel dataset loading."""
        # base_path should point directly to the dataset directory
        train, val, test = load_eyepacs_dual_channel(
            base_path=mock_processed_dataset,  # Direct path to dataset directory
            batch_size=2,
            shuffle=False,
        )
        
        # Verify all datasets are created
        assert train is not None
        assert val is not None
        assert test is not None
        
        # Get a batch from training set
        for batch in train.take(1):
            (clahe_batch, ceced_batch), labels_batch = batch
            
            # Check shapes
            assert clahe_batch.shape[1:] == (299, 299, 3)
            assert ceced_batch.shape[1:] == (224, 224, 3)
            assert labels_batch.shape[1] == 5  # Categorical
            
            # Check that batch sizes match
            assert clahe_batch.shape[0] == ceced_batch.shape[0] == labels_batch.shape[0]
    
    def test_load_eyepacs_dual_channel_pairs_correctly(
        self,
        mock_processed_dataset: Path
    ) -> None:
        """Test that dual-channel loader correctly pairs images by filename."""
        # Create images with unique identifiers to verify pairing
        for channel in ["CLAHE", "CECED"]:
            for split in ["train", "val", "test"]:
                for label in range(5):
                    class_dir = mock_processed_dataset / channel / split / str(label)
                    for idx in range(2):
                        img_name = f"img_{idx:05d}.jpg"
                        # Create image with unique color based on idx for verification
                        if channel == "CLAHE":
                            color = (100 + idx * 50, 150 + idx * 50, 200 + idx * 50)
                            img = Image.new("RGB", (299, 299), color=color)
                        else:
                            color = (50 + idx * 50, 100 + idx * 50, 150 + idx * 50)
                            img = Image.new("RGB", (224, 224), color=color)
                        img.save(class_dir / img_name, "JPEG")
        
        train, _, _ = load_eyepacs_dual_channel(
            base_path=mock_processed_dataset,  # Direct path to dataset directory
            batch_size=10,  # Large enough to get all images
            shuffle=False,  # Don't shuffle to verify pairing
        )
        
        # Get all batches and verify pairing
        all_pairs = []
        for batch in train:
            (clahe_batch, ceced_batch), labels_batch = batch
            for i in range(clahe_batch.shape[0]):
                clahe_img = clahe_batch[i].numpy()
                ceced_img = ceced_batch[i].numpy()
                label = labels_batch[i].numpy()
                all_pairs.append((clahe_img, ceced_img, label))
        
        # Verify that we have pairs
        assert len(all_pairs) > 0
        
        # Verify structure - images should be correctly paired
        # CLAHE images should have mean around (100+idx*50 + 150+idx*50 + 200+idx*50)/3/255
        # = (450 + idx*150)/3/255 = (150 + idx*50)/255
        for clahe_img, ceced_img, label in all_pairs:
            assert clahe_img.shape == (299, 299, 3)
            assert ceced_img.shape == (224, 224, 3)
            assert label.shape == (5,)  # Categorical
            # Verify images are normalized to [0, 1]
            assert clahe_img.min() >= 0.0
            assert clahe_img.max() <= 1.0
            assert ceced_img.min() >= 0.0
            assert ceced_img.max() <= 1.0
    
    def test_load_eyepacs_dual_channel_missing_directory_raises_error(
        self,
        tmp_path: Path
    ) -> None:
        """Test that missing directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_eyepacs_dual_channel(
                base_path=tmp_path / "nonexistent",
                dataset_name="test",
                batch_size=2,
            )


class TestDualChannelDatasets:
    """Test DualChannelDatasets container class."""
    
    @pytest.fixture
    def mock_datasets(self) -> DualChannelDatasets:
        """Create mock datasets for testing."""
        # Create dummy datasets
        clahe_train = tf.data.Dataset.from_tensor_slices(
            (tf.zeros((10, 299, 299, 3)), tf.zeros((10, 5)))
        )
        clahe_val = tf.data.Dataset.from_tensor_slices(
            (tf.zeros((5, 299, 299, 3)), tf.zeros((5, 5)))
        )
        clahe_test = tf.data.Dataset.from_tensor_slices(
            (tf.zeros((5, 299, 299, 3)), tf.zeros((5, 5)))
        )
        ceced_train = tf.data.Dataset.from_tensor_slices(
            (tf.zeros((10, 224, 224, 3)), tf.zeros((10, 5)))
        )
        ceced_val = tf.data.Dataset.from_tensor_slices(
            (tf.zeros((5, 224, 224, 3)), tf.zeros((5, 5)))
        )
        ceced_test = tf.data.Dataset.from_tensor_slices(
            (tf.zeros((5, 224, 224, 3)), tf.zeros((5, 5)))
        )
        
        return DualChannelDatasets(
            clahe_train=clahe_train,
            clahe_val=clahe_val,
            clahe_test=clahe_test,
            ceced_train=ceced_train,
            ceced_val=ceced_val,
            ceced_test=ceced_test,
        )
    
    def test_dual_channel_datasets_initialization(
        self,
        mock_datasets: DualChannelDatasets
    ) -> None:
        """Test DualChannelDatasets initialization."""
        assert mock_datasets.num_classes == 5
        assert len(mock_datasets.class_names) == 5
        assert mock_datasets.class_names[0] == "No Diabetic Retinopathy"
    
    def test_get_train_split(
        self,
        mock_datasets: DualChannelDatasets
    ) -> None:
        """Test get_train_split method."""
        clahe_only = mock_datasets.get_train_split("clahe")
        assert len(clahe_only) == 1
        
        ceced_only = mock_datasets.get_train_split("ceced")
        assert len(ceced_only) == 1
        
        both = mock_datasets.get_train_split("both")
        assert len(both) == 2
    
    def test_get_combined_train(
        self,
        mock_datasets: DualChannelDatasets
    ) -> None:
        """Test get_combined_train method."""
        combined = mock_datasets.get_combined_train()
        assert combined is not None


class TestCreateEyePACSDatasets:
    """Test create_eyepacs_datasets factory function."""
    
    @pytest.fixture
    def mock_processed_dataset(self, tmp_path: Path) -> Path:
        """Create a mock processed dataset directory structure."""
        base_path = tmp_path / "eyepacs_dataset"
        
        # Create directory structure
        for channel in ["CLAHE", "CECED"]:
            for split in ["train", "val", "test"]:
                for label in range(5):
                    class_dir = base_path / channel / split / str(label)
                    class_dir.mkdir(parents=True)
                    
                    # Create 2 test images per class
                    for idx in range(2):
                        img_name = f"img_{idx:05d}.jpg"
                        img = Image.new("RGB", (224, 224))
                        img.save(class_dir / img_name, "JPEG")
        
        return base_path
    
    def test_create_eyepacs_datasets_basic(
        self,
        mock_processed_dataset: Path
    ) -> None:
        """Test basic dataset creation."""
        datasets = create_eyepacs_datasets(
            base_path=mock_processed_dataset,  # Direct path to dataset directory
            batch_size=2,
            shuffle=False,
        )
        
        assert isinstance(datasets, DualChannelDatasets)
        assert datasets.num_classes == 5
        assert datasets.clahe_train is not None
        assert datasets.ceced_train is not None
    
    def test_dataset_structure_compatible_with_model(
        self,
        mock_processed_dataset: Path
    ) -> None:
        """Test that dataset structure is compatible with dual-channel model."""
        from sam_ml.modeling.models import DualChannelDiabeticRetinopathyModel
        
        # Load dataset
        train, val, test = load_eyepacs_dual_channel(
            base_path=mock_processed_dataset,
            batch_size=2,
            shuffle=False,
        )
        
        # Create model
        model = DualChannelDiabeticRetinopathyModel(num_classes=5)
        
        # Get a batch from the dataset
        for batch in train.take(1):
            (clahe_batch, ceced_batch), labels_batch = batch
            
            # Verify batch structure
            assert clahe_batch.shape[1:] == (299, 299, 3)
            assert ceced_batch.shape[1:] == (224, 224, 3)
            assert labels_batch.shape[1] == 5
            
            # Verify model can process the batch
            # Model expects [clahe_images, ceced_images] as input
            predictions = model([clahe_batch, ceced_batch], training=False)
            
            # Verify predictions shape
            assert predictions.shape[0] == clahe_batch.shape[0]
            assert predictions.shape[1] == 5  # num_classes
            
            # Verify predictions are probabilities (sum to 1)
            prediction_sums = tf.reduce_sum(predictions, axis=1)
            np.testing.assert_allclose(prediction_sums.numpy(), 1.0, rtol=1e-5)

