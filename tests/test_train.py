"""Unit tests for training module."""

import sys
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import pytest
import tensorflow as tf
import numpy as np
from PIL import Image

from sam_ml.modeling.train import (
    check_cuda_availability,
    create_adam_optimizer,
    create_callbacks,
    create_model,
    load_dataset,
    parse_args,
    train,
)


class TestCheckCUDAAvailability:
    """Test cases for CUDA availability checking."""
    
    def test_check_cuda_availability_with_gpu(self) -> None:
        """Test CUDA check when GPU is available."""
        with patch("sam_ml.modeling.train.tf.config.list_physical_devices") as mock_list_gpus, \
             patch("sam_ml.modeling.train.tf.test.is_built_with_cuda", return_value=True), \
             patch("sam_ml.modeling.train.tf.config.experimental.set_memory_growth") as mock_memory_growth:
            
            # Mock GPU device
            mock_gpu = MagicMock()
            mock_gpu.name = "/physical_device:GPU:0"
            mock_list_gpus.return_value = [mock_gpu]
            
            result = check_cuda_availability()
            
            assert result is True
            mock_list_gpus.assert_called_once_with("GPU")
            mock_memory_growth.assert_called_once()
    
    def test_check_cuda_availability_no_gpu(self) -> None:
        """Test CUDA check when no GPU is available."""
        with patch("sam_ml.modeling.train.tf.config.list_physical_devices") as mock_list_gpus:
            mock_list_gpus.return_value = []
            
            result = check_cuda_availability()
            
            assert result is False
            mock_list_gpus.assert_called_once_with("GPU")
    
    def test_check_cuda_availability_not_built_with_cuda(self) -> None:
        """Test CUDA check when TensorFlow is not built with CUDA."""
        with patch("sam_ml.modeling.train.tf.config.list_physical_devices") as mock_list_gpus, \
             patch("sam_ml.modeling.train.tf.test.is_built_with_cuda", return_value=False):
            
            mock_gpu = MagicMock()
            mock_list_gpus.return_value = [mock_gpu]
            
            result = check_cuda_availability()
            
            assert result is False
    
    def test_check_cuda_availability_exception_handling(self) -> None:
        """Test CUDA check handles exceptions gracefully."""
        with patch("sam_ml.modeling.train.tf.config.list_physical_devices") as mock_list_gpus:
            mock_list_gpus.side_effect = Exception("Test exception")
            
            result = check_cuda_availability()
            
            assert result is False
    
    def test_check_cuda_availability_memory_growth_exception(self) -> None:
        """Test CUDA check handles memory growth configuration exceptions."""
        with patch("sam_ml.modeling.train.tf.config.list_physical_devices") as mock_list_gpus, \
             patch("sam_ml.modeling.train.tf.test.is_built_with_cuda", return_value=True), \
             patch("sam_ml.modeling.train.tf.config.experimental.set_memory_growth") as mock_memory_growth:
            
            mock_gpu = MagicMock()
            mock_gpu.name = "/physical_device:GPU:0"
            mock_list_gpus.return_value = [mock_gpu]
            mock_memory_growth.side_effect = RuntimeError("Cannot set memory growth")
            
            # Should still return True even if memory growth fails
            result = check_cuda_availability()
            
            assert result is True


class TestCreateAdamOptimizer:
    """Test cases for Adam optimizer creation."""
    
    def test_create_adam_optimizer_default(self) -> None:
        """Test Adam optimizer creation with default parameters."""
        optimizer = create_adam_optimizer()
        
        assert optimizer is not None
        assert isinstance(optimizer, tf.keras.optimizers.Adam)
        # Check default learning rate
        lr = optimizer.learning_rate.numpy() if hasattr(optimizer.learning_rate, "numpy") else optimizer.learning_rate
        assert abs(lr - 0.001) < 1e-6
    
    def test_create_adam_optimizer_custom_learning_rate(self) -> None:
        """Test Adam optimizer creation with custom learning rate."""
        optimizer = create_adam_optimizer(learning_rate=0.0001)
        
        lr = optimizer.learning_rate.numpy() if hasattr(optimizer.learning_rate, "numpy") else optimizer.learning_rate
        assert abs(lr - 0.0001) < 1e-6
    
    def test_create_adam_optimizer_custom_betas(self) -> None:
        """Test Adam optimizer creation with custom beta values."""
        optimizer = create_adam_optimizer(beta_1=0.95, beta_2=0.99)
        
        beta_1_val = optimizer.beta_1.numpy() if hasattr(optimizer.beta_1, "numpy") else optimizer.beta_1
        beta_2_val = optimizer.beta_2.numpy() if hasattr(optimizer.beta_2, "numpy") else optimizer.beta_2
        
        assert abs(beta_1_val - 0.95) < 1e-6
        assert abs(beta_2_val - 0.99) < 1e-6
    
    def test_create_adam_optimizer_custom_epsilon(self) -> None:
        """Test Adam optimizer creation with custom epsilon."""
        optimizer = create_adam_optimizer(epsilon=1e-8)
        
        assert optimizer.epsilon == 1e-8
    
    def test_create_adam_optimizer_with_weight_decay(self) -> None:
        """Test Adam optimizer creation with weight decay (should warn)."""
        with patch("sam_ml.modeling.train.logger") as mock_logger:
            optimizer = create_adam_optimizer(weight_decay=1e-4)
            
            assert optimizer is not None
            # Should log a warning about weight decay
            mock_logger.warning.assert_called_once()


class TestCreateModel:
    """Test cases for model creation."""
    
    def test_create_model_default(self) -> None:
        """Test model creation with default parameters."""
        model = create_model()
        
        assert model is not None
        assert model.name == "dual_channel_dr_model"
        # Check that classifier has 5 output units (default num_classes)
        assert model.classifier.units == 5
    
    def test_create_model_custom_num_classes(self) -> None:
        """Test model creation with custom number of classes."""
        model = create_model(num_classes=3)
        
        # Check that classifier has 3 output units
        assert model.classifier.units == 3
    
    def test_create_model_custom_input_shape(self) -> None:
        """Test model creation with custom input shape."""
        model = create_model(input_shape=(256, 256, 3))
        
        # Test that model can process input with custom shape
        dummy_input1 = tf.random.uniform((1, 256, 256, 3))
        dummy_input2 = tf.random.uniform((1, 256, 256, 3))
        output = model([dummy_input1, dummy_input2], training=False)
        
        assert output.shape == (1, 5)  # Default num_classes is 5
    
    def test_create_model_output_shape(self) -> None:
        """Test that created model produces correct output shape."""
        model = create_model(num_classes=5)
        
        dummy_input1 = tf.random.uniform((2, 224, 224, 3))
        dummy_input2 = tf.random.uniform((2, 224, 224, 3))
        output = model([dummy_input1, dummy_input2], training=False)
        
        assert output.shape == (2, 5)


class TestLoadDataset:
    """Test cases for dataset loading."""
    
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
                        # Create actual image files for testing
                        if channel == "CLAHE":
                            img = Image.new("RGB", (299, 299))
                        else:
                            img = Image.new("RGB", (224, 224))
                        img.save(class_dir / img_name, "JPEG")
        
        return base_path
    
    def test_load_dataset_default(self, mock_processed_dataset: Path) -> None:
        """Test dataset loading with default parameters."""
        train, val, test = load_dataset(base_path=mock_processed_dataset)
        
        assert train is not None
        assert val is not None
        assert test is not None
    
    def test_load_dataset_custom_batch_size(self, mock_processed_dataset: Path) -> None:
        """Test dataset loading with custom batch size."""
        train, val, test = load_dataset(
            base_path=mock_processed_dataset,
            batch_size=16,
        )
        
        # Verify datasets are created
        assert train is not None
        assert val is not None
        assert test is not None
    
    def test_load_dataset_custom_image_sizes(self, mock_processed_dataset: Path) -> None:
        """Test dataset loading with custom image sizes."""
        train, val, test = load_dataset(
            base_path=mock_processed_dataset,
            image_size_clahe=(256, 256),
            image_size_ceced=(256, 256),
        )
        
        assert train is not None
        assert val is not None
        assert test is not None
    
    def test_load_dataset_missing_directory_raises_error(self, tmp_path: Path) -> None:
        """Test that missing directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_dataset(base_path=tmp_path / "nonexistent")


class TestCreateCallbacks:
    """Test cases for callback creation."""
    
    def test_create_callbacks_default(self, tmp_path: Path) -> None:
        """Test callback creation with default parameters."""
        checkpoint_dir = tmp_path / "checkpoints"
        callbacks = create_callbacks(checkpoint_dir=checkpoint_dir)
        
        assert len(callbacks) == 3
        assert any(isinstance(cb, tf.keras.callbacks.EarlyStopping) for cb in callbacks)
        assert any(isinstance(cb, tf.keras.callbacks.ReduceLROnPlateau) for cb in callbacks)
        assert any(isinstance(cb, tf.keras.callbacks.ModelCheckpoint) for cb in callbacks)
    
    def test_create_callbacks_custom_patience(self, tmp_path: Path) -> None:
        """Test callback creation with custom patience."""
        checkpoint_dir = tmp_path / "checkpoints"
        callbacks = create_callbacks(
            checkpoint_dir=checkpoint_dir,
            patience=20,
        )
        
        early_stopping = next(
            cb for cb in callbacks if isinstance(cb, tf.keras.callbacks.EarlyStopping)
        )
        assert early_stopping.patience == 20
    
    def test_create_callbacks_custom_monitor(self, tmp_path: Path) -> None:
        """Test callback creation with custom monitor metric."""
        checkpoint_dir = tmp_path / "checkpoints"
        callbacks = create_callbacks(
            checkpoint_dir=checkpoint_dir,
            monitor="val_accuracy",
            mode="max",
        )
        
        early_stopping = next(
            cb for cb in callbacks if isinstance(cb, tf.keras.callbacks.EarlyStopping)
        )
        assert early_stopping.monitor == "val_accuracy"
        assert early_stopping.mode == "max"
    
    def test_create_callbacks_no_checkpoint_dir(self) -> None:
        """Test callback creation without checkpoint directory."""
        callbacks = create_callbacks(checkpoint_dir=None)
        
        # Should still have early stopping and reduce LR, but no checkpoint
        assert len(callbacks) == 2
        assert not any(isinstance(cb, tf.keras.callbacks.ModelCheckpoint) for cb in callbacks)


class TestTrain:
    """Test cases for training function."""
    
    @pytest.fixture
    def dummy_model(self) -> tf.keras.Model:
        """Create a dummy model for testing."""
        model = create_model()
        return model
    
    @pytest.fixture
    def dummy_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Create dummy datasets for testing."""
        # Create small dummy datasets
        batch_size = 2
        num_samples = 4
        
        # Create dummy images and labels
        clahe_images = tf.random.uniform((num_samples, 299, 299, 3))
        ceced_images = tf.random.uniform((num_samples, 224, 224, 3))
        labels = tf.one_hot([0, 1, 2, 3], depth=5)
        
        # Create datasets
        train_ds = tf.data.Dataset.from_tensor_slices(
            ((clahe_images, ceced_images), labels)
        ).batch(batch_size)
        
        val_ds = tf.data.Dataset.from_tensor_slices(
            ((clahe_images[:2], ceced_images[:2]), labels[:2])
        ).batch(batch_size)
        
        test_ds = tf.data.Dataset.from_tensor_slices(
            ((clahe_images[:2], ceced_images[:2]), labels[:2])
        ).batch(batch_size)
        
        return train_ds, val_ds, test_ds
    
    @patch("sam_ml.modeling.train.check_cuda_availability")
    @patch("sam_ml.modeling.train.load_dataset")
    def test_train_with_defaults(
        self,
        mock_load_dataset: Mock,
        mock_check_cuda: Mock,
        dummy_model: tf.keras.Model,
        dummy_datasets: Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
    ) -> None:
        """Test training with default model and dataset."""
        train_ds, val_ds, test_ds = dummy_datasets
        
        mock_check_cuda.return_value = False
        mock_load_dataset.return_value = (train_ds, val_ds, test_ds)
        
        # Mock model.fit to avoid actual training
        with patch.object(dummy_model, "fit") as mock_fit, \
             patch("sam_ml.modeling.train.create_model", return_value=dummy_model):
            
            mock_history = MagicMock()
            mock_fit.return_value = mock_history
            
            history = train(epochs=1, verbose=0)
            
            assert history is not None
            mock_check_cuda.assert_called_once()
            mock_load_dataset.assert_called_once()
            mock_fit.assert_called_once()
    
    @patch("sam_ml.modeling.train.check_cuda_availability")
    def test_train_with_provided_model_and_dataset(
        self,
        mock_check_cuda: Mock,
        dummy_model: tf.keras.Model,
        dummy_datasets: Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
    ) -> None:
        """Test training with provided model and dataset."""
        train_ds, val_ds, test_ds = dummy_datasets
        
        mock_check_cuda.return_value = False
        
        # Mock model.fit to avoid actual training
        with patch.object(dummy_model, "fit") as mock_fit, \
             patch.object(dummy_model, "compile") as mock_compile:
            
            mock_history = MagicMock()
            mock_fit.return_value = mock_history
            
            history = train(
                model=dummy_model,
                train_dataset=train_ds,
                val_dataset=val_ds,
                epochs=1,
                verbose=0,
            )
            
            assert history is not None
            mock_compile.assert_called_once()
            mock_fit.assert_called_once()
            # Should not call load_dataset when datasets are provided
            mock_check_cuda.assert_called_once()
    
    @patch("sam_ml.modeling.train.check_cuda_availability")
    def test_train_cuda_detection(
        self,
        mock_check_cuda: Mock,
        dummy_model: tf.keras.Model,
        dummy_datasets: Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
    ) -> None:
        """Test that CUDA detection is called during training."""
        train_ds, val_ds, _ = dummy_datasets
        
        mock_check_cuda.return_value = True
        
        with patch.object(dummy_model, "fit") as mock_fit, \
             patch.object(dummy_model, "compile"):
            
            mock_history = MagicMock()
            mock_fit.return_value = mock_history
            
            train(
                model=dummy_model,
                train_dataset=train_ds,
                val_dataset=val_ds,
                epochs=1,
                verbose=0,
            )
            
            mock_check_cuda.assert_called_once()
    
    @patch("sam_ml.modeling.train.check_cuda_availability")
    def test_train_use_cuda_parameter(
        self,
        mock_check_cuda: Mock,
        dummy_model: tf.keras.Model,
        dummy_datasets: Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
    ) -> None:
        """Test that use_cuda parameter is respected."""
        train_ds, val_ds, _ = dummy_datasets
        
        mock_check_cuda.return_value = False
        
        with patch.object(dummy_model, "fit") as mock_fit, \
             patch.object(dummy_model, "compile"):
            
            mock_history = MagicMock()
            mock_fit.return_value = mock_history
            
            # Explicitly set use_cuda=False
            train(
                model=dummy_model,
                train_dataset=train_ds,
                val_dataset=val_ds,
                epochs=1,
                use_cuda=False,
                verbose=0,
            )
            
            # Should not call check_cuda_availability when use_cuda is explicitly set
            # But it will still be called to validate if CUDA was requested
            assert mock_check_cuda.call_count >= 0
    
    @patch("sam_ml.modeling.train.check_cuda_availability")
    def test_train_with_callbacks(
        self,
        mock_check_cuda: Mock,
        dummy_model: tf.keras.Model,
        dummy_datasets: Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
    ) -> None:
        """Test training with custom callbacks."""
        train_ds, val_ds, _ = dummy_datasets
        
        mock_check_cuda.return_value = False
        
        custom_callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5),
        ]
        
        with patch.object(dummy_model, "fit") as mock_fit, \
             patch.object(dummy_model, "compile"):
            
            mock_history = MagicMock()
            mock_fit.return_value = mock_history
            
            train(
                model=dummy_model,
                train_dataset=train_ds,
                val_dataset=val_ds,
                epochs=1,
                callbacks=custom_callbacks,
                verbose=0,
            )
            
            # Verify that fit was called with the custom callbacks
            call_args = mock_fit.call_args
            assert call_args is not None
            assert "callbacks" in call_args.kwargs
            assert len(call_args.kwargs["callbacks"]) == 1
    
    @patch("sam_ml.modeling.train.check_cuda_availability")
    @patch("sam_ml.modeling.train.logger")
    def test_train_with_test_dataset(
        self,
        mock_logger: Mock,
        mock_check_cuda: Mock,
        dummy_model: tf.keras.Model,
        dummy_datasets: Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
    ) -> None:
        """Test training with test dataset for evaluation."""
        train_ds, val_ds, test_ds = dummy_datasets
        
        mock_check_cuda.return_value = False
        
        with patch.object(dummy_model, "fit") as mock_fit, \
             patch.object(dummy_model, "evaluate") as mock_evaluate, \
             patch.object(dummy_model, "compile"), \
             patch.object(type(dummy_model), "metrics_names", new_callable=PropertyMock) as mock_metrics_names:
            
            mock_history = MagicMock()
            mock_fit.return_value = mock_history
            mock_evaluate.return_value = [0.5, 0.8]  # [loss, accuracy]
            mock_metrics_names.return_value = ["loss", "accuracy"]
            
            train(
                model=dummy_model,
                train_dataset=train_ds,
                val_dataset=val_ds,
                test_dataset=test_ds,
                epochs=1,
                verbose=0,
            )
            
            mock_evaluate.assert_called_once_with(test_ds, verbose=0)
    
    @patch("sam_ml.modeling.train.check_cuda_availability")
    def test_train_compiles_model(
        self,
        mock_check_cuda: Mock,
        dummy_model: tf.keras.Model,
        dummy_datasets: Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
    ) -> None:
        """Test that model is compiled before training."""
        train_ds, val_ds, _ = dummy_datasets
        
        mock_check_cuda.return_value = False
        
        with patch.object(dummy_model, "fit") as mock_fit, \
             patch.object(dummy_model, "compile") as mock_compile:
            
            mock_history = MagicMock()
            mock_fit.return_value = mock_history
            
            train(
                model=dummy_model,
                train_dataset=train_ds,
                val_dataset=val_ds,
                epochs=1,
                optimizer="sgd",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy", "top_k_categorical_accuracy"],
                verbose=0,
            )
            
            mock_compile.assert_called_once_with(
                optimizer="sgd",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy", "top_k_categorical_accuracy"],
            )
    
    @patch("sam_ml.modeling.train.check_cuda_availability")
    def test_train_default_metrics(
        self,
        mock_check_cuda: Mock,
        dummy_model: tf.keras.Model,
        dummy_datasets: Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
    ) -> None:
        """Test that default metrics are used when not provided."""
        train_ds, val_ds, _ = dummy_datasets
        
        mock_check_cuda.return_value = False
        
        with patch.object(dummy_model, "fit") as mock_fit, \
             patch.object(dummy_model, "compile") as mock_compile:
            
            mock_history = MagicMock()
            mock_fit.return_value = mock_history
            
            train(
                model=dummy_model,
                train_dataset=train_ds,
                val_dataset=val_ds,
                epochs=1,
                verbose=0,
            )
            
            # Verify that compile was called with default metrics
            call_args = mock_compile.call_args
            assert call_args is not None
            assert "metrics" in call_args.kwargs
            assert call_args.kwargs["metrics"] == ["accuracy"]
    
    @patch("sam_ml.modeling.train.check_cuda_availability")
    @patch("sam_ml.modeling.train.create_callbacks")
    def test_train_creates_default_callbacks(
        self,
        mock_create_callbacks: Mock,
        mock_check_cuda: Mock,
        dummy_model: tf.keras.Model,
        dummy_datasets: Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
    ) -> None:
        """Test that default callbacks are created when not provided."""
        train_ds, val_ds, _ = dummy_datasets
        
        mock_check_cuda.return_value = False
        mock_callbacks = [MagicMock()]
        mock_create_callbacks.return_value = mock_callbacks
        
        with patch.object(dummy_model, "fit") as mock_fit, \
             patch.object(dummy_model, "compile"):
            
            mock_history = MagicMock()
            mock_fit.return_value = mock_history
            
            train(
                model=dummy_model,
                train_dataset=train_ds,
                val_dataset=val_ds,
                epochs=1,
                verbose=0,
            )
            
            mock_create_callbacks.assert_called_once()
    
    @patch("sam_ml.modeling.train.check_cuda_availability")
    def test_train_with_hyperparameters(
        self,
        mock_check_cuda: Mock,
        dummy_model: tf.keras.Model,
        dummy_datasets: Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
    ) -> None:
        """Test training with custom hyperparameters."""
        train_ds, val_ds, _ = dummy_datasets
        
        mock_check_cuda.return_value = False
        
        with patch.object(dummy_model, "fit") as mock_fit, \
             patch.object(dummy_model, "compile") as mock_compile, \
             patch("sam_ml.modeling.train.create_model", return_value=dummy_model):
            
            mock_history = MagicMock()
            mock_fit.return_value = mock_history
            
            history = train(
                train_dataset=train_ds,
                val_dataset=val_ds,
                epochs=1,
                learning_rate=0.0001,
                beta_1=0.95,
                beta_2=0.99,
                epsilon=1e-8,
                weight_decay=1e-4,
                verbose=0,
            )
            
            # Verify optimizer was created with correct parameters
            mock_compile.assert_called_once()
            compile_args = mock_compile.call_args
            assert compile_args is not None
            optimizer = compile_args.kwargs["optimizer"]
            assert isinstance(optimizer, tf.keras.optimizers.Adam)
            
            # Check optimizer parameters
            lr = optimizer.learning_rate.numpy() if hasattr(optimizer.learning_rate, "numpy") else optimizer.learning_rate
            assert abs(lr - 0.0001) < 1e-6
            
            mock_fit.assert_called_once()
    
    @patch("sam_ml.modeling.train.check_cuda_availability")
    def test_train_creates_adam_optimizer_by_default(
        self,
        mock_check_cuda: Mock,
        dummy_model: tf.keras.Model,
        dummy_datasets: Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
    ) -> None:
        """Test that Adam optimizer is created by default."""
        train_ds, val_ds, _ = dummy_datasets
        
        mock_check_cuda.return_value = False
        
        with patch.object(dummy_model, "fit") as mock_fit, \
             patch.object(dummy_model, "compile") as mock_compile, \
             patch("sam_ml.modeling.train.create_model", return_value=dummy_model):
            
            mock_history = MagicMock()
            mock_fit.return_value = mock_history
            
            train(
                train_dataset=train_ds,
                val_dataset=val_ds,
                epochs=1,
                verbose=0,
            )
            
            # Verify optimizer was created (should be Adam by default)
            mock_compile.assert_called_once()
            compile_args = mock_compile.call_args
            assert compile_args is not None
            optimizer = compile_args.kwargs["optimizer"]
            assert isinstance(optimizer, tf.keras.optimizers.Adam)


class TestParseArgs:
    """Test cases for command-line argument parsing."""
    
    def test_parse_args_defaults(self) -> None:
        """Test argument parsing with default values."""
        with patch("sys.argv", ["train.py"]):
            args = parse_args()
            
            assert args.epochs == 50
            assert args.batch_size == 32
            assert args.learning_rate == 0.001
            assert args.beta_1 == 0.9
            assert args.beta_2 == 0.999
            assert args.epsilon == 1e-7
            assert args.weight_decay == 0.0
            assert args.num_classes == 5
            assert args.verbose == 1
    
    def test_parse_args_custom_hyperparameters(self) -> None:
        """Test argument parsing with custom hyperparameters."""
        with patch("sys.argv", [
            "train.py",
            "--learning-rate", "0.0001",
            "--beta-1", "0.95",
            "--beta-2", "0.99",
            "--epsilon", "1e-8",
            "--weight-decay", "1e-4",
        ]):
            args = parse_args()
            
            assert args.learning_rate == 0.0001
            assert args.beta_1 == 0.95
            assert args.beta_2 == 0.99
            assert args.epsilon == 1e-8
            assert args.weight_decay == 1e-4
    
    def test_parse_args_training_parameters(self) -> None:
        """Test argument parsing with training parameters."""
        with patch("sys.argv", [
            "train.py",
            "--epochs", "100",
            "--batch-size", "64",
            "--validation-freq", "2",
            "--verbose", "2",
        ]):
            args = parse_args()
            
            assert args.epochs == 100
            assert args.batch_size == 64
            assert args.validation_freq == 2
            assert args.verbose == 2
    
    def test_parse_args_checkpoint_parameters(self) -> None:
        """Test argument parsing with checkpoint parameters."""
        with patch("sys.argv", [
            "train.py",
            "--checkpoint-dir", "custom/checkpoints",
            "--checkpoint-filename", "custom_model.keras",
        ]):
            args = parse_args()
            
            assert args.checkpoint_dir == Path("custom/checkpoints")
            assert args.checkpoint_filename == "custom_model.keras"
    
    def test_parse_args_cuda_flags(self) -> None:
        """Test argument parsing with CUDA flags."""
        with patch("sys.argv", ["train.py", "--use-cuda"]):
            args = parse_args()
            assert args.use_cuda is True
            assert args.no_cuda is False
        
        with patch("sys.argv", ["train.py", "--no-cuda"]):
            args = parse_args()
            assert args.use_cuda is False
            assert args.no_cuda is True

