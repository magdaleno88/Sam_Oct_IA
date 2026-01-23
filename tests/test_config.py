"""Unit tests for configuration system."""

from pathlib import Path

import pytest

from sam_ml.config import (
    Config,
    ModelConfig,
    PreprocessingConfig,
    SchedulerConfig,
    TrainingConfig,
    get_config,
    get_model_config,
    get_preprocessing_config,
    get_scheduler_config,
    get_training_config,
    reset_config,
)


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_values(self) -> None:
        """Test default model configuration values."""
        config = ModelConfig()
        
        assert config.num_classes == 5
        assert config.learning_rate == 1e-4
        assert config.optimizer == "adam"
        assert config.weight_decay == 1e-4
        assert config.input_shape == (3, 512, 512)

    def test_custom_values(self) -> None:
        """Test setting custom model configuration values."""
        config = ModelConfig(
            num_classes=3,
            learning_rate=1e-3,
            optimizer="sgd",
            weight_decay=0.01,
            input_shape=(3, 256, 256),
        )
        
        assert config.num_classes == 3
        assert config.learning_rate == 1e-3
        assert config.optimizer == "sgd"
        assert config.weight_decay == 0.01
        assert config.input_shape == (3, 256, 256)

    def test_validation_num_classes(self) -> None:
        """Test validation for num_classes."""
        with pytest.raises(Exception):  # Pydantic validation error
            ModelConfig(num_classes=0)
        
        with pytest.raises(Exception):
            ModelConfig(num_classes=-1)

    def test_validation_learning_rate(self) -> None:
        """Test validation for learning_rate."""
        with pytest.raises(Exception):
            ModelConfig(learning_rate=0.0)
        
        with pytest.raises(Exception):
            ModelConfig(learning_rate=-1.0)

    def test_validation_input_shape(self) -> None:
        """Test validation for input_shape."""
        # Pydantic validates tuple length at type level (raises ValidationError, not ValueError)
        with pytest.raises(Exception):  # Pydantic ValidationError
            ModelConfig(input_shape=(3, 512))  # type: ignore
        
        # Custom validator checks for positive values (runs after type validation)
        with pytest.raises(ValueError, match="must be positive"):
            ModelConfig(input_shape=(3, 0, 512))  # type: ignore


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self) -> None:
        """Test default training configuration values."""
        config = TrainingConfig()
        
        assert config.batch_size == 32
        assert config.num_epochs == 50
        assert config.patience == 10
        assert config.data_dir == Path("data/processed/ddr2019")
        assert config.output_dir == Path("outputs")
        assert config.gpus is None

    def test_custom_values(self) -> None:
        """Test setting custom training configuration values."""
        config = TrainingConfig(
            batch_size=64,
            num_epochs=100,
            patience=15,
            data_dir="custom/data",
            output_dir="custom/outputs",
            gpus=2,
        )
        
        assert config.batch_size == 64
        assert config.num_epochs == 100
        assert config.patience == 15
        assert config.data_dir == Path("custom/data")
        assert config.output_dir == Path("custom/outputs")
        assert config.gpus == 2

    def test_path_conversion(self) -> None:
        """Test that string paths are converted to Path objects."""
        config = TrainingConfig(data_dir="test/path", output_dir="test/output")
        
        assert isinstance(config.data_dir, Path)
        assert isinstance(config.output_dir, Path)
        assert config.data_dir == Path("test/path")
        assert config.output_dir == Path("test/output")


class TestSchedulerConfig:
    """Tests for SchedulerConfig."""

    def test_default_values(self) -> None:
        """Test default scheduler configuration values."""
        config = SchedulerConfig()
        
        assert config.factor == 0.5
        assert config.patience == 5
        assert config.mode == "min"
        assert config.monitor == "val_loss"
        assert config.verbose is True

    def test_custom_values(self) -> None:
        """Test setting custom scheduler configuration values."""
        config = SchedulerConfig(
            factor=0.1,
            patience=10,
            mode="max",
            monitor="val_accuracy",
            verbose=False,
        )
        
        assert config.factor == 0.1
        assert config.patience == 10
        assert config.mode == "max"
        assert config.monitor == "val_accuracy"
        assert config.verbose is False

    def test_validation_factor(self) -> None:
        """Test validation for factor."""
        with pytest.raises(Exception):
            SchedulerConfig(factor=0.0)
        
        with pytest.raises(Exception):
            SchedulerConfig(factor=1.0)


class TestPreprocessingConfig:
    """Tests for PreprocessingConfig."""

    def test_default_values(self) -> None:
        """Test default preprocessing configuration values."""
        config = PreprocessingConfig()
        
        assert config.min_size == 512
        assert config.target_size == (512, 512)
        assert config.ddr2019_raw_img_dir == Path("data/raw/ddr2019/DR_grading/DR_grading")
        assert config.ddr2019_raw_csv_path == Path("data/raw/ddr2019/DR_grading.csv")
        assert config.ddr2019_processed_dir == Path("data/processed/ddr2019")

    def test_custom_values(self) -> None:
        """Test setting custom preprocessing configuration values."""
        config = PreprocessingConfig(
            min_size=256,
            target_size=(256, 256),
            ddr2019_raw_img_dir="custom/raw",
            ddr2019_raw_csv_path="custom/labels.csv",
            ddr2019_processed_dir="custom/processed",
        )
        
        assert config.min_size == 256
        assert config.target_size == (256, 256)
        assert config.ddr2019_raw_img_dir == Path("custom/raw")
        assert config.ddr2019_raw_csv_path == Path("custom/labels.csv")
        assert config.ddr2019_processed_dir == Path("custom/processed")

    def test_validation_target_size(self) -> None:
        """Test validation for target_size."""
        # Pydantic validates tuple length at type level
        with pytest.raises(Exception):  # Pydantic validation error
            PreprocessingConfig(target_size=(512,))  # type: ignore
        
        # Custom validator checks for positive values
        with pytest.raises(ValueError, match="must be positive"):
            PreprocessingConfig(target_size=(0, 512))  # type: ignore


class TestConfig:
    """Tests for main Config class."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = Config.get_default()
        
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.scheduler, SchedulerConfig)
        assert isinstance(config.preprocessing, PreprocessingConfig)

    def test_nested_config_access(self) -> None:
        """Test accessing nested configuration."""
        config = Config.get_default()
        
        assert config.model.num_classes == 5
        assert config.training.batch_size == 32
        assert config.scheduler.factor == 0.5
        assert config.preprocessing.min_size == 512


class TestConfigAccessors:
    """Tests for config accessor functions."""

    def test_get_config(self) -> None:
        """Test get_config function."""
        config = get_config()
        
        assert isinstance(config, Config)
        # Should return the same instance on subsequent calls
        assert get_config() is config

    def test_reset_config(self) -> None:
        """Test reset_config function."""
        config1 = get_config()
        reset_config()
        config2 = get_config()
        
        # Should be different instances after reset
        assert config1 is not config2

    def test_get_model_config(self) -> None:
        """Test get_model_config function."""
        model_config = get_model_config()
        
        assert isinstance(model_config, ModelConfig)
        assert model_config.num_classes == 5

    def test_get_training_config(self) -> None:
        """Test get_training_config function."""
        training_config = get_training_config()
        
        assert isinstance(training_config, TrainingConfig)
        assert training_config.batch_size == 32

    def test_get_scheduler_config(self) -> None:
        """Test get_scheduler_config function."""
        scheduler_config = get_scheduler_config()
        
        assert isinstance(scheduler_config, SchedulerConfig)
        assert scheduler_config.factor == 0.5

    def test_get_preprocessing_config(self) -> None:
        """Test get_preprocessing_config function."""
        preprocessing_config = get_preprocessing_config()
        
        assert isinstance(preprocessing_config, PreprocessingConfig)
        assert preprocessing_config.min_size == 512
