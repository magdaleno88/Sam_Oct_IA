"""Centralized configuration for the SAM-AI project using Pydantic."""

from pathlib import Path
from typing import Literal, Tuple

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseSettings):
    """Configuration for model hyperparameters."""

    model_config = SettingsConfigDict(
        env_prefix="SAM_MODEL_",
        case_sensitive=False,
        extra="ignore",
    )

    # Model architecture
    num_classes: int = Field(
        default=5,
        description="Number of output classes for classification",
        gt=0,
    )
    input_shape: Tuple[int, int, int] = Field(
        default=(3, 512, 512),
        description="Input image shape (channels, height, width)",
    )

    # Optimizer settings
    learning_rate: float = Field(
        default=1e-4,
        description="Learning rate for optimizer",
        gt=0.0,
    )
    optimizer: Literal["adam", "sgd"] = Field(
        default="adam",
        description="Optimizer name",
    )
    weight_decay: float = Field(
        default=1e-4,
        description="Weight decay (L2 regularization) coefficient",
        ge=0.0,
    )

    @field_validator("input_shape")
    @classmethod
    def validate_input_shape(cls, v: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Validate input shape has 3 dimensions."""
        if len(v) != 3:
            raise ValueError("input_shape must have 3 dimensions (channels, height, width)")
        if any(dim <= 0 for dim in v):
            raise ValueError("All dimensions in input_shape must be positive")
        return v


class TrainingConfig(BaseSettings):
    """Configuration for training hyperparameters."""

    model_config = SettingsConfigDict(
        env_prefix="SAM_TRAINING_",
        case_sensitive=False,
        extra="ignore",
    )

    # Training hyperparameters
    batch_size: int = Field(
        default=32,
        description="Batch size for training",
        gt=0,
    )
    num_epochs: int = Field(
        default=50,
        description="Number of training epochs",
        gt=0,
    )
    patience: int = Field(
        default=10,
        description="Early stopping patience",
        gt=0,
    )

    # Data paths
    data_dir: Path = Field(
        default=Path("data/processed/ddr2019"),
        description="Directory containing processed dataset",
    )
    output_dir: Path = Field(
        default=Path("outputs"),
        description="Directory to save model checkpoints and logs",
    )

    # Hardware
    gpus: int | None = Field(
        default=None,
        description="Number of GPUs to use (None for CPU)",
        ge=0,
    )

    @field_validator("data_dir", "output_dir", mode="before")
    @classmethod
    def validate_path(cls, v: str | Path) -> Path:
        """Convert string paths to Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v


class SchedulerConfig(BaseSettings):
    """Configuration for learning rate scheduler."""

    model_config = SettingsConfigDict(
        env_prefix="SAM_SCHEDULER_",
        case_sensitive=False,
        extra="ignore",
    )

    factor: float = Field(
        default=0.5,
        description="Factor by which learning rate is reduced",
        gt=0.0,
        lt=1.0,
    )
    patience: int = Field(
        default=5,
        description="Number of epochs to wait before reducing learning rate",
        gt=0,
    )
    mode: Literal["min", "max"] = Field(
        default="min",
        description="Mode for ReduceLROnPlateau",
    )
    monitor: str = Field(
        default="val_loss",
        description="Metric to monitor for learning rate reduction",
    )
    verbose: bool = Field(
        default=True,
        description="Whether to print learning rate updates",
    )


class PreprocessingConfig(BaseSettings):
    """Configuration for preprocessing parameters."""

    model_config = SettingsConfigDict(
        env_prefix="SAM_PREPROCESSING_",
        case_sensitive=False,
        extra="ignore",
    )

    # Image processing
    min_size: int = Field(
        default=512,
        description="Minimum image size (width and height) required to process",
        gt=0,
    )
    target_size: Tuple[int, int] = Field(
        default=(512, 512),
        description="Target size (width, height) for resizing",
    )

    # DDR2019 dataset paths
    ddr2019_raw_img_dir: Path = Field(
        default=Path("data/raw/ddr2019/DR_grading/DR_grading"),
        description="DDR2019 raw images directory",
    )
    ddr2019_raw_csv_path: Path = Field(
        default=Path("data/raw/ddr2019/DR_grading.csv"),
        description="DDR2019 raw CSV labels file",
    )
    ddr2019_processed_dir: Path = Field(
        default=Path("data/processed/ddr2019"),
        description="DDR2019 processed output directory",
    )

    @field_validator("target_size")
    @classmethod
    def validate_target_size(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        """Validate target size has 2 dimensions."""
        if len(v) != 2:
            raise ValueError("target_size must have 2 dimensions (width, height)")
        if any(dim <= 0 for dim in v):
            raise ValueError("All dimensions in target_size must be positive")
        return v

    @field_validator(
        "ddr2019_raw_img_dir",
        "ddr2019_raw_csv_path",
        "ddr2019_processed_dir",
        mode="before",
    )
    @classmethod
    def validate_path(cls, v: str | Path) -> Path:
        """Convert string paths to Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v


class Config(BaseSettings):
    """Main configuration class combining all sub-configurations."""

    model_config = SettingsConfigDict(
        env_prefix="SAM_",
        case_sensitive=False,
        extra="ignore",
    )

    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)

    @classmethod
    def get_default(cls) -> "Config":
        """Get default configuration instance."""
        return cls()


# Global configuration instance
_config: Config | None = None


def get_config() -> Config:
    """
    Get the global configuration instance.
    
    Returns:
        Configuration instance
    """
    global _config
    if _config is None:
        _config = Config.get_default()
    return _config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None


# Convenience accessors for common configurations
def get_model_config() -> ModelConfig:
    """Get model configuration."""
    return get_config().model


def get_training_config() -> TrainingConfig:
    """Get training configuration."""
    return get_config().training


def get_scheduler_config() -> SchedulerConfig:
    """Get scheduler configuration."""
    return get_config().scheduler


def get_preprocessing_config() -> PreprocessingConfig:
    """Get preprocessing configuration."""
    return get_config().preprocessing
