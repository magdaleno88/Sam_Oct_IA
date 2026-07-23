"""Typed YAML configuration for OCT experiments."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from sam_ml.oct.constants import CLASS_NAMES


class OCTDataConfig(BaseModel):
    root: Path = Path("data/raw")
    manifest_dir: Path = Path("data/manifests")
    image_size: int = Field(224, gt=0)
    classes: tuple[str, ...] = CLASS_NAMES
    patient_level_split: bool = True
    allow_image_level_split: bool = False
    val_fraction: float = Field(0.1, gt=0, lt=1)
    test_fraction: float = Field(0.15, gt=0, lt=1)
    seed: int = 42
    exclusions_file: Path = Path("data/manifests/excluded_images.csv")

    @field_validator("classes")
    @classmethod
    def fixed_classes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(value) != CLASS_NAMES:
            raise ValueError(f"OCT classes and order must be {CLASS_NAMES}")
        return value


class OCTPreprocessingConfig(BaseModel):
    """Conservative, class-agnostic OCT cleanup settings."""

    enabled: bool = True
    target_size: int = Field(224, gt=0)
    output_root: Path = Path("data/processed/oct")
    quality_control_dir: Path = Path("reports/oct_preprocessing")
    white_threshold: int = Field(250, ge=245, le=255)
    min_artifact_area: int = Field(64, gt=0)
    min_artifact_area_fraction: float = Field(0.0005, ge=0, lt=1)
    min_near_white_fraction: float = Field(0.9, ge=0.5, le=1)
    border_width: int = Field(2, gt=0)
    corner_margin_fraction: float = Field(0.2, gt=0, le=0.5)
    min_corner_margin_fraction: float = Field(0.5, ge=0, le=1)
    mask_dilation: int = Field(2, ge=0, le=10)
    background_top_fraction: float = Field(0.35, gt=0, le=0.75)
    background_patch_size: int = Field(32, gt=3)
    background_max_mean: float = Field(100.0, ge=0, le=255)
    background_min_std: float = Field(1.5, ge=0)
    fill_inpaint_weight: float = Field(0.35, ge=0, le=1)
    crop_empty_margins: bool = True
    max_crop_fraction_per_side: float = Field(0.08, ge=0, le=0.25)
    max_aspect_ratio: float = Field(2.0, gt=1)
    extreme_aspect_mode: Literal["letterbox", "overlapping_windows", "macular_center"] = "letterbox"
    overlap_fraction: float = Field(0.25, ge=0, lt=1)
    percentile_normalization: bool = False
    lower_percentile: float = Field(1.0, ge=0, lt=50)
    upper_percentile: float = Field(99.0, gt=50, le=100)
    clahe: bool = False
    clahe_clip_limit: float = Field(1.5, gt=0, le=4)
    light_denoise: Literal["none", "median", "bilateral"] = "none"
    orientation_correction: bool = False
    seed: int = 42
    audit_sample_size: int = Field(32, ge=0)


class DatasetSamplingConfig(BaseModel):
    percentage: float = Field(10.0, gt=0, le=100)
    unit: Literal["auto", "patient", "image"] = "auto"
    mode: Literal["copy", "hardlink", "symlink", "manifest"] = "copy"
    preserve_splits: bool = True
    minimum_per_class: int = Field(1, ge=0)


class DatasetBalancingConfig(BaseModel):
    enabled: bool = False
    strategy: Literal["moderate"] = "moderate"
    mode: Literal["physical", "sampler", "class-weights"] = "class-weights"
    splits: tuple[str, ...] = ("train",)
    max_ratio: float = Field(2.0, ge=1)
    max_undersample_fraction: float = Field(0.30, ge=0, lt=1)
    max_oversample_factor: float = Field(1.50, ge=1)
    min_class_weight: float = Field(0.5, gt=0)
    max_class_weight: float = Field(2.0, gt=0)
    sampling_unit: Literal["auto", "patient", "image"] = "auto"


class DatasetManagementConfig(BaseModel):
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    seed: int = 42
    reports_dir: Path = Path("reports/dataset_management")
    sampling: DatasetSamplingConfig = Field(default_factory=DatasetSamplingConfig)
    balancing: DatasetBalancingConfig = Field(default_factory=DatasetBalancingConfig)

class OCTModelConfig(BaseModel):
    name: Literal["baseline_resnet50", "improved_resnet50"] = "improved_resnet50"
    pretrained: bool = True
    num_classes: int = Field(4, gt=1)
    dropout: float = Field(0.2, ge=0, lt=1)
    replace_stride_with_dilation: tuple[bool, bool, bool] = (False, True, True)
    freeze_backbone: bool = False


class OCTTrainingConfig(BaseModel):
    seed: int = 42
    learning_rate: float = Field(1e-5, gt=0)
    weight_decay: float = Field(1e-4, ge=0)
    batch_size: int = Field(16, gt=0)
    effective_batch_size: int = Field(128, gt=0)
    max_steps: int = Field(10_000, gt=0)
    max_epochs: int = Field(100, gt=0)
    early_stopping_patience: int = Field(10, gt=0)
    mixed_precision: bool = True
    ensemble_size: int = Field(4, gt=0)
    balance_mode: Literal["none", "class_weights", "weighted_sampler", "focal"] = "none"
    focal_gamma: float = Field(2.0, gt=0)
    num_workers: int = Field(0, ge=0)
    monitor: Literal["val_loss", "val_macro_auc"] = "val_loss"


class OCTExplainabilityConfig(BaseModel):
    occlusion_window: int = Field(28, gt=0)
    occlusion_stride: int = Field(7, gt=0)
    occlusion_value: float = 0.0
    gradcam: bool = True
    high_confidence: float = Field(0.9, ge=0, le=1)
    borderline_min: float = Field(0.25, ge=0, le=1)
    borderline_max: float = Field(0.6, ge=0, le=1)
    margin_threshold: float = Field(0.1, ge=0, le=1)


class OCTConfig(BaseModel):
    data: OCTDataConfig = Field(default_factory=OCTDataConfig)
    preprocessing: OCTPreprocessingConfig = Field(default_factory=OCTPreprocessingConfig)
    dataset_management: DatasetManagementConfig = Field(default_factory=DatasetManagementConfig)
    model: OCTModelConfig = Field(default_factory=OCTModelConfig)
    training: OCTTrainingConfig = Field(default_factory=OCTTrainingConfig)
    explainability: OCTExplainabilityConfig = Field(default_factory=OCTExplainabilityConfig)


def load_config(path: str | Path) -> OCTConfig:
    """Load and validate an OCT YAML configuration."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return OCTConfig.model_validate(raw)
