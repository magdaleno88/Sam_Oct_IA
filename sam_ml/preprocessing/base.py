"""Abstract base class and registry for dataset preprocessors."""

import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from sam_ml.config import get_preprocessing_config
from sam_ml.preprocessing.middleware import (
    BaseMiddleware,
    MiddlewareContext,
    get_middleware,
)
from sam_ml.preprocessing.utils import load_image_bgr, save_image_bgr
from tqdm import tqdm


def _defaults_from_config() -> dict[str, Any]:
    """Get default values from PreprocessingConfig for preprocessor config models."""
    c = get_preprocessing_config()
    return {
        "raw_img_dir": str(c.ddr2019_raw_img_dir),
        "raw_csv_path": str(c.ddr2019_raw_csv_path),
        "processed_dir": str(c.ddr2019_processed_dir),
        "min_size": c.min_size,
        "target_size": tuple(c.target_size),
        "middleware": c.default_middleware,
        "output_subdir": c.default_output_subdir,
    }


class BasePreprocessorConfig(BaseModel):
    """Common configuration for preprocessors; defaults from config.py."""

    raw_img_dir: str = Field(default_factory=lambda: _defaults_from_config()["raw_img_dir"])
    raw_csv_path: str = Field(default_factory=lambda: _defaults_from_config()["raw_csv_path"])
    processed_dir: str = Field(default_factory=lambda: _defaults_from_config()["processed_dir"])
    min_size: int = Field(default_factory=lambda: _defaults_from_config()["min_size"], gt=0)
    target_size: tuple[int, int] = Field(
        default_factory=lambda: _defaults_from_config()["target_size"]
    )
    middleware: str = Field(default_factory=lambda: _defaults_from_config()["middleware"])
    output_subdir: str = Field(default_factory=lambda: _defaults_from_config()["output_subdir"])

    model_config = {"extra": "forbid"}


# Registry: keyword -> preprocessor class
PREPROCESSOR_REGISTRY: dict[str, type["BasePreprocessor"]] = {}


def register_preprocessor(key: str) -> Callable[[type["BasePreprocessor"]], type["BasePreprocessor"]]:
    """Decorator to register a preprocessor class in the registry."""

    def decorator(cls: type["BasePreprocessor"]) -> type["BasePreprocessor"]:
        if key in PREPROCESSOR_REGISTRY:
            raise ValueError(f"Preprocessor key '{key}' is already registered")
        PREPROCESSOR_REGISTRY[key] = cls
        return cls

    return decorator


def get_preprocessor(key: str) -> type["BasePreprocessor"]:
    """Get a preprocessor class from the registry."""
    if key not in PREPROCESSOR_REGISTRY:
        available = ", ".join(PREPROCESSOR_REGISTRY.keys()) if PREPROCESSOR_REGISTRY else "none"
        raise KeyError(
            f"Preprocessor '{key}' not found in registry. Available: {available}"
        )
    return PREPROCESSOR_REGISTRY[key]


def list_preprocessors() -> list[str]:
    """List all registered preprocessor keys."""
    return list(PREPROCESSOR_REGISTRY.keys())


def run_core_loop(
    raw_img_dir: str,
    processed_dir: str,
    middleware: BaseMiddleware,
    context: MiddlewareContext,
) -> tuple[int, set[str]]:
    """Shared core loop: load each JPG as BGR, run middleware.process(), write outputs.

    Returns (processed_count, processed_filenames).
    """
    raw_path = Path(raw_img_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw images directory not found: {raw_img_dir}")
    image_files = list(raw_path.glob("*.jpg"))
    if not image_files:
        raise ValueError(f"No JPG files found in {raw_img_dir}")

    processed_dir_path = Path(processed_dir)
    processed_count = 0
    processed_filenames: set[str] = set()
    skipped_small_count = 0
    min_size = context.get("min_size", 512)

    for img_file in tqdm(image_files, desc="Processing images"):
        try:
            img_bgr = load_image_bgr(img_file)
            if img_bgr is None:
                print(f"Warning: Could not read {img_file.name}")
                continue
            results = middleware.process(img_bgr, img_file.name, context)
            if not results:
                skipped_small_count += 1
                continue
            for out_key, out_img in results:
                out_dir = processed_dir_path / out_key
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / img_file.name
                if not save_image_bgr(out_path, out_img):
                    print(f"Warning: Could not write {out_path}")
            processed_count += 1
            processed_filenames.add(img_file.name)
        except Exception as e:
            print(f"Warning: Failed to process {img_file.name}: {e}")
            continue

    if skipped_small_count > 0:
        print(
            f"Info: Skipped {skipped_small_count} images smaller than {min_size}x{min_size}"
        )
    return processed_count, processed_filenames


class BasePreprocessor(ABC):
    """Abstract base for dataset preprocessors.

    Registered by keyword (first CLI arg). Subclasses define config model,
    add_arguments(), and run() using the shared core loop and dataset-specific
    label conversion.
    """

    @classmethod
    @abstractmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add common and preprocessor-specific CLI arguments."""
        ...

    @abstractmethod
    def run(self, config: BasePreprocessorConfig) -> dict[str, Any]:
        """Run preprocessing; returns dict with images_processed, labels_path, processed_filenames."""
        ...
