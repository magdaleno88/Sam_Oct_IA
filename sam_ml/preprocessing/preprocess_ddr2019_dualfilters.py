"""Preprocessor for DDR2019 dual-filter export (CLAHE + CECED)."""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from sam_ml.config import get_preprocessing_config
from sam_ml.preprocessing.base import (
    BasePreprocessor,
    BasePreprocessorConfig,
    register_preprocessor,
    run_core_loop,
)
from sam_ml.preprocessing.middleware import MiddlewareContext, get_middleware


class Ddr2019DualFiltersPreprocessorConfig(BasePreprocessorConfig):
    """Configuration for DDR2019 dual-filter export."""

    clahe_size: tuple[int, int] = (299, 299)
    ceced_size: tuple[int, int] = (224, 224)
    clahe_subdir: str = "images_clahe"
    ceced_subdir: str = "images_ceced"
    labels_filename: str = "labels_dual.csv"


def convert_labels_dual_csv(
    raw_csv_path: str,
    processed_dir: str,
    processed_filenames: set[str],
    clahe_subdir: str = "images_clahe",
    ceced_subdir: str = "images_ceced",
    output_filename: str = "labels_dual.csv",
) -> Path:
    """Create explicit dual-path label mapping CSV for synchronized folders."""
    raw_path = Path(raw_csv_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw CSV file not found: {raw_csv_path}")

    df = pd.read_csv(raw_path)
    required_columns = {"id_code", "diagnosis"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df.rename(columns={"id_code": "filename", "diagnosis": "label"})
    df["filename"] = df["filename"].astype(str)
    df = df[df["filename"].isin(processed_filenames)].copy()

    output_dir = Path(processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clahe_dir = output_dir / clahe_subdir
    ceced_dir = output_dir / ceced_subdir

    df["clahe_path"] = df["filename"].map(lambda x: f"{clahe_subdir}/{x}")
    df["ceced_path"] = df["filename"].map(lambda x: f"{ceced_subdir}/{x}")

    def _exists_both(filename: str) -> bool:
        return (clahe_dir / filename).exists() and (ceced_dir / filename).exists()

    original_count = len(df)
    exists_mask = df["filename"].map(_exists_both).astype(bool)
    df = df[exists_mask].copy()
    removed = original_count - len(df)
    if removed > 0:
        print(f"Info: Removed {removed} labels without synchronized CLAHE/CECED outputs")

    out_df = df.reindex(columns=["clahe_path", "ceced_path", "label"]).copy()
    out_path = output_dir / output_filename
    out_df.to_csv(out_path, index=False)
    return out_path


@register_preprocessor("ddr2019_dualfilters")
class Ddr2019DualFiltersPreprocessor(BasePreprocessor):
    """DDR2019 preprocessor that exports synchronized CLAHE and CECED datasets."""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        config = get_preprocessing_config()
        parser.add_argument(
            "--raw-img-dir",
            type=str,
            default=str(config.ddr2019_raw_img_dir),
            help="Path to raw images directory",
        )
        parser.add_argument(
            "--raw-csv-path",
            type=str,
            default=str(config.ddr2019_raw_csv_path),
            help="Path to raw CSV file",
        )
        parser.add_argument(
            "--processed-dir",
            type=str,
            default=None,
            help="Full path to processed output directory (overrides --output-name)",
        )
        parser.add_argument(
            "--output-name",
            type=str,
            default=None,
            metavar="FOLDER",
            help="Output folder name under data/processed. Ignored if --processed-dir is set.",
        )
        parser.add_argument(
            "--min-size",
            type=int,
            default=config.min_size,
            help=f"Minimum size (width and height) to process. Default: {config.min_size}",
        )
        parser.add_argument(
            "--clahe-size",
            type=int,
            nargs=2,
            metavar=("WIDTH", "HEIGHT"),
            default=[299, 299],
            help="Target size for CLAHE images. Default: 299 299",
        )
        parser.add_argument(
            "--ceced-size",
            type=int,
            nargs=2,
            metavar=("WIDTH", "HEIGHT"),
            default=[224, 224],
            help="Target size for CECED images. Default: 224 224",
        )
        parser.add_argument(
            "--clahe-subdir",
            type=str,
            default="images_clahe",
            help="Output subdirectory for CLAHE images. Default: images_clahe",
        )
        parser.add_argument(
            "--ceced-subdir",
            type=str,
            default="images_ceced",
            help="Output subdirectory for CECED images. Default: images_ceced",
        )
        parser.add_argument(
            "--labels-filename",
            type=str,
            default="labels_dual.csv",
            help="Output labels filename. Default: labels_dual.csv",
        )

    @classmethod
    def build_config_from_args(cls, parsed: argparse.Namespace) -> BasePreprocessorConfig:
        config = get_preprocessing_config()
        processed_dir = getattr(parsed, "processed_dir", None)
        if not processed_dir:
            if getattr(parsed, "output_name", None):
                base_processed = config.ddr2019_processed_dir.parent
                processed_dir = str(Path(base_processed) / parsed.output_name)
            else:
                # Default: data/processed/<preprocessor keyword>
                base_processed = config.ddr2019_processed_dir.parent
                processed_dir = str(Path(base_processed) / "ddr2019_dualfilters")

        return Ddr2019DualFiltersPreprocessorConfig(
            raw_img_dir=getattr(parsed, "raw_img_dir", str(config.ddr2019_raw_img_dir)),
            raw_csv_path=getattr(parsed, "raw_csv_path", str(config.ddr2019_raw_csv_path)),
            processed_dir=processed_dir,
            min_size=getattr(parsed, "min_size", config.min_size),
            target_size=(299, 299),
            middleware="dual_filters_multisize",
            output_subdir=config.default_output_subdir,
            clahe_size=tuple(getattr(parsed, "clahe_size", [299, 299])),
            ceced_size=tuple(getattr(parsed, "ceced_size", [224, 224])),
            clahe_subdir=getattr(parsed, "clahe_subdir", "images_clahe"),
            ceced_subdir=getattr(parsed, "ceced_subdir", "images_ceced"),
            labels_filename=getattr(parsed, "labels_filename", "labels_dual.csv"),
        )

    def run(self, config: BasePreprocessorConfig) -> dict[str, Any]:
        if not isinstance(config, Ddr2019DualFiltersPreprocessorConfig):
            raise TypeError("Expected Ddr2019DualFiltersPreprocessorConfig")

        context: MiddlewareContext = {
            "min_size": config.min_size,
            "clahe_target_size": config.clahe_size,
            "ceced_target_size": config.ceced_size,
        }
        middleware = get_middleware(
            config.middleware,
            min_size=config.min_size,
            clahe_target_size=config.clahe_size,
            ceced_target_size=config.ceced_size,
            clahe_output_key=config.clahe_subdir,
            ceced_output_key=config.ceced_subdir,
        )
        images_processed, processed_filenames = run_core_loop(
            raw_img_dir=config.raw_img_dir,
            processed_dir=config.processed_dir,
            middleware=middleware,
            context=context,
        )

        labels_path = convert_labels_dual_csv(
            raw_csv_path=config.raw_csv_path,
            processed_dir=config.processed_dir,
            processed_filenames=processed_filenames,
            clahe_subdir=config.clahe_subdir,
            ceced_subdir=config.ceced_subdir,
            output_filename=config.labels_filename,
        )
        return {
            "images_processed": images_processed,
            "labels_path": labels_path,
            "processed_filenames": processed_filenames,
        }
