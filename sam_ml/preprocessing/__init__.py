"""Preprocessing module with polymorphic dataset processors."""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Import preprocessor modules so they register with PREPROCESSOR_REGISTRY
from sam_ml.preprocessing import preprocess_ddr2019  # noqa: F401

from sam_ml.config import get_preprocessing_config
from sam_ml.preprocessing.base import (
    BasePreprocessorConfig,
    get_preprocessor,
    list_preprocessors,
)


def _build_config_from_args(parsed: argparse.Namespace, keyword: str) -> BasePreprocessorConfig:
    """Build preprocessor config from parsed CLI args; resolve processed_dir from output_name."""
    config = get_preprocessing_config()
    processed_dir = getattr(parsed, "processed_dir", None)
    if not processed_dir:
        if getattr(parsed, "output_name", None):
            base_processed = config.ddr2019_processed_dir.parent
            processed_dir = str(Path(base_processed) / parsed.output_name)
        else:
            processed_dir = str(config.ddr2019_processed_dir)

    return BasePreprocessorConfig(
        raw_img_dir=getattr(parsed, "raw_img_dir", str(config.ddr2019_raw_img_dir)),
        raw_csv_path=getattr(parsed, "raw_csv_path", str(config.ddr2019_raw_csv_path)),
        processed_dir=processed_dir,
        min_size=getattr(parsed, "min_size", config.min_size),
        target_size=tuple(getattr(parsed, "target_size", list(config.target_size))),
        middleware=getattr(parsed, "middleware", config.default_middleware),
        output_subdir=config.default_output_subdir,
    )


def main(args: Optional[list[str]] = None) -> int:
    """Main entry point for preprocessing scripts.

    Two-phase parse: first argument is the preprocessor keyword (e.g. ddr2019);
    the selected preprocessor adds its CLI arguments, then we parse, build config,
    and run preprocessor.run(config).
    """
    if args is None:
        args = sys.argv[1:]

    # If --help/-h is first, prepend default keyword so we show full help
    if args and args[0] in ("--help", "-h"):
        preprocessors = list_preprocessors()
        if preprocessors:
            args = [preprocessors[0]] + list(args)
        else:
            args = ["ddr2019"] + list(args)

    # Phase 1: get keyword (first positional); disable help so --help goes to phase 2
    phase1_parser = argparse.ArgumentParser(add_help=False)
    phase1_parser.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Preprocessor keyword (e.g. ddr2019)",
    )
    phase1_ns, _ = phase1_parser.parse_known_args(args)
    keyword = phase1_ns.dataset
    if not keyword:
        phase1_parser.print_help()
        print("\nError: missing required argument: dataset (preprocessor keyword)", file=sys.stderr)
        return 1

    # Lookup preprocessor (registry already populated by imports at top)
    try:
        preprocessor_cls = get_preprocessor(keyword)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Phase 2: full parser with preprocessor-specific arguments
    phase2_parser = argparse.ArgumentParser(
        description="Preprocess datasets for SAM-AI project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  preprocess-dataset ddr2019
  preprocess-dataset ddr2019 --raw-img-dir /path/to/images
  preprocess-dataset ddr2019 --min-size 512 --target-size 512 512
  preprocess-dataset ddr2019 --output-name ddr2019_384 --target-size 384 384
        """,
    )
    phase2_parser.add_argument(
        "dataset",
        help="Preprocessor keyword (e.g. 'ddr2019')",
    )
    preprocessor_cls.add_arguments(phase2_parser)
    parsed = phase2_parser.parse_args(args)

    try:
        preprocessor_config = _build_config_from_args(parsed, keyword)
        preprocessor = preprocessor_cls()
        print(f"Starting preprocessing for dataset: {keyword}")

        raw_img_dir = preprocessor_config.raw_img_dir
        raw_path = Path(raw_img_dir)
        original_image_count = len(list(raw_path.glob("*.jpg"))) if raw_path.exists() else 0

        results = preprocessor.run(preprocessor_config)

        print(f"\nPreprocessing complete for {keyword}:")
        print(f"  - Original dataset: {original_image_count} images")
        print(f"  - Processed dataset: {results['images_processed']} images")
        print(
            f"  - Images skipped: {original_image_count - results['images_processed']} images (too small or would require upscaling)"
        )
        print(f"  - Labels saved to: {results['labels_path']}")
        return 0
    except Exception as e:
        print(f"Error during preprocessing: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
