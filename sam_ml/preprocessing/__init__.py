"""Preprocessing module with polymorphic dataset processors."""

import argparse
import sys
from typing import Optional


def main(args: Optional[list[str]] = None) -> int:
    """Main entry point for preprocessing scripts.
    
    Routes to the appropriate preprocessing script based on the dataset parameter.
    
    Args:
        args: Command-line arguments. If None, uses sys.argv.
    
    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parser = argparse.ArgumentParser(
        description="Preprocess datasets for SAM-AI project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  preprocess-dataset ddr2019
  preprocess-dataset ddr2019 --raw-img-dir /path/to/images
  preprocess-dataset ddr2019 --resize-shape 256 256
        """,
    )
    
    parser.add_argument(
        "dataset",
        choices=["ddr2019"],
        help="Dataset to preprocess (e.g., 'ddr2019')",
    )
    
    parser.add_argument(
        "--raw-img-dir",
        type=str,
        help="Path to raw images directory (overrides default)",
    )
    
    parser.add_argument(
        "--raw-csv-path",
        type=str,
        help="Path to raw CSV file (overrides default)",
    )
    
    parser.add_argument(
        "--processed-dir",
        type=str,
        help="Path to processed output directory (overrides default)",
    )
    
    parser.add_argument(
        "--resize-shape",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Target size for image resizing (width height). If not specified, images keep their original size.",
    )
    
    if args is None:
        args = sys.argv[1:]
    
    parsed_args = parser.parse_args(args)
    
    # Route to appropriate preprocessing script
    if parsed_args.dataset == "ddr2019":
        from sam_ml.preprocessing.preprocess_ddr2019 import preprocess_ddr2019
        
        # Prepare keyword arguments
        kwargs = {}
        if parsed_args.raw_img_dir:
            kwargs["raw_img_dir"] = parsed_args.raw_img_dir
        if parsed_args.raw_csv_path:
            kwargs["raw_csv_path"] = parsed_args.raw_csv_path
        if parsed_args.processed_dir:
            kwargs["processed_dir"] = parsed_args.processed_dir
        if parsed_args.resize_shape:
            kwargs["resize_shape"] = tuple(parsed_args.resize_shape)
        
        try:
            print(f"Starting preprocessing for dataset: {parsed_args.dataset}")
            results = preprocess_ddr2019(**kwargs)
            print(f"\nPreprocessing complete for {parsed_args.dataset}:")
            print(f"  - Images processed: {results['images_processed']}")
            print(f"  - Labels saved to: {results['labels_path']}")
            return 0
        except Exception as e:
            print(f"Error during preprocessing: {e}", file=sys.stderr)
            return 1
    else:
        # This should not happen due to choices constraint, but handle it anyway
        print(f"Unknown dataset: {parsed_args.dataset}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())