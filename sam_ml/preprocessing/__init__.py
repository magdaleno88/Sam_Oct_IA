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
  preprocess-dataset ddr2019 --min-size 512 --target-size 512 512
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
        "--min-size",
        type=int,
        default=512,
        help="Minimum size (width and height) required to process an image. Defaults to 512.",
    )
    
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=[512, 512],
        help="Target size for image resizing (width height). Defaults to 512 512.",
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
        kwargs["min_size"] = parsed_args.min_size
        kwargs["target_size"] = tuple(parsed_args.target_size)
        
        try:
            print(f"Starting preprocessing for dataset: {parsed_args.dataset}")
            
            # Count original images before processing
            from pathlib import Path
            raw_img_dir = kwargs.get("raw_img_dir") or "data/raw/ddr2019/DR_grading/DR_grading"
            raw_path = Path(raw_img_dir)
            if raw_path.exists():
                original_image_count = len(list(raw_path.glob("*.jpg")))
            else:
                original_image_count = 0
            
            results = preprocess_ddr2019(**kwargs)
            
            print(f"\nPreprocessing complete for {parsed_args.dataset}:")
            print(f"  - Original dataset: {original_image_count} images")
            print(f"  - Processed dataset: {results['images_processed']} images")
            print(f"  - Images skipped: {original_image_count - results['images_processed']} images (too small or would require upscaling)")
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