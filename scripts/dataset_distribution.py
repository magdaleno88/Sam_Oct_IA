#!/usr/bin/env python3
"""Report OCT image counts without decoding image contents."""

import argparse
import json
from pathlib import Path

from sam_ml.oct.config import load_config
from sam_ml.oct.dataset_management import save_distribution, scan_dataset
from sam_ml.utils.progress import progress_disabled, stage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/oct.yaml")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--report-prefix", default="original")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    reports = Path(args.output_dir) if args.output_dir else config.dataset_management.reports_dir
    disabled = progress_disabled(args.no_progress)
    stage(f"Entrada: {args.input_root}")
    frame = scan_dataset(args.input_root, config.dataset_management.extensions, disabled)
    summary = save_distribution(frame, reports, args.report_prefix)
    stage("Distribucion del dataset")
    stage("Clase      Imagenes   Porcentaje")
    for name, count in summary["class_counts"].items():
        stage(f"{name:<10} {count:>8,}   {summary['class_percentages'][name]:>9.1f}%")
    stage(f"Total: {summary['total_images']:,}")
    ratio = summary["majority_minority_ratio"]
    stage(f"Relacion mayoria/minoria: {ratio:.2f}" if ratio is not None else "Relacion mayoria/minoria: n/a")
    for warning in summary["warnings"]:
        stage(f"Advertencia: {warning}")
    stage(f"Reportes: {reports}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
