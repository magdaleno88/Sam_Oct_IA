#!/usr/bin/env python3
"""Create moderate physical or logical OCT class balancing artifacts."""

import argparse

from sam_ml.oct.config import load_config
from sam_ml.oct.dataset_management import balance_dataset
from sam_ml.utils.progress import progress_disabled, stage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/oct.yaml")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--strategy", choices=["moderate"], default="moderate")
    parser.add_argument("--balance-mode", choices=["physical", "sampler", "class-weights"])
    parser.add_argument("--max-ratio", type=float)
    parser.add_argument("--max-undersample-fraction", type=float)
    parser.add_argument("--max-oversample-factor", type=float)
    parser.add_argument("--sampling-unit", choices=["auto", "patient", "image"])
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    settings = config.dataset_management.balancing
    if args.max_ratio is not None: settings.max_ratio = args.max_ratio
    if args.max_undersample_fraction is not None: settings.max_undersample_fraction = args.max_undersample_fraction
    if args.max_oversample_factor is not None: settings.max_oversample_factor = args.max_oversample_factor
    if args.sampling_unit is not None: settings.sampling_unit = args.sampling_unit
    stage(f"Entrada: {args.input_root}")
    stage(f"Modo: {args.balance_mode or settings.mode}")
    try:
        result = balance_dataset(
            args.input_root, config.dataset_management, output_root=args.output_root,
            balance_mode=args.balance_mode, splits=tuple(args.splits) if args.splits else None,
            seed=args.seed, overwrite=args.overwrite,
            progress_disabled=progress_disabled(args.no_progress),
        )
    except KeyboardInterrupt:
        stage("Proceso cancelado por el usuario.")
        return 130
    stage(f"Conteos antes: {result['before_counts']}")
    stage(f"Conteos despues: {result['after_counts']}")
    stage(f"Relacion residual: {result['residual_ratio']:.2f}")
    stage(f"Pesos: {result['class_weights']}")
    stage(f"Reportes: {config.dataset_management.reports_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
