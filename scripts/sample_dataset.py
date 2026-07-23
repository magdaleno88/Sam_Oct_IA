#!/usr/bin/env python3
"""Create a reproducible split/class-stratified OCT subset."""

import argparse
import json

from sam_ml.oct.config import load_config
from sam_ml.oct.dataset_management import sample_dataset
from sam_ml.utils.progress import progress_disabled, stage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/oct.yaml")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--percentage", type=float)
    parser.add_argument("--sampling-unit", choices=["auto", "patient", "image"])
    parser.add_argument("--mode", choices=["copy", "hardlink", "symlink", "manifest"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    management = config.dataset_management
    percentage = args.percentage if args.percentage is not None else management.sampling.percentage
    stage(f"Configuracion: {args.config}")
    stage(f"Entrada: {args.input_root}")
    stage(f"Salida: {args.output_root}")
    stage(f"Porcentaje solicitado: {percentage:g}%")
    try:
        _, details = sample_dataset(
            args.input_root, args.output_root, percentage, management,
            sampling_unit=args.sampling_unit, mode=args.mode, seed=args.seed,
            overwrite=args.overwrite, progress_disabled=progress_disabled(args.no_progress),
        )
    except KeyboardInterrupt:
        stage("Proceso cancelado por el usuario.")
        return 130
    stage(f"Porcentaje real: {details['actual_percentage']:.2f}%")
    stage(f"Seleccionadas: {details['selected']:,}/{details['original']:,}")
    stage(f"Unidad: {details['sampling_unit']}")
    stage(f"Modo: {details['mode']}")
    stage(f"Manifiesto: {management.reports_dir / 'sample_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
