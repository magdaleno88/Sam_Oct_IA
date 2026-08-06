#!/usr/bin/env python3
"""Prepare a sampled, balanced and preprocessed OCT dataset in one command."""

from __future__ import annotations

import argparse
import json

from sam_ml.oct.config import load_config
from sam_ml.oct.preparation import prepare_oct_dataset
from sam_ml.utils.progress import progress_disabled, stage


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/oct.yaml")
    parser.add_argument("--input-root")
    parser.add_argument("--output-root")
    parser.add_argument("--train-percentage", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sampling-unit", choices=("auto", "patient", "image"))
    parser.add_argument("--sampling-mode", choices=("copy", "manifest"))
    parser.add_argument("--balance-mode", choices=("none", "moderate", "moderate-physical", "class-weights"))
    parser.add_argument("--max-ratio", type=float)
    parser.add_argument("--overwrite", action="store_true", default=None)
    parser.add_argument("--allow-image-level-sampling", action="store_true", default=None)
    parser.add_argument("--keep-temp-on-error", action="store_true", default=None)
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument("--generate-training-config", action="store_true", dest="generate_config")
    config_group.add_argument("--no-generate-training-config", action="store_false", dest="generate_config")
    parser.set_defaults(generate_config=None)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    stage(f"Configuracion: {args.config}")
    try:
        summary = prepare_oct_dataset(
            config, input_root=args.input_root, output_root=args.output_root,
            train_percentage=args.train_percentage, seed=args.seed,
            sampling_unit=args.sampling_unit, sampling_mode=args.sampling_mode,
            balance_mode=args.balance_mode, max_ratio=args.max_ratio,
            overwrite=args.overwrite,
            allow_image_level_sampling=args.allow_image_level_sampling,
            keep_temp_on_error=args.keep_temp_on_error,
            generate_training_config=args.generate_config,
            progress_disabled=progress_disabled(args.no_progress),
        )
    except KeyboardInterrupt:
        stage("Preparacion cancelada; no se publico un dataset parcial.")
        raise SystemExit(130)
    stage("Preparacion completada")
    stage(f"Train original: {summary['original_train']:,}")
    stage(f"Train seleccionado: {summary['selected_train']:,} ({summary['train_percentage_actual']:.2f}%)")
    stage(f"Train despues del balanceo: {summary['balanced_train']:,}")
    stage(f"Test original/final: {summary['original_test']:,}/{summary['final_test']:,}")
    stage("Integridad de test: OK")
    stage(f"Salida: {summary['output_root']}")
    stage(f"Reporte: {summary['reports']}")
    if summary.get("generated_config"):
        stage("Para entrenar:")
        stage(
            "uv run python scripts/train_oct.py "
            f"--config {summary['generated_config']} --model baseline_resnet50 "
            f"--experiment {summary['experiment_name']}_baseline"
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
