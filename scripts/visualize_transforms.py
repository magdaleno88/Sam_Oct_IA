#!/usr/bin/env python3
"""Save deterministic side-by-side samples of original and augmented OCT images."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from sam_ml.oct.config import load_config
from sam_ml.oct.dataset import build_oct_transform, denormalize_oct
from sam_ml.utils.progress import progress_disabled, stage, track_progress


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/oct.yaml")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--output", default="reports/transform_samples.png")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)
    disabled = progress_disabled(args.no_progress)
    stage(f"Configuracion: {args.config}")
    config = load_config(args.config)
    manifest = config.data.manifest_dir / "train.csv"
    stage(f"Leyendo manifiesto: {manifest}")
    rows = pd.read_csv(manifest).head(args.count)
    stage(f"Se visualizaran {len(rows):,} imagenes.")
    if rows.empty:
        stage("No hay imagenes para visualizar.")
        return 0
    transform = build_oct_transform(True, config.data.image_size)
    panels = []
    for path in track_progress(
        rows["image_path"], description="Generando muestras", total=len(rows),
        unit="img", disabled=disabled,
    ):
        with Image.open(path) as source:
            source = source.convert("L").resize((config.data.image_size, config.data.image_size))
            transformed = denormalize_oct(transform(source)).mean(0).numpy()
        pair = np.concatenate([np.asarray(source), (transformed * 255).astype(np.uint8)], axis=1)
        panels.append(pair)
    canvas = Image.fromarray(np.concatenate(panels, axis=0))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    stage(f"Salida: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
