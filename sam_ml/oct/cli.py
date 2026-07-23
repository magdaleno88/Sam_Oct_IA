"""Small command-line adapters for OCT data and inference."""

import argparse
import json
import time

from sam_ml.oct.config import load_config
from sam_ml.oct.data import audit_dataset, create_manifests
from sam_ml.oct.inference import load_checkpoint_model, predict_image
from sam_ml.oct.preprocessing import DatasetProcessingCancelled, preprocess_dataset
from sam_ml.utils.progress import progress_disabled, stage


def _add_progress_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Disable animated progress bars for CI or redirected logs",
    )


def _duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def audit_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/oct.yaml")
    parser.add_argument("--blur-threshold", type=float)
    _add_progress_option(parser)
    args = parser.parse_args(argv)
    disabled = progress_disabled(args.no_progress)
    stage(f"Configuracion: {args.config}")
    config = load_config(args.config)
    stage(f"Entrada: {config.data.root}")
    try:
        summary = audit_dataset(
            config, args.blur_threshold, progress_disabled=disabled, show_stages=True,
        )
    except KeyboardInterrupt:
        stage("Proceso cancelado por el usuario.")
        raise SystemExit(130)
    print(json.dumps(summary, indent=2))


def create_splits_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/oct.yaml")
    _add_progress_option(parser)
    args = parser.parse_args(argv)
    disabled = progress_disabled(args.no_progress)
    stage(f"Configuracion: {args.config}")
    config = load_config(args.config)
    stage(f"Entrada: {config.data.root}")
    stage(f"Salida: {config.data.manifest_dir}")
    try:
        manifests = create_manifests(
            config, progress_disabled=disabled, show_stages=True,
        )
    except KeyboardInterrupt:
        stage("Proceso cancelado por el usuario.")
        raise SystemExit(130)
    print(json.dumps({name: len(frame) for name, frame in manifests.items()}, indent=2))


def preprocess_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Clean OCT border artifacts without modifying source images"
    )
    parser.add_argument("--config", default="configs/oct.yaml")
    parser.add_argument("--input-root")
    parser.add_argument("--output-root")
    _add_progress_option(parser)
    args = parser.parse_args(argv)
    disabled = progress_disabled(args.no_progress)
    config = load_config(args.config)
    input_root = args.input_root or config.data.root
    output_root = args.output_root or config.preprocessing.output_root
    stage(f"Configuracion: {args.config}")
    stage(f"Entrada: {input_root}")
    stage(f"Salida: {output_root}")
    stage("Buscando archivos OCT...")
    started = time.perf_counter()
    try:
        report = preprocess_dataset(
            input_root, output_root, config.preprocessing,
            progress_disabled=disabled, show_stages=True,
        )
    except DatasetProcessingCancelled as exc:
        stage("Proceso cancelado por el usuario.")
        stage(f"Procesadas antes de cancelar: {exc.processed:,}/{exc.total:,}")
        raise SystemExit(130)
    elapsed = time.perf_counter() - started
    summary = report.attrs.get("summary", {})
    processed = int(summary.get("processed", 0))
    stage("Preprocesamiento completado")
    stage(f"Procesadas: {processed:,}")
    stage(f"Corregidas: {int(summary.get('corrected', 0)):,}")
    stage(f"Sin cambios: {int(summary.get('unchanged', 0)):,}")
    stage(f"Panoramicas marcadas: {int(summary.get('panoramic', 0)):,}")
    stage(f"Omitidas: {int(summary.get('skipped', 0)):,}")
    stage(f"Errores: {int(summary.get('errors', 0)):,}")
    stage(f"Tiempo total: {_duration(elapsed)}")
    stage(f"Velocidad media: {processed / elapsed if elapsed else 0:.1f} imagenes/s")
    stage(f"Salida: {output_root}")
    stage(f"Reporte: {config.preprocessing.quality_control_dir / 'preprocessing_report.csv'}")


def predict_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--config", default="configs/oct.yaml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    kwargs = {"num_classes": config.model.num_classes, "pretrained": False,
              "dropout": config.model.dropout, "freeze_backbone": False}
    if config.model.name == "improved_resnet50":
        kwargs["replace_stride_with_dilation"] = config.model.replace_stride_with_dilation
    models = [load_checkpoint_model(item, config.model.name, kwargs) for item in args.checkpoint]
    print(json.dumps(predict_image(
        models, args.image, args.device, config.data.image_size, config.preprocessing
    ), indent=2, ensure_ascii=False))
