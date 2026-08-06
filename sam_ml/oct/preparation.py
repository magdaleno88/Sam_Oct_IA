"""Transactional end-to-end preparation of train/test OCT datasets."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from sam_ml.oct.config import OCTConfig
from sam_ml.oct.constants import CLASS_NAMES, CLASS_TO_INDEX
from sam_ml.oct.dataset_management import (
    _moderate_targets,
    _select_indices,
    distribution_summary,
    moderate_class_weights,
    save_distribution,
    scan_dataset,
)
from sam_ml.oct.preprocessing import (
    load_oct_image,
    preprocess_oct_image,
    save_quality_control,
    write_oct_image,
)
from sam_ml.utils.progress import create_progress, stage


class PreparationError(RuntimeError):
    """Raised when a prepared dataset cannot be safely published."""


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _counts(frame: pd.DataFrame) -> dict[str, int]:
    values = frame["class_name"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    return {name: int(value) for name, value in values.items()}


def _validate_layout(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        raise PreparationError("No supported OCT images were found")
    unexpected = sorted(set(frame["split"]) - {"train", "test"})
    if unexpected:
        raise PreparationError(f"Only official train/test folders are accepted; found: {unexpected}")
    train = frame[frame["split"] == "train"].copy()
    test = frame[frame["split"] == "test"].copy()
    if train.empty or test.empty:
        raise PreparationError("Both train and test splits are required")
    for split_name, subset in (("train", train), ("test", test)):
        missing = [name for name, value in _counts(subset).items() if value == 0]
        if missing:
            raise PreparationError(f"{split_name} has empty classes: {missing}")
    duplicates = frame["source_path"].duplicated(keep=False)
    if duplicates.any():
        raise PreparationError("Duplicate source paths were detected")
    return train, test


def _sampling_unit(train: pd.DataFrame, requested: str, allow_image: bool) -> Literal["patient", "image"]:
    reliable = train["patient_id"].notna().all()
    if requested == "patient" and not reliable:
        raise PreparationError("Patient sampling requested but patient IDs are not reliable")
    if requested == "auto":
        if reliable:
            return "patient"
        if not allow_image:
            raise PreparationError(
                "Patient IDs are not reliable. Re-run with --allow-image-level-sampling "
                "to explicitly accept leakage risk"
            )
        return "image"
    if requested == "image" and not allow_image:
        raise PreparationError("Image-level sampling requires --allow-image-level-sampling")
    return requested  # type: ignore[return-value]


def _sample_train(
    train: pd.DataFrame, percentage: float, unit: Literal["patient", "image"],
    seed: int, minimum: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    chosen: set[int] = set()
    for class_name, subset in train.groupby("class_name", sort=True):
        target = min(len(subset), max(minimum, round(len(subset) * percentage / 100)))
        chosen.update(_select_indices(subset, target, unit, rng))
    manifest = train.copy()
    manifest["selected"] = manifest.index.isin(chosen)
    manifest["sampling_unit"] = unit
    manifest["seed"] = seed
    manifest["requested_percentage"] = percentage
    selected = manifest[manifest["selected"]].copy().reset_index(drop=True)
    if any(value == 0 for value in _counts(selected).values()):
        raise PreparationError("Sampling produced an empty training class")
    return selected, manifest.reset_index(drop=True)


def _balance_train(
    selected: pd.DataFrame, config: OCTConfig, mode: str, max_ratio: float, seed: int,
    unit: Literal["patient", "image"],
) -> tuple[pd.DataFrame, dict[str, float], dict[str, Any]]:
    before = _counts(selected)
    weights = moderate_class_weights(
        before,
        config.dataset_management.balancing.min_class_weight,
        config.dataset_management.balancing.max_class_weight,
    )
    balanced = selected.copy()
    balanced["duplicate_index"] = 0
    if mode == "moderate-physical":
        preparation_balance = config.dataset_preparation.balancing
        settings = config.dataset_management.balancing.model_copy(update={
            "max_ratio": max_ratio,
            "max_undersample_fraction": preparation_balance.max_undersample_fraction,
            "max_oversample_factor": preparation_balance.max_oversample_factor,
        })
        targets = _moderate_targets(before, settings)
        rng = np.random.default_rng(seed)
        pieces: list[pd.DataFrame] = []
        for name, subset in selected.groupby("class_name", sort=True):
            target = targets[name]
            if target <= len(subset):
                chosen = subset.loc[sorted(_select_indices(subset, target, unit, rng))].copy()
            else:
                chosen = subset.copy()
                groups = (
                    [group for _, group in subset.groupby("patient_id", sort=True)]
                    if unit == "patient" else [subset.loc[[idx]] for idx in sorted(subset.index)]
                )
                order = rng.permutation(len(groups))
                extras: list[pd.DataFrame] = []
                cursor = 0
                while len(chosen) + sum(len(item) for item in extras) < target:
                    group = groups[int(order[cursor % len(order)])]
                    if len(chosen) + sum(len(item) for item in extras) + len(group) > target:
                        break
                    extras.append(group.copy())
                    cursor += 1
                if extras:
                    chosen = pd.concat([chosen, *extras], ignore_index=True)
            piece = chosen.reset_index(drop=True)
            piece["duplicate_index"] = piece.groupby("relative_path").cumcount()
            pieces.append(piece)
        balanced = pd.concat(pieces, ignore_index=True)
    elif mode not in {"none", "class-weights"}:
        raise PreparationError(f"Unknown balance mode: {mode}")
    after = _counts(balanced)
    positive = [value for value in after.values() if value]
    return balanced, weights, {
        "mode": mode, "before_counts": before, "after_counts": after,
        "residual_ratio": max(positive) / min(positive),
    }


def _destination_relative(row: Any, split: str) -> Path:
    source_relative = Path(str(row.relative_path))
    tail = Path(*source_relative.parts[2:]) if len(source_relative.parts) > 2 else Path(source_relative.name)
    relative = Path(split) / str(row.class_name) / tail
    duplicate = int(getattr(row, "duplicate_index", 0))
    if duplicate:
        relative = relative.with_name(f"{relative.stem}__repeat_{duplicate:03d}{relative.suffix}")
    return relative


def _process_split(
    frame: pd.DataFrame, split: str, temp_root: Path, report_root: Path,
    config: OCTConfig, disabled: bool, qc_remaining: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with create_progress(
        description=f"Preprocesando {split}", total=len(frame), unit="img", disabled=disabled,
    ) as progress:
        for index, row in enumerate(frame.itertuples(index=False)):
            relative = _destination_relative(row, split)
            output = temp_root / relative
            record = {
                "split": split, "class_name": row.class_name, "patient_id": row.patient_id,
                "source_path": row.source_path, "output_relative_path": relative.as_posix(),
                "output_path": str(output), "status": "error", "warnings": "",
                "duplicate_index": int(getattr(row, "duplicate_index", 0)),
            }
            try:
                original = load_oct_image(row.source_path)
                result = preprocess_oct_image(original, config.preprocessing, config.preprocessing.seed)
                write_oct_image(output, result.image)
                record.update(result.metadata)
                record["status"] = "ok"
                if qc_remaining[0] > 0:
                    save_quality_control(report_root / "quality_control", relative, original, result)
                    qc_remaining[0] -= 1
            except Exception as exc:
                record["warnings"] = f"{type(exc).__name__}: {exc}"
            rows.append(record)
            progress.update(1)
    return pd.DataFrame(rows)


def _integrity(
    source_test: pd.DataFrame, processed_test: pd.DataFrame, temp_root: Path, hashes: bool,
) -> dict[str, Any]:
    successful = processed_test[processed_test["status"] == "ok"]
    expected_paths = set(source_test["relative_path"].astype(str))
    actual_paths = set(successful["output_relative_path"].astype(str))
    files_on_disk = {
        path.relative_to(temp_root).as_posix()
        for path in (temp_root / "test").rglob("*") if path.is_file()
    }
    mapping_unique = successful["source_path"].nunique() == len(source_test)
    output_unique = successful["output_relative_path"].nunique() == len(source_test)
    source_counts, output_counts = _counts(source_test), _counts(successful)
    result: dict[str, Any] = {
        "passed": bool(
            len(successful) == len(source_test) and mapping_unique and output_unique
            and source_counts == output_counts and expected_paths == actual_paths == files_on_disk
        ),
        "source_total": len(source_test), "output_total": len(successful),
        "source_class_counts": source_counts, "output_class_counts": output_counts,
        "one_to_one_mapping": bool(mapping_unique and output_unique),
        "relative_paths_preserved": expected_paths == actual_paths,
        "no_missing_or_extra_files": actual_paths == files_on_disk,
        "sampled": False, "balanced": False,
    }
    if hashes:
        digest = hashlib.sha256()
        for relative in sorted(successful["output_relative_path"]):
            digest.update((temp_root / relative).read_bytes())
        result["ordered_output_sha256"] = digest.hexdigest()
    return result


def _training_manifests(
    final_train: pd.DataFrame, final_test: pd.DataFrame, final_root: Path,
    report_root: Path, config: OCTConfig, allow_image: bool,
) -> Path:
    """Create manifests expected by the existing trainer without changing its code."""
    successful = final_train[final_train["status"] == "ok"].copy()
    base = successful[successful["duplicate_index"] == 0].copy()
    grouped = base["patient_id"].notna().all()
    if grouped:
        per_class = base.groupby("class_name")["patient_id"].nunique()
        grouped = bool(not per_class.empty and per_class.min() >= 2)
    if grouped:
        splits = min(max(2, round(1 / config.data.val_fraction)), int(per_class.min()))
        splitter = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=config.data.seed)
        _, val_indices = next(splitter.split(base, base["class_name"], base["patient_id"]))
        val_patients = set(base.iloc[val_indices]["patient_id"])
        is_val = successful["patient_id"].isin(val_patients)
    else:
        if not allow_image:
            raise PreparationError(
                "The prepared sample is too small for patient-level validation manifests; "
                "explicit image-level permission is required"
            )
        _, val_indices = train_test_split(
            np.arange(len(successful)), test_size=config.data.val_fraction,
            random_state=config.data.seed, stratify=successful["class_name"],
        )
        is_val = pd.Series(False, index=successful.index)
        is_val.iloc[val_indices] = True

    manifest_dir = report_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    def convert(frame: pd.DataFrame, split: str) -> pd.DataFrame:
        result = pd.DataFrame({
            "image_path": frame["output_relative_path"].map(lambda p: str(final_root / p)),
            "label": frame["class_name"],
            "class_index": frame["class_name"].map(CLASS_TO_INDEX),
            "patient_id": frame["patient_id"], "source": "prepared_oct", "split": split,
        })
        return result

    # Repetitions never enter validation; validation retains one base image per selected path.
    validation_rows = successful[is_val & successful["duplicate_index"].eq(0)]
    training_rows = successful[~is_val]
    convert(training_rows, "train").to_csv(manifest_dir / "train.csv", index=False)
    convert(validation_rows, "val").to_csv(manifest_dir / "val.csv", index=False)
    convert(final_test[final_test["status"] == "ok"], "test").to_csv(manifest_dir / "test.csv", index=False)
    return manifest_dir


def _generated_config(
    config: OCTConfig, final_root: Path, manifest_dir: Path, experiment: str,
    balance_mode: str,
) -> Path:
    generated_dir = config.dataset_preparation.generated_config_dir
    generated_dir.mkdir(parents=True, exist_ok=True)
    output = generated_dir / f"{experiment}.yaml"
    raw = config.model_dump(mode="json")
    raw["data"]["root"] = str(final_root)
    raw["data"]["manifest_dir"] = str(manifest_dir)
    raw["data"]["exclusions_file"] = str(manifest_dir / "excluded_images.csv")
    raw["preprocessing"]["enabled"] = False
    raw["preprocessing"]["output_root"] = str(final_root)
    raw["training"]["balance_mode"] = "class_weights" if balance_mode == "class-weights" else "none"
    output.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return output


def prepare_oct_dataset(
    config: OCTConfig, *, input_root: str | Path | None = None,
    output_root: str | Path | None = None, train_percentage: float | None = None,
    seed: int | None = None, sampling_unit: str | None = None,
    sampling_mode: str | None = None, balance_mode: str | None = None,
    max_ratio: float | None = None, overwrite: bool | None = None,
    allow_image_level_sampling: bool | None = None,
    keep_temp_on_error: bool | None = None, progress_disabled: bool = True,
    generate_training_config: bool | None = None,
) -> dict[str, Any]:
    """Prepare and atomically publish an OCT train/test dataset."""
    settings = config.dataset_preparation
    source = Path(input_root or settings.input_root).resolve()
    final = Path(output_root or settings.output_root).resolve()
    percentage = settings.train_sampling.percentage if train_percentage is None else train_percentage
    actual_seed = settings.seed if seed is None else seed
    requested_unit = sampling_unit or settings.train_sampling.unit
    operation_mode = sampling_mode or settings.train_sampling.mode
    configured_balance = settings.balancing.mode if settings.balancing.enabled else "none"
    actual_balance = balance_mode or configured_balance
    actual_balance = "moderate-physical" if actual_balance == "moderate" else actual_balance
    ratio = settings.balancing.max_ratio if max_ratio is None else max_ratio
    replace = settings.overwrite if overwrite is None else overwrite
    allow_image = settings.train_sampling.allow_image_level_sampling if allow_image_level_sampling is None else allow_image_level_sampling
    keep_temp = settings.keep_temp_on_error if keep_temp_on_error is None else keep_temp_on_error
    generate_config = settings.generate_training_config if generate_training_config is None else generate_training_config
    if not 0 < percentage <= 100:
        raise PreparationError("train percentage must satisfy 0 < percentage <= 100")
    if operation_mode not in {"copy", "manifest"}:
        raise PreparationError(f"Unknown sampling mode: {operation_mode}")
    if not settings.preprocessing.enabled or not settings.preprocessing.use_existing_config:
        raise PreparationError("Unified preparation requires the existing OCT preprocessing pipeline")
    if final == source or final.is_relative_to(source):
        raise PreparationError("Output must be separate from and outside the input dataset")
    final.parent.mkdir(parents=True, exist_ok=True)
    temp = final.parent / f".{final.name}.tmp"
    experiment = settings.experiment_name or final.name
    report_root = settings.reports_root / experiment
    report_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        if final.exists() and not replace:
            raise FileExistsError(f"Output already exists: {final}")
        if temp.exists():
            raise PreparationError(f"Stale temporary directory exists: {temp}")
        stage("Indexando dataset OCT...")
        frame = scan_dataset(source, config.dataset_management.extensions, progress_disabled)
        train, test = _validate_layout(frame)
        required = sum(Path(path).stat().st_size for path in pd.concat([train, test])["source_path"])
        if shutil.disk_usage(final.parent).free < required:
            raise PreparationError("Insufficient free disk space for transactional preparation")
        temp.mkdir(parents=True)
        stage("Seleccionando pacientes o imagenes de train...")
        unit = _sampling_unit(train, requested_unit, allow_image)
        selected, selection_manifest = _sample_train(
            train, percentage, unit, actual_seed, settings.train_sampling.minimum_per_class,
        )
        stage("Balanceando train...")
        balanced, weights, balance_summary = _balance_train(
            selected, config, actual_balance, ratio, actual_seed, unit,
        )
        balanced["output_relative_path"] = [
            _destination_relative(row, "train").as_posix()
            for row in balanced.itertuples(index=False)
        ]
        if balanced["output_relative_path"].duplicated().any():
            raise PreparationError("Repeated train destination names were detected")
        selection_manifest.to_csv(report_root / "selection_manifest.csv", index=False)
        balanced.to_csv(report_root / "balance_manifest.csv", index=False)
        save_distribution(frame, report_root, "original")
        save_distribution(selected, report_root, "selected_train")
        save_distribution(balanced, report_root, "balanced_train")
        _json_write(report_root / "class_weights.json", weights)
        qc_remaining = [config.preprocessing.audit_sample_size]
        train_report = _process_split(
            balanced, "train", temp, report_root, config, progress_disabled, qc_remaining,
        )
        test_frame = test.copy(); test_frame["duplicate_index"] = 0
        expected_test_paths = [
            _destination_relative(row, "test").as_posix()
            for row in test_frame.itertuples(index=False)
        ]
        if len(expected_test_paths) != len(set(expected_test_paths)):
            raise PreparationError("Repeated test destination names were detected")
        test_report = _process_split(
            test_frame, "test", temp, report_root, config, progress_disabled, qc_remaining,
        )
        preprocessing = pd.concat([train_report, test_report], ignore_index=True)
        preprocessing.to_csv(report_root / "preprocessing_report.csv", index=False)
        preprocessing[preprocessing["status"] != "ok"].to_csv(
            report_root / "preprocessing_errors.csv", index=False,
        )
        stage("Verificando integridad de test...")
        integrity = _integrity(test, test_report, temp, settings.integrity.calculate_hashes)
        _json_write(report_root / "test_integrity_report.json", integrity)
        if not integrity["passed"]:
            raise PreparationError("Test integrity verification failed; output was not published")
        stage("Generando reportes y manifiestos...")
        successful_train = train_report[train_report["status"] == "ok"].copy()
        final_distribution = pd.concat([successful_train, test_report], ignore_index=True)
        save_distribution(final_distribution, report_root, "final")
        manifest_dir = _training_manifests(
            train_report, test_report, final, report_root, config, allow_image,
        )
        generated = None
        if generate_config:
            generated = _generated_config(
                config, final, manifest_dir, experiment, actual_balance,
            )
        corrected = int((preprocessing.get("white_area_pixels", 0) > 0).sum())
        panoramic = int(preprocessing.get("warnings", pd.Series(dtype=str)).astype(str).str.contains("extreme_aspect_ratio").sum())
        processed = int((preprocessing["status"] == "ok").sum())
        elapsed = time.perf_counter() - started
        summary = {
            "status": "complete", "input_root": str(source), "output_root": str(final),
            "experiment_name": experiment, "seed": actual_seed,
            "original_total": len(frame),
            "train_percentage_requested": percentage,
            "train_percentage_actual": len(selected) / len(train) * 100,
            "sampling_unit": unit, "sampling_mode": operation_mode,
            "balance": balance_summary, "class_weights": weights,
            "original_train": len(train), "selected_train": len(selected),
            "balanced_train": len(balanced), "final_train": len(successful_train),
            "original_test": len(test), "final_test": len(test_report),
            "original_counts": _counts(frame), "selected_train_counts": _counts(selected),
            "balanced_train_counts": _counts(balanced), "final_counts": _counts(final_distribution),
            "original_patients": int(frame["patient_id"].nunique(dropna=True)),
            "selected_train_patients": int(selected["patient_id"].nunique(dropna=True)),
            "train_errors": int((train_report["status"] != "ok").sum()),
            "preprocessed": processed, "corrected": corrected,
            "unchanged": processed - corrected, "panoramic": panoramic,
            "errors": int((preprocessing["status"] != "ok").sum()),
            "test_integrity": integrity, "duration_seconds": elapsed,
            "images_per_second": processed / elapsed if elapsed else 0.0,
            "generated_config": str(generated) if generated else None,
            "reports": str(report_root),
        }
        backup = final.parent / f".{final.name}.previous"
        if backup.exists():
            raise PreparationError(f"Stale publication backup exists: {backup}")
        if final.exists():
            final.replace(backup)
        try:
            temp.replace(final)
        except BaseException:
            if backup.exists() and not final.exists():
                backup.replace(final)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        _json_write(report_root / "preparation_summary.json", summary)
        pd.DataFrame([summary]).to_csv(report_root / "preparation_summary.csv", index=False)
        return summary
    except BaseException as exc:
        error = {
            "status": "error", "type": type(exc).__name__, "message": str(exc),
            "input_root": str(source), "output_root": str(final), "temporary_root": str(temp),
            "temporary_kept": bool(keep_temp and temp.exists()),
        }
        _json_write(report_root / "preparation_error.json", error)
        if temp.exists() and not keep_temp:
            shutil.rmtree(temp)
        raise
