"""Path-only OCT distribution, reproducible sampling, and moderate balancing tools."""

from __future__ import annotations

import json
import math
import os
import shutil
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from sam_ml.oct.config import DatasetBalancingConfig, DatasetManagementConfig
from sam_ml.oct.constants import CLASS_NAMES
from sam_ml.oct.data import SPLIT_ALIASES, infer_patient_id
from sam_ml.utils.progress import create_progress, stage


def _validate_root(root: str | Path) -> Path:
    path = Path(root).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Input dataset directory not found: {path}")
    return path


def _validate_output(input_root: Path, output_root: str | Path) -> Path:
    output = Path(output_root).resolve()
    if output == input_root:
        raise ValueError("output-root must be different from input-root")
    if output.is_relative_to(input_root):
        raise ValueError("output-root cannot be inside input-root")
    return output


def scan_dataset(
    root: str | Path, extensions: tuple[str, ...], progress_disabled: bool = True,
) -> pd.DataFrame:
    """Index supported image paths without opening or decoding them."""
    root = _validate_root(root)
    extensions = tuple(item.lower() if item.startswith(".") else f".{item.lower()}" for item in extensions)
    official = any((root / name).is_dir() for name in SPLIT_ALIASES)
    bases = [(root / name, split) for name, split in SPLIT_ALIASES.items() if (root / name).is_dir()] if official else [(root, "unassigned")]
    rows: list[dict[str, object]] = []
    unknown: set[str] = set()
    with create_progress(description="Analizando imagenes", unit="archivo", disabled=progress_disabled) as progress:
        for base, split in bases:
            for directory in sorted(item for item in base.iterdir() if item.is_dir()):
                if directory.name.upper() not in CLASS_NAMES:
                    if any(path.is_file() and path.suffix.lower() in extensions for path in directory.rglob("*")):
                        unknown.add(directory.name)
                    continue
                class_name = directory.name.upper()
                for path in sorted(directory.rglob("*")):
                    if path.is_file() and path.suffix.lower() in extensions:
                        relative = path.relative_to(root)
                        rows.append({
                            "source_path": str(path), "relative_path": relative.as_posix(),
                            "split": split, "class_name": class_name,
                            "patient_id": infer_patient_id(path),
                        })
                        progress.update(1)
    if unknown:
        raise ValueError(f"Unknown non-empty class directories: {sorted(unknown)}")
    return pd.DataFrame(rows, columns=["source_path", "relative_path", "split", "class_name", "patient_id"])


def distribution_summary(frame: pd.DataFrame, low_class_threshold: float = 0.05) -> dict[str, object]:
    counts = frame["class_name"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    total = int(counts.sum())
    nonzero = counts[counts > 0]
    majority = str(counts.idxmax()) if total else None
    minority = str(nonzero.idxmin()) if not nonzero.empty else None
    ratio = float(nonzero.max() / nonzero.min()) if len(nonzero) else None
    warnings_list = []
    empty = [name for name, value in counts.items() if value == 0]
    if empty:
        warnings_list.append(f"Empty classes: {', '.join(empty)}")
    for name, value in counts.items():
        proportion = value / total if total else 0
        if value and proportion < low_class_threshold:
            warnings_list.append(f"{name} is considerably underrepresented ({proportion:.1%})")
    if ratio is not None and ratio > 2:
        warnings_list.append(f"Possible class imbalance: majority/minority ratio={ratio:.2f}")
    split_counts = {}
    for split, subset in frame.groupby("split", sort=False):
        split_counts[str(split)] = {
            name: int(value) for name, value in subset["class_name"].value_counts().reindex(CLASS_NAMES, fill_value=0).items()
        }
    return {
        "total_images": total,
        "class_counts": {name: int(value) for name, value in counts.items()},
        "class_percentages": {name: float(value / total * 100) if total else 0.0 for name, value in counts.items()},
        "split_counts": split_counts,
        "split_percentages": {split: float(sum(values.values()) / total * 100) if total else 0.0 for split, values in split_counts.items()},
        "majority_class": majority, "minority_class": minority,
        "majority_minority_ratio": ratio, "warnings": warnings_list,
    }


def save_distribution(
    frame: pd.DataFrame, reports_dir: str | Path, prefix: str,
) -> dict[str, object]:
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    summary = distribution_summary(frame)
    rows = []
    for split, values in summary["split_counts"].items():
        split_total = sum(values.values())
        for name, count in values.items():
            rows.append({"split": split, "class_name": name, "images": count,
                         "split_percentage": count / split_total * 100 if split_total else 0})
    pd.DataFrame(rows, columns=["split", "class_name", "images", "split_percentage"]).to_csv(
        reports / f"{prefix}_distribution.csv", index=False
    )
    (reports / f"{prefix}_distribution.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def _select_indices(
    frame: pd.DataFrame, target: int, unit: Literal["patient", "image"], rng: np.random.Generator,
) -> set[int]:
    if target >= len(frame):
        return set(frame.index)
    if unit == "image":
        choices = rng.choice(np.array(sorted(frame.index)), size=target, replace=False)
        return set(int(item) for item in choices)
    groups = [(patient, sorted(group.index)) for patient, group in frame.groupby("patient_id", sort=True)]
    order = rng.permutation(len(groups))
    cumulative: list[tuple[int, int]] = [(0, 0)]
    running = 0
    for position, group_index in enumerate(order, start=1):
        running += len(groups[int(group_index)][1])
        cumulative.append((position, running))
    best_position, _ = min(cumulative[1:], key=lambda item: (abs(item[1] - target), item[1] < 1))
    selected: set[int] = set()
    for group_index in order[:best_position]:
        selected.update(groups[int(group_index)][1])
    return selected


def _resolve_unit(frame: pd.DataFrame, requested: str) -> Literal["patient", "image"]:
    reliable = not frame.empty and frame["patient_id"].notna().all()
    if requested == "patient" and not reliable:
        raise ValueError("Patient sampling requested but reliable patient_id is unavailable")
    if requested == "auto":
        if reliable:
            return "patient"
        warnings.warn("Falling back to image-level sampling; future split leakage is possible", RuntimeWarning)
        return "image"
    return requested  # type: ignore[return-value]


def _transfer(source: Path, destination: Path, mode: str, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {destination}")
        destination.unlink()
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source.resolve())
    else:
        raise ValueError(f"Unsupported transfer mode: {mode}")


def sample_dataset(
    input_root: str | Path, output_root: str | Path, percentage: float,
    config: DatasetManagementConfig, sampling_unit: str | None = None,
    mode: str | None = None, seed: int | None = None, overwrite: bool = False,
    progress_disabled: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Stratify within split/class, preserving whole patients whenever reliable."""
    if not 0 < percentage <= 100:
        raise ValueError("percentage must satisfy 0 < percentage <= 100")
    source_root = _validate_root(input_root)
    destination_root = _validate_output(source_root, output_root)
    actual_seed = config.seed if seed is None else seed
    requested_unit = sampling_unit or config.sampling.unit
    operation_mode = mode or config.sampling.mode
    if operation_mode not in {"copy", "hardlink", "symlink", "manifest"}:
        raise ValueError(f"Unknown mode: {operation_mode}")
    frame = scan_dataset(source_root, config.extensions, progress_disabled)
    if frame.empty:
        raise ValueError("Dataset contains no supported images")
    known_patients = frame.dropna(subset=["patient_id"])
    patient_split_counts = known_patients.groupby("patient_id")["split"].nunique()
    leaking_patients = patient_split_counts[patient_split_counts > 1]
    if not leaking_patients.empty and requested_unit in {"auto", "patient"}:
        raise ValueError(
            f"{len(leaking_patients)} patient IDs occur in multiple existing splits; "
            "audit leakage before patient-level sampling"
        )
    rng = np.random.default_rng(actual_seed)
    selected_indices: set[int] = set()
    actual_units: set[str] = set()
    actual_by_stratum = {}
    for (split, class_name), subset in frame.groupby(["split", "class_name"], sort=True):
        if subset.empty:
            continue
        target = max(config.sampling.minimum_per_class, round(len(subset) * percentage / 100))
        target = min(target, len(subset))
        unit = _resolve_unit(subset, requested_unit)
        actual_units.add(unit)
        chosen = _select_indices(subset, target, unit, rng)
        selected_indices.update(chosen)
        actual_by_stratum[f"{split}/{class_name}"] = len(chosen) / len(subset) * 100
    manifest = frame.copy()
    manifest["selected"] = manifest.index.isin(selected_indices)
    manifest["destination_path"] = manifest["relative_path"].map(lambda item: str(destination_root / item))
    manifest["sampling_unit"] = "mixed" if len(actual_units) > 1 else next(iter(actual_units))
    manifest["seed"] = actual_seed
    manifest["requested_percentage"] = percentage
    manifest["actual_percentage"] = len(selected_indices) / len(frame) * 100
    manifest["operation_mode"] = operation_mode
    selected = manifest[manifest["selected"]]
    if operation_mode != "manifest":
        with create_progress(description=f"{operation_mode.capitalize()} archivos", total=len(selected), unit="archivo", disabled=progress_disabled) as progress:
            for row in selected.itertuples():
                _transfer(Path(row.source_path), Path(row.destination_path), operation_mode, overwrite)
                progress.update(1)
    reports = config.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(reports / "sample_manifest.csv", index=False)
    (reports / "sample_manifest.json").write_text(manifest.to_json(orient="records", indent=2), encoding="utf-8")
    original_summary = save_distribution(frame, reports, "original")
    sampled_summary = save_distribution(selected, reports, "sampled")
    details = {
        "requested_percentage": percentage,
        "actual_percentage": len(selected) / len(frame) * 100,
        "actual_percentage_by_split_class": actual_by_stratum,
        "selected": len(selected), "original": len(frame),
        "sampling_unit": manifest["sampling_unit"].iloc[0], "mode": operation_mode,
        "original_distribution": original_summary, "sampled_distribution": sampled_summary,
    }
    return manifest, details


def moderate_class_weights(
    counts: dict[str, int], minimum: float = 0.5, maximum: float = 2.0,
) -> dict[str, float]:
    positive = np.array([counts[name] for name in CLASS_NAMES if counts.get(name, 0) > 0], dtype=float)
    if positive.size == 0:
        raise ValueError("Cannot calculate weights for an empty dataset")
    median = float(np.median(positive))
    weights = {name: float(np.clip(math.sqrt(median / count), minimum, maximum))
               for name, count in counts.items() if count > 0}
    mean = float(np.mean(list(weights.values())))
    return {name: float(np.clip(value / mean, minimum, maximum)) for name, value in weights.items()}


def _moderate_targets(counts: dict[str, int], config: DatasetBalancingConfig) -> dict[str, int]:
    positive = [value for value in counts.values() if value > 0]
    if not positive:
        raise ValueError("Cannot balance an empty training split")
    median = float(np.median(positive))
    targets = {}
    for name, count in counts.items():
        if count == 0:
            targets[name] = 0
        elif count > median:
            targets[name] = max(round(median), math.ceil(count * (1 - config.max_undersample_fraction)))
        elif count < median:
            targets[name] = min(round(median), math.floor(count * config.max_oversample_factor))
        else:
            targets[name] = count
    positive_names = [name for name, value in targets.items() if value > 0]
    maximum_target = max(targets[name] for name in positive_names)
    desired_minimum = math.ceil(maximum_target / config.max_ratio)
    for name in positive_names:
        count = counts[name]
        if targets[name] < desired_minimum:
            targets[name] = min(
                desired_minimum,
                math.floor(count * config.max_oversample_factor),
            )
    minimum_target = min(targets[name] for name in positive_names)
    cap = math.floor(minimum_target * config.max_ratio)
    for name in positive_names:
        lower_bound = math.ceil(counts[name] * (1 - config.max_undersample_fraction))
        targets[name] = max(lower_bound, min(targets[name], cap))
    return targets


def balance_dataset(
    input_root: str | Path, config: DatasetManagementConfig,
    output_root: str | Path | None = None, balance_mode: str | None = None,
    splits: tuple[str, ...] | None = None, seed: int | None = None,
    overwrite: bool = False, progress_disabled: bool = True,
) -> dict[str, object]:
    """Generate moderate class weights/sampler weights or a separately materialized dataset."""
    source_root = _validate_root(input_root)
    settings = config.balancing
    mode = balance_mode or settings.mode
    target_splits = splits or settings.splits
    if any(split in {"val", "test"} for split in target_splits):
        warnings.warn("Balancing validation/test changes their real distribution", RuntimeWarning)
    frame = scan_dataset(source_root, config.extensions, progress_disabled)
    if frame.empty:
        raise ValueError("Dataset contains no supported images")
    affected = frame[frame["split"].isin(target_splits)]
    counts = {name: int(value) for name, value in affected["class_name"].value_counts().reindex(CLASS_NAMES, fill_value=0).items()}
    if any(value == 0 for value in counts.values()):
        raise ValueError(f"Cannot balance empty classes: {[name for name, value in counts.items() if value == 0]}")
    weights = moderate_class_weights(counts, settings.min_class_weight, settings.max_class_weight)
    reports = config.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "class_weights.json").write_text(json.dumps(weights, indent=2), encoding="utf-8")
    balanced = frame.copy()
    balanced["sample_weight"] = balanced.apply(
        lambda row: weights.get(row["class_name"], 1.0) if row["split"] in target_splits else 1.0,
        axis=1,
    )
    if mode == "sampler":
        balanced.to_csv(reports / "balanced_manifest.csv", index=False)
    elif mode == "class-weights":
        pass
    elif mode == "physical":
        if output_root is None:
            raise ValueError("output-root is required for physical balancing")
        destination_root = _validate_output(source_root, output_root)
        rng = np.random.default_rng(config.seed if seed is None else seed)
        targets = _moderate_targets(counts, settings)
        materialized = []
        for (split, class_name), subset in frame.groupby(["split", "class_name"], sort=True):
            if split not in target_splits:
                chosen = subset.copy()
            else:
                unit = _resolve_unit(subset, settings.sampling_unit)
                target = targets[class_name]
                if target <= len(subset):
                    chosen = subset.loc[sorted(_select_indices(subset, target, unit, rng))].copy()
                else:
                    chosen = subset.copy()
                    extras = []
                    groups = [group for _, group in subset.groupby("patient_id", sort=True)] if unit == "patient" else [subset.loc[[idx]] for idx in sorted(subset.index)]
                    order = rng.permutation(len(groups))
                    cursor = 0
                    while len(chosen) + sum(len(item) for item in extras) < target:
                        group = groups[int(order[cursor % len(order)])]
                        if len(chosen) + sum(len(item) for item in extras) + len(group) > target:
                            break
                        extras.append(group.copy())
                        cursor += 1
                    if extras:
                        chosen = pd.concat([chosen, *extras], ignore_index=True)
            chosen = chosen.reset_index(drop=True)
            chosen["duplicate_index"] = chosen.groupby("relative_path").cumcount()
            materialized.append(chosen)
        output_frame = pd.concat(materialized, ignore_index=True)
        with create_progress(description="Copiando dataset balanceado", total=len(output_frame), unit="archivo", disabled=progress_disabled) as progress:
            for row in output_frame.itertuples():
                relative = Path(row.relative_path)
                destination = destination_root / relative
                if row.duplicate_index:
                    destination = destination.with_name(f"{destination.stem}__repeat{row.duplicate_index}{destination.suffix}")
                _transfer(Path(row.source_path), destination, "copy", overwrite)
                progress.update(1)
        balanced = output_frame
    else:
        raise ValueError(f"Unknown balance mode: {mode}")
    save_distribution(frame, reports, "sampled")
    summary = save_distribution(balanced, reports, "balanced")
    balanced_affected = balanced[balanced["split"].isin(target_splits)]
    residual_counts = {
        name: int(value) for name, value in balanced_affected["class_name"].value_counts().reindex(CLASS_NAMES, fill_value=0).items()
    }
    nonzero = [value for value in residual_counts.values() if value]
    result = {
        "mode": mode, "splits": list(target_splits), "before_counts": counts,
        "class_weights": weights, "after_counts": residual_counts,
        "residual_ratio": max(nonzero) / min(nonzero) if nonzero else None,
    }
    return result
