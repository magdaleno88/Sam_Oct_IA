"""Dataset discovery, quality audit, split creation, and leakage checks."""

from __future__ import annotations

import csv
import hashlib
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from sam_ml.oct.config import OCTConfig
from sam_ml.oct.constants import CLASS_NAMES, CLASS_TO_INDEX
from sam_ml.utils.progress import create_progress, stage, track_progress

IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff"}
SPLIT_ALIASES = {"train": "train", "val": "val", "validation": "val", "test": "test"}


def infer_patient_id(path: str | Path) -> str | None:
    """Extract the Kermany case token, e.g. `CNV-1016042-1.jpeg` -> `CNV-1016042`.

    The function deliberately returns None for names without the documented class-case-image
    pattern instead of inventing an identifier.
    """
    stem = Path(path).stem
    match = re.match(r"^(CNV|DME|DRUSEN|NORMAL)[-_]([A-Za-z0-9]+)[-_]\d+$", stem, re.I)
    return f"{match.group(1).upper()}-{match.group(2)}" if match else None


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_images(root: str | Path, progress_disabled: bool = True) -> pd.DataFrame:
    """Discover official split or flat class layouts without reading pixels."""
    root = Path(root)
    rows: list[dict[str, object]] = []
    official = any((root / split).is_dir() for split in SPLIT_ALIASES)
    search_roots = []
    if official:
        search_roots = [
            (folder, split)
            for folder, split in SPLIT_ALIASES.items()
            if (root / folder).is_dir()
        ]
    else:
        search_roots = [("", "unassigned")]

    with create_progress(
        description="Buscando imagenes OCT", unit="archivo",
        disabled=progress_disabled,
    ) as progress:
        for folder, split in search_roots:
            base = root / folder
            for label in CLASS_NAMES:
                class_dir = base / label
                if not class_dir.is_dir():
                    continue
                for path in class_dir.rglob("*"):
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                        rows.append({
                            "image_path": path.as_posix(),
                            "label": label,
                            "class_index": CLASS_TO_INDEX[label],
                            "patient_id": infer_patient_id(path),
                            "source": "kermany_ucsd",
                            "split": split,
                        })
                        progress.update(1)
    return pd.DataFrame(rows, columns=[
        "image_path", "label", "class_index", "patient_id", "source", "split"
    ])


def _read_exclusions(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(path)
    return set(frame.get("image_path", pd.Series(dtype=str)).astype(str))


def audit_dataset(
    config: OCTConfig, blur_threshold: float | None = None,
    progress_disabled: bool = True, show_stages: bool = False,
) -> dict[str, object]:
    """Inspect every image and write a non-destructive quality report."""
    if show_stages:
        stage("Buscando imagenes...")
    frame = discover_images(config.data.root, progress_disabled=progress_disabled)
    if show_stages:
        stage(f"Se encontraron {len(frame):,} imagenes.")
        stage("Auditando archivos OCT...")
    config.data.manifest_dir.mkdir(parents=True, exist_ok=True)
    quality_rows = []
    hashes: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    formats: Counter[str] = Counter()

    records = frame.to_dict("records")
    progress = create_progress(
        description="Auditando OCT", total=len(records), unit="img",
        disabled=progress_disabled,
    )
    valid_count = invalid_count = warning_count = 0
    for index, row in enumerate(records, start=1):
        path = Path(str(row["image_path"]))
        status = "ok"
        width = height = channels = 0
        blur_score: float | None = None
        digest = ""
        reason = ""
        try:
            digest = sha256_file(path)
            hashes[digest] += 1
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
                channels = len(image.getbands())
                formats[str(image.format or path.suffix.lstrip(".")).upper()] += 1
                gray = np.asarray(image.convert("L"))
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            dimensions[f"{width}x{height}"] += 1
            if min(width, height) < config.data.image_size:
                status, reason = "suspect", "dimension_below_model_input"
            if blur_threshold is not None and blur_score < blur_threshold:
                status = "suspect"
                reason = ";".join(filter(None, [reason, "low_laplacian_variance"]))
        except Exception as exc:
            status, reason = "corrupt", f"{type(exc).__name__}: {exc}"
        if status == "corrupt":
            invalid_count += 1
        elif status == "suspect":
            warning_count += 1
        else:
            valid_count += 1
        quality_rows.append({
            **row, "sha256": digest, "width": width, "height": height,
            "channels": channels, "blur_score": blur_score, "status": status,
            "reason": reason,
        })
        progress.update(1)
        if index % 25 == 0 or index == len(records):
            progress.set_postfix(
                valid=valid_count, invalid=invalid_count, warnings=warning_count,
                refresh=False,
            )
    progress.close()
    if show_stages:
        stage("Validando estructura...")

    quality = pd.DataFrame(quality_rows)
    if not quality.empty:
        quality["exact_duplicate"] = quality["sha256"].map(hashes).fillna(0).gt(1)
    quality_path = config.data.manifest_dir / "quality_report.csv"
    if show_stages:
        stage("Generando reporte...")
    quality.to_csv(quality_path, index=False)
    if not config.data.exclusions_file.exists():
        config.data.exclusions_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["image_path", "reason", "reviewed_by"]).to_csv(
            config.data.exclusions_file, index=False
        )
    return {
        "total_images": len(frame),
        "images_per_class": frame["label"].value_counts().to_dict() if not frame.empty else {},
        "patients_per_class": frame.dropna(subset=["patient_id"]).groupby("label")["patient_id"].nunique().to_dict() if not frame.empty else {},
        "corrupt_images": int((quality.get("status") == "corrupt").sum()) if not quality.empty else 0,
        "suspect_images": int((quality.get("status") == "suspect").sum()) if not quality.empty else 0,
        "exact_duplicate_groups": sum(count > 1 for count in hashes.values()),
        "dimensions": dict(dimensions),
        "formats": dict(formats),
        "quality_report": str(quality_path),
    }


def _group_holdout(frame: pd.DataFrame, fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use StratifiedGroupKFold for a reproducible, patient-level stratified holdout."""
    groups_per_class = frame.groupby("label")["patient_id"].nunique()
    min_groups = int(groups_per_class.min())
    if min_groups < 2:
        raise ValueError("Patient-level split requires at least two patient groups per class")
    requested_splits = max(2, round(1 / fraction))
    n_splits = min(requested_splits, min_groups)
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed,
    )
    train_idx, holdout_idx = next(
        splitter.split(frame, frame["label"], frame["patient_id"])
    )
    return frame.iloc[train_idx].copy(), frame.iloc[holdout_idx].copy()


def _image_holdout(frame: pd.DataFrame, fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    warnings.warn("image-level split, risk of leakage", RuntimeWarning, stacklevel=2)
    train_idx, test_idx = train_test_split(
        np.arange(len(frame)), test_size=fraction, random_state=seed, stratify=frame["label"]
    )
    return frame.iloc[train_idx].copy(), frame.iloc[test_idx].copy()


def validate_no_leakage(
    manifests: dict[str, pd.DataFrame], require_patients: bool = True,
    progress_disabled: bool = True,
) -> None:
    """Fail on patient or exact-file overlap between splits."""
    names = list(manifests)
    hash_by_split: dict[str, set[str]] = {}
    total = sum(len(frame) for frame in manifests.values())
    progress = create_progress(
        description="Validando splits", total=total, unit="registro",
        disabled=progress_disabled,
    )
    for name, frame in manifests.items():
        hashes = set()
        for path in frame["image_path"]:
            hashes.add(sha256_file(path))
            progress.update(1)
        hash_by_split[name] = hashes
    progress.close()
    for i, left_name in enumerate(names):
        left = manifests[left_name]
        for right_name in names[i + 1:]:
            right = manifests[right_name]
            common_paths = set(left["image_path"]) & set(right["image_path"])
            if common_paths:
                raise ValueError(f"Image paths overlap between {left_name} and {right_name}")
            if hash_by_split[left_name] & hash_by_split[right_name]:
                raise ValueError(f"Exact image duplicates occur between {left_name} and {right_name}")
            left_patients = set(left["patient_id"].dropna().astype(str))
            right_patients = set(right["patient_id"].dropna().astype(str))
            if left_patients & right_patients:
                raise ValueError(f"Patient leakage between {left_name} and {right_name}")
    if require_patients and any(frame["patient_id"].isna().any() for frame in manifests.values()):
        raise ValueError("Reliable patient_id is unavailable for one or more images")


def create_manifests(
    config: OCTConfig, progress_disabled: bool = True, show_stages: bool = False,
) -> dict[str, pd.DataFrame]:
    """Create train/val/test CSVs while preserving an official test split."""
    if show_stages:
        stage("Buscando imagenes y extrayendo identificadores de pacientes...")
    frame = discover_images(config.data.root, progress_disabled=progress_disabled)
    if show_stages:
        stage(f"Se encontraron {len(frame):,} registros.")
    manifests = split_dataset_frame(
        config, frame, progress_disabled=progress_disabled, show_stages=show_stages,
    )
    config.data.manifest_dir.mkdir(parents=True, exist_ok=True)
    if show_stages:
        stage("Escribiendo manifiestos...")
    for name, subset in manifests.items():
        subset.to_csv(config.data.manifest_dir / f"{name}.csv", index=False)
    return manifests


def split_dataset_frame(
    config: OCTConfig,
    frame: pd.DataFrame,
    progress_disabled: bool = True,
    show_stages: bool = False,
) -> dict[str, pd.DataFrame]:
    """Split a discovered dataset entirely in memory without modifying its files."""
    if frame.empty:
        raise FileNotFoundError(f"No OCT images found under {config.data.root}")
    exclusions = _read_exclusions(config.data.exclusions_file)
    frame = frame[~frame["image_path"].isin(exclusions)].reset_index(drop=True)
    has_patients = frame["patient_id"].notna().all()
    grouped = config.data.patient_level_split and has_patients
    if config.data.patient_level_split and not has_patients and not config.data.allow_image_level_split:
        raise ValueError(
            "Patient IDs could not be inferred reliably. Set allow_image_level_split=true "
            "only to accept the documented leakage risk."
        )
    split_fn = _group_holdout if grouped else _image_holdout
    if show_stages:
        stage("Asignando pacientes a train, validation y test...")

    official_test = frame[frame["split"] == "test"].copy()
    official_val = frame[frame["split"] == "val"].copy()
    if not official_test.empty:
        train_pool = frame[frame["split"] == "train"].copy()
        if train_pool.empty:
            raise ValueError("An official test split exists, but no train split was found")
        if official_val.empty:
            train, val = split_fn(train_pool, config.data.val_fraction, config.data.seed)
        else:
            train, val = train_pool, official_val
        test = official_test
    else:
        pool = frame.copy()
        train_val, test = split_fn(pool, config.data.test_fraction, config.data.seed)
        relative_val = config.data.val_fraction / (1 - config.data.test_fraction)
        train, val = split_fn(train_val, relative_val, config.data.seed + 1)

    manifests = {"train": train, "val": val, "test": test}
    for name, subset in manifests.items():
        subset.loc[:, "split"] = name
        subset.loc[:, "source"] = (
            subset["source"].astype(str) + (";patient-level" if grouped else ";image-level split, risk of leakage")
        )
    validate_no_leakage(
        manifests, require_patients=grouped,
        progress_disabled=progress_disabled,
    )
    return manifests


def load_dataset_splits(
    config: OCTConfig,
    progress_disabled: bool = True,
) -> tuple[dict[str, pd.DataFrame], str]:
    """Load existing manifests, or discover and split train/test entirely in memory.

    Existing train/val/test CSVs retain priority for backward compatibility. When none exist,
    the dataset must expose train and test directories; validation is derived only from train.
    """
    paths = {
        name: config.data.manifest_dir / f"{name}.csv"
        for name in ("train", "val", "test")
    }
    existing = {name for name, path in paths.items() if path.exists()}
    if existing:
        if existing != set(paths):
            missing = sorted(set(paths) - existing)
            raise FileNotFoundError(
                "Incomplete manifest set. Remove the partial manifests to use automatic "
                f"in-memory splitting, or create the missing files: {missing}"
            )
        manifests = {name: pd.read_csv(path) for name, path in paths.items()}
        validate_no_leakage(
            manifests,
            require_patients=(
                config.data.patient_level_split and not config.data.allow_image_level_split
            ),
            progress_disabled=progress_disabled,
        )
        return manifests, "manifest"

    frame = discover_images(config.data.root, progress_disabled=progress_disabled)
    discovered_splits = set(frame["split"]) if not frame.empty else set()
    if "train" not in discovered_splits or "test" not in discovered_splits:
        raise ValueError(
            "Automatic in-memory splitting requires dataset directories train/ and test/. "
            "Existing train.csv, val.csv and test.csv can be used for other layouts."
        )
    unexpected = discovered_splits - {"train", "test"}
    if unexpected:
        raise ValueError(
            "Automatic mode expects only train/ and test/ directories; "
            f"found additional splits: {sorted(unexpected)}"
        )
    return (
        split_dataset_frame(config, frame, progress_disabled=progress_disabled),
        "in-memory",
    )


def manifest_sha256(path: str | Path) -> str:
    return sha256_file(path)


def grouped_cross_validation_folds(
    manifest: pd.DataFrame, n_splits: int = 10, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create strict stratified participant folds, reducing count when required."""
    if manifest["patient_id"].isna().any():
        raise ValueError("Cross-validation requires reliable patient_id for every image")
    available = int(manifest.groupby("label")["patient_id"].nunique().min())
    actual = min(n_splits, available)
    if actual < 2:
        raise ValueError("At least two patient groups per class are required")
    if actual < n_splits:
        warnings.warn(f"Reducing n_splits from {n_splits} to {actual}: insufficient groups per class")
    splitter = StratifiedGroupKFold(n_splits=actual, shuffle=True, random_state=seed)
    return list(splitter.split(manifest, manifest["label"], manifest["patient_id"]))
