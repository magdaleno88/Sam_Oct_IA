"""Conservative OCT preprocessing focused on border-connected white artifacts."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from sam_ml.oct.config import OCTPreprocessingConfig
from sam_ml.utils.progress import create_progress


class DatasetProcessingCancelled(KeyboardInterrupt):
    """Carry partial progress to the CLI after Ctrl+C."""

    def __init__(self, processed: int, total: int) -> None:
        super().__init__("OCT preprocessing cancelled")
        self.processed = processed
        self.total = total


@dataclass
class BackgroundStatistics:
    mean: float
    median: float
    std: float
    samples: np.ndarray = field(repr=False)


@dataclass
class PreprocessingResult:
    image: np.ndarray
    artifact_mask: np.ndarray
    corrected: np.ndarray
    cropped: np.ndarray
    padded: np.ndarray
    metadata: dict[str, object]


def load_oct_image(path: str | Path) -> np.ndarray:
    """Read a JPEG/PNG without changing its initial 0..255 intensity scale."""
    encoded = np.fromfile(Path(path), dtype=np.uint8)
    if encoded.size == 0:
        raise ValueError("empty, corrupt, or invalid image")
    try:
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    except cv2.error as exc:
        raise ValueError("empty, corrupt, or invalid image") from exc
    if image is None or image.size == 0 or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("empty, corrupt, or invalid image")
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2:
        raise ValueError(f"unsupported image shape: {image.shape}")
    if image.dtype != np.uint8:
        finite = image[np.isfinite(image)]
        if finite.size == 0:
            raise ValueError("image contains no finite pixels")
        low, high = float(finite.min()), float(finite.max())
        image = np.clip((image - low) * 255 / max(high - low, 1), 0, 255).astype(np.uint8)
    return image


def detect_border_white_artifacts(
    image: np.ndarray, config: OCTPreprocessingConfig
) -> np.ndarray:
    """Select compact near-white components touching borders and concentrated in margins."""
    candidate = (image >= config.white_threshold).astype(np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    height, width = image.shape
    min_area = max(config.min_artifact_area, round(image.size * config.min_artifact_area_fraction))
    margin_x = max(1, round(width * config.corner_margin_fraction))
    margin_y = max(1, round(height * config.corner_margin_fraction))
    accepted = np.zeros_like(candidate)
    for component in range(1, count):
        x, y, w, h, area = stats[component]
        if area < min_area:
            continue
        touches = (
            x <= config.border_width or y <= config.border_width
            or x + w >= width - config.border_width or y + h >= height - config.border_width
        )
        if not touches:
            continue
        component_mask = labels == component
        ys, xs = np.nonzero(component_mask)
        in_margin = (xs < margin_x) | (xs >= width - margin_x) | (ys < margin_y) | (ys >= height - margin_y)
        margin_fraction = float(in_margin.mean())
        near_white_fraction = float((image[component_mask] >= config.white_threshold).mean())
        if margin_fraction >= config.min_corner_margin_fraction and near_white_fraction >= config.min_near_white_fraction:
            accepted[component_mask] = 255
    return accepted


def refine_artifact_mask(mask: np.ndarray, config: OCTPreprocessingConfig) -> np.ndarray:
    """Close JPEG pinholes and slightly cover compression halos."""
    if not np.any(mask):
        return mask.copy()
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    if config.mask_dilation:
        size = 2 * config.mask_dilation + 1
        closed = cv2.dilate(closed, np.ones((size, size), np.uint8), iterations=1)
    return closed


def estimate_background_texture(
    image: np.ndarray, artifact_mask: np.ndarray, config: OCTPreprocessingConfig
) -> BackgroundStatistics:
    """Select dark, granular patches from several upper-image positions."""
    height, width = image.shape
    upper = max(config.background_patch_size, round(height * config.background_top_fraction))
    patch = min(config.background_patch_size, height, width)
    step = max(4, patch // 2)
    candidates: list[np.ndarray] = []
    for y in range(0, max(1, upper - patch + 1), step):
        for x in range(0, max(1, width - patch + 1), step):
            values = image[y:y + patch, x:x + patch]
            invalid = artifact_mask[y:y + patch, x:x + patch] > 0
            usable = values[~invalid & (values < config.white_threshold)]
            if usable.size >= values.size * 0.8:
                mean, std = float(usable.mean()), float(usable.std())
                if mean <= config.background_max_mean and std >= config.background_min_std:
                    candidates.append(usable)
    if candidates:
        samples = np.concatenate(candidates).astype(np.uint8)
    else:
        valid = image[(artifact_mask == 0) & (image < config.white_threshold)]
        if valid.size == 0:
            valid = np.array([0], dtype=np.uint8)
        cutoff = np.percentile(valid, 40)
        samples = valid[valid <= cutoff]
        if samples.size == 0:
            samples = valid
    return BackgroundStatistics(
        mean=float(samples.mean()), median=float(np.median(samples)),
        std=max(float(samples.std()), 1.0), samples=samples,
    )


def fill_with_background_texture(
    image: np.ndarray, mask: np.ndarray, background: BackgroundStatistics,
    config: OCTPreprocessingConfig, seed: int | None = None,
) -> np.ndarray:
    """Blend local inpainting with deterministic samples from the image's own background."""
    if not np.any(mask):
        return image.copy()
    rng = np.random.default_rng(config.seed if seed is None else seed)
    inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA).astype(np.float32)
    sampled = rng.choice(background.samples, size=image.shape, replace=True).astype(np.float32)
    sampled = cv2.GaussianBlur(sampled, (3, 3), 0) + rng.normal(0, background.std * 0.35, image.shape)
    texture = config.fill_inpaint_weight * inpainted + (1 - config.fill_inpaint_weight) * sampled
    distance = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    alpha = np.clip(distance / max(config.mask_dilation + 2, 2), 0, 1)
    result = image.astype(np.float32)
    result = result * (1 - alpha) + texture * alpha
    result[mask == 0] = image[mask == 0]
    return np.clip(result, 0, 255).astype(np.uint8)


def crop_safe_empty_margins(
    image: np.ndarray, config: OCTPreprocessingConfig
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Remove only low-variance, low-gradient edge strips; never centrally crop anatomy."""
    if not config.crop_empty_margins:
        return image.copy(), (0, 0, 0, 0)
    height, width = image.shape
    gradient = cv2.magnitude(
        cv2.Sobel(image, cv2.CV_32F, 1, 0), cv2.Sobel(image, cv2.CV_32F, 0, 1)
    )
    row_signal = image.std(axis=1) + gradient.mean(axis=1)
    col_signal = image.std(axis=0) + gradient.mean(axis=0)
    row_threshold = max(1.0, float(np.percentile(row_signal, 15)) * 0.5)
    col_threshold = max(1.0, float(np.percentile(col_signal, 15)) * 0.5)

    def edge_count(signal: np.ndarray, reverse: bool, limit: int, threshold: float) -> int:
        values = signal[::-1] if reverse else signal
        count = 0
        for value in values[:limit]:
            if value > threshold:
                break
            count += 1
        return count

    max_y = round(height * config.max_crop_fraction_per_side)
    max_x = round(width * config.max_crop_fraction_per_side)
    top = edge_count(row_signal, False, max_y, row_threshold)
    bottom = edge_count(row_signal, True, max_y, row_threshold)
    left = edge_count(col_signal, False, max_x, col_threshold)
    right = edge_count(col_signal, True, max_x, col_threshold)
    cropped = image[top:height - bottom if bottom else height, left:width - right if right else width]
    return cropped, (left, top, right, bottom)


def handle_extreme_aspect_ratio(
    image: np.ndarray, config: OCTPreprocessingConfig
) -> tuple[np.ndarray, list[str]]:
    """Flag panoramic scans and default to anatomy-preserving letterboxing."""
    height, width = image.shape
    ratio = max(width / height, height / width)
    if ratio <= config.max_aspect_ratio:
        return image, []
    warning = f"extreme_aspect_ratio:{ratio:.3f};mode:{config.extreme_aspect_mode}"
    if config.extreme_aspect_mode == "macular_center":
        # No reliable macula detector exists in this project; preserve anatomy instead.
        warning += ";macula_not_reliably_localized_fallback_letterbox"
    elif config.extreme_aspect_mode == "overlapping_windows":
        warning += ";window_export_requires_dataset_mode_fallback_letterbox"
    return image, [warning]


def _texture_canvas(shape: tuple[int, int], background: BackgroundStatistics, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    canvas = rng.choice(background.samples, size=shape, replace=True).astype(np.float32)
    low_frequency = cv2.GaussianBlur(canvas, (5, 5), 0)
    noise = rng.normal(0, background.std * 0.25, shape)
    return np.clip(0.65 * canvas + 0.35 * low_frequency + noise, 0, 255).astype(np.uint8)


def pad_to_square_with_texture(
    image: np.ndarray, background: BackgroundStatistics, seed: int
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Center the retina on a square canvas synthesized from its own background."""
    height, width = image.shape
    side = max(height, width)
    left = (side - width) // 2
    right = side - width - left
    top = (side - height) // 2
    bottom = side - height - top
    canvas = _texture_canvas((side, side), background, seed)
    canvas[top:top + height, left:left + width] = image
    return canvas, (left, top, right, bottom)


def resize_oct_image(image: np.ndarray, size: int) -> np.ndarray:
    interpolation = cv2.INTER_AREA if max(image.shape) > size else cv2.INTER_CUBIC
    return cv2.resize(image, (size, size), interpolation=interpolation)


def _optional_enhancements(image: np.ndarray, config: OCTPreprocessingConfig) -> np.ndarray:
    result = image
    if config.percentile_normalization:
        low, high = np.percentile(result, [config.lower_percentile, config.upper_percentile])
        result = np.clip((result.astype(float) - low) * 255 / max(high - low, 1), 0, 255).astype(np.uint8)
    if config.clahe:
        result = cv2.createCLAHE(config.clahe_clip_limit, (8, 8)).apply(result)
    if config.light_denoise == "median":
        result = cv2.medianBlur(result, 3)
    elif config.light_denoise == "bilateral":
        result = cv2.bilateralFilter(result, 5, 20, 20)
    return result


def preprocess_oct_image(
    image: np.ndarray, config: OCTPreprocessingConfig, seed: int | None = None
) -> PreprocessingResult:
    """Run cleanup -> safe crop -> textured square padding -> resize."""
    if seed is None:
        content_digest = hashlib.sha256(image.tobytes()).digest()
        actual_seed = config.seed + int.from_bytes(content_digest[:4], "little")
    else:
        actual_seed = seed
    original = image.copy()
    raw_mask = detect_border_white_artifacts(original, config)
    mask = refine_artifact_mask(raw_mask, config)
    background = estimate_background_texture(original, mask, config)
    corrected = fill_with_background_texture(original, mask, background, config, actual_seed)
    cropped, crop = crop_safe_empty_margins(corrected, config)
    aspect_input, warnings = handle_extreme_aspect_ratio(cropped, config)
    padded, padding = pad_to_square_with_texture(aspect_input, background, actual_seed + 1)
    final = resize_oct_image(_optional_enhancements(padded, config), config.target_size)
    height, width = original.shape
    metadata = {
        "original_width": width, "original_height": height,
        "cropped_width": cropped.shape[1], "cropped_height": cropped.shape[0],
        "white_area_pixels": int((mask > 0).sum()),
        "corrected_percentage": float((mask > 0).mean() * 100),
        "crop_left": crop[0], "crop_top": crop[1], "crop_right": crop[2], "crop_bottom": crop[3],
        "padding_left": padding[0], "padding_top": padding[1],
        "padding_right": padding[2], "padding_bottom": padding[3],
        "final_width": final.shape[1], "final_height": final.shape[0],
        "aspect_ratio": float(width / height), "warnings": ";".join(warnings), "status": "ok",
        "background_mean": background.mean, "background_median": background.median,
        "background_std": background.std,
    }
    return PreprocessingResult(final, mask, corrected, cropped, padded, metadata)


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix if path.suffix.lower() in {".jpg", ".jpeg", ".png"} else ".png", image)
    if not success:
        raise OSError(f"Could not encode {path}")
    encoded.tofile(path)


def save_quality_control(
    qc_dir: Path, relative: Path, original: np.ndarray, result: PreprocessingResult
) -> None:
    stem = relative.with_suffix("")
    _write_image(qc_dir / "original" / stem.with_suffix(".png"), original)
    _write_image(qc_dir / "mask" / stem.with_suffix(".png"), result.artifact_mask)
    _write_image(qc_dir / "corrected" / stem.with_suffix(".png"), result.corrected)
    _write_image(qc_dir / "padded" / stem.with_suffix(".png"), result.padded)
    _write_image(qc_dir / "final" / stem.with_suffix(".png"), result.image)


def preprocess_dataset(
    input_root: str | Path, output_root: str | Path, config: OCTPreprocessingConfig,
    progress_disabled: bool = True, show_stages: bool = False,
) -> pd.DataFrame:
    """Process recursively without overwriting originals and preserve relative class/split paths."""
    input_root, output_root = Path(input_root), Path(output_root)
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    paths = sorted(path for path in input_root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)
    if show_stages:
        from sam_ml.utils.progress import stage
        stage(f"Se encontraron {len(paths):,} imagenes.")
    rows: list[dict[str, object]] = []
    corrected_count = unchanged_count = panoramic_count = skipped_count = error_count = 0
    try:
        with create_progress(
            description="Preprocesando OCT", total=len(paths), unit="img",
            disabled=progress_disabled,
        ) as progress:
            for index, path in enumerate(paths):
                relative = path.relative_to(input_root)
                row: dict[str, object] = {"image_path": str(path), "output_path": str(output_root / relative)}
                try:
                    original = load_oct_image(path)
                    result = preprocess_oct_image(original, config, config.seed + index)
                    _write_image(output_root / relative, result.image)
                    row.update(result.metadata)
                    if int(result.metadata["white_area_pixels"]) > 0:
                        corrected_count += 1
                    else:
                        unchanged_count += 1
                    if "extreme_aspect_ratio" in str(result.metadata["warnings"]):
                        panoramic_count += 1
                    if index < config.audit_sample_size:
                        save_quality_control(config.quality_control_dir, relative, original, result)
                except Exception as exc:
                    error_count += 1
                    row.update({"status": "error", "warnings": f"{type(exc).__name__}: {exc}"})
                rows.append(row)
                progress.update(1)
                if (index + 1) % 25 == 0 or index + 1 == len(paths):
                    progress.set_postfix(
                        corrected=corrected_count, unchanged=unchanged_count,
                        panoramic=panoramic_count, skipped=skipped_count,
                        errors=error_count, refresh=False,
                    )
    except KeyboardInterrupt as exc:
        _write_preprocessing_report(rows, config)
        raise DatasetProcessingCancelled(len(rows), len(paths)) from exc
    report = _write_preprocessing_report(rows, config)
    report.attrs["summary"] = {
        "processed": corrected_count + unchanged_count,
        "corrected": corrected_count, "unchanged": unchanged_count,
        "panoramic": panoramic_count, "skipped": skipped_count, "errors": error_count,
        "total": len(paths),
    }
    return report


def _write_preprocessing_report(
    rows: list[dict[str, object]], config: OCTPreprocessingConfig,
) -> pd.DataFrame:
    """Persist full or partial results; also handles an empty dataset."""
    report = pd.DataFrame(rows)
    if report.empty:
        report = pd.DataFrame(columns=["image_path", "output_path", "status", "warnings"])
    config.quality_control_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(config.quality_control_dir / "preprocessing_report.csv", index=False)
    (config.quality_control_dir / "preprocessing_report.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return report
