"""Synthetic safety tests for OCT border-artifact preprocessing."""

from pathlib import Path

import cv2
import numpy as np

from sam_ml.oct.config import OCTPreprocessingConfig
from sam_ml.oct.preprocessing import (
    detect_border_white_artifacts,
    load_oct_image,
    preprocess_dataset,
    preprocess_oct_image,
)


def _synthetic_oct(size: tuple[int, int] = (128, 192), seed: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    height, width = size
    image = np.clip(rng.normal(24, 6, size), 0, 255).astype(np.uint8)
    for offset, intensity in ((0, 85), (5, 125), (10, 175), (15, 105)):
        y = round(height * 0.62) + offset
        cv2.ellipse(image, (width // 2, y), (width // 2 - 12, 12), 0, 180, 360, intensity, 2)
    return image


def _config(target: int = 128) -> OCTPreprocessingConfig:
    return OCTPreprocessingConfig(
        target_size=target, min_artifact_area=20, min_artifact_area_fraction=0,
        mask_dilation=1, crop_empty_margins=False, seed=17,
    )


def test_border_triangle_removed_but_internal_bright_region_preserved():
    image = _synthetic_oct()
    cv2.fillPoly(image, [np.array([[0, 0], [55, 0], [0, 45]], np.int32)], 255)
    cv2.circle(image, (96, 75), 7, 255, -1)
    config = _config()
    mask = detect_border_white_artifacts(image, config)
    assert mask[4, 4] == 255
    assert mask[75, 96] == 0
    result = preprocess_oct_image(image, config)
    assert result.image.shape == (128, 128)
    assert result.corrected[4, 4] < 245
    assert result.corrected[75, 96] == 255


def test_panorama_is_letterboxed_and_flagged_without_cropping_retina():
    image = _synthetic_oct((64, 256))
    config = _config(96)
    result = preprocess_oct_image(image, config)
    assert result.padded.shape == (256, 256)
    assert result.metadata["padding_top"] == 96
    assert "extreme_aspect_ratio" in result.metadata["warnings"]
    assert result.metadata["cropped_width"] == 256


def test_same_seed_and_content_are_deterministic():
    image = _synthetic_oct()
    cv2.fillPoly(image, [np.array([[0, 0], [35, 0], [0, 30]], np.int32)], 255)
    first = preprocess_oct_image(image, _config()).image
    second = preprocess_oct_image(image, _config()).image
    assert np.array_equal(first, second)


def test_clean_square_image_changes_minimally():
    image = _synthetic_oct((128, 128))
    result = preprocess_oct_image(image, _config(128))
    assert result.metadata["white_area_pixels"] == 0
    assert np.mean(np.abs(result.image.astype(float) - image.astype(float))) < 0.01


def test_dataset_preserves_structure_and_records_corrupt_files(tmp_path):
    source = tmp_path / "raw"
    destination = tmp_path / "processed"
    valid = source / "train" / "CNV" / "CNV-1-1.jpeg"
    corrupt = source / "train" / "DME" / "DME-2-1.jpeg"
    valid.parent.mkdir(parents=True)
    corrupt.parent.mkdir(parents=True)
    cv2.imwrite(str(valid), _synthetic_oct())
    corrupt.write_bytes(b"corrupt")
    config = _config()
    config.quality_control_dir = tmp_path / "qc"
    report = preprocess_dataset(source, destination, config)
    assert (destination / "train" / "CNV" / valid.name).exists()
    assert not (destination / "train" / "DME" / corrupt.name).exists()
    assert set(report["status"]) == {"ok", "error"}
    assert "train/CNV" in report.loc[report["status"] == "ok", "output_path"].iloc[0].replace("\\", "/")


def test_load_oct_image_rejects_corrupt_file(tmp_path):
    path = tmp_path / "bad.jpeg"
    path.write_bytes(b"")
    try:
        load_oct_image(path)
    except ValueError as exc:
        assert "corrupt" in str(exc)
    else:
        raise AssertionError("corrupt image was accepted")
