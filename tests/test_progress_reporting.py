"""Progress reporting must remain optional and scientifically transparent."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from sam_ml.oct.cli import audit_main, create_splits_main, preprocess_main
from sam_ml.oct.config import OCTPreprocessingConfig
from sam_ml.oct.preprocessing import preprocess_dataset
from sam_ml.utils.progress import create_progress, track_progress
from scripts.visualize_transforms import main as visualize_main


def _image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((32, 48), value, dtype=np.uint8))


def _config_file(tmp_path: Path) -> Path:
    path = tmp_path / "oct.yaml"
    path.write_text(yaml.safe_dump({
        "data": {"root": str(tmp_path / "raw"), "manifest_dir": str(tmp_path / "manifests")},
        "preprocessing": {
            "output_root": str(tmp_path / "processed"),
            "quality_control_dir": str(tmp_path / "qc"),
            "target_size": 32,
        },
    }), encoding="utf-8")
    return path


def test_helper_works_with_and_without_total():
    assert list(track_progress(range(3), description="test", disabled=True)) == [0, 1, 2]
    with create_progress(description="test", total=None, disabled=True) as progress:
        progress.update(1)


def test_preprocessing_updates_once_per_file(monkeypatch, tmp_path):
    source = tmp_path / "raw"
    _image(source / "CNV" / "a.jpeg", 30)
    _image(source / "DME" / "b.jpeg", 40)

    class FakeProgress:
        def __init__(self): self.updates = 0
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def update(self, amount=1): self.updates += amount
        def set_postfix(self, **_kwargs): pass

    fake = FakeProgress()
    monkeypatch.setattr("sam_ml.oct.preprocessing.create_progress", lambda **_kwargs: fake)
    config = OCTPreprocessingConfig(
        target_size=32, output_root=tmp_path / "out",
        quality_control_dir=tmp_path / "qc", audit_sample_size=0,
    )
    preprocess_dataset(source, config.output_root, config, progress_disabled=False)
    assert fake.updates == 2


def test_progress_toggle_does_not_change_generated_images(tmp_path):
    source = tmp_path / "raw"
    image = np.full((32, 48), 25, dtype=np.uint8)
    cv2.fillPoly(image, [np.array([[0, 0], [12, 0], [0, 12]], np.int32)], 255)
    path = source / "CNV" / "sample.jpeg"
    path.parent.mkdir(parents=True)
    cv2.imwrite(str(path), image)
    config = OCTPreprocessingConfig(
        target_size=32, quality_control_dir=tmp_path / "qc",
        min_artifact_area=5, min_artifact_area_fraction=0, audit_sample_size=0,
    )
    preprocess_dataset(source, tmp_path / "with", config, progress_disabled=False)
    preprocess_dataset(source, tmp_path / "without", config, progress_disabled=True)
    assert np.array_equal(
        cv2.imread(str(tmp_path / "with" / "CNV" / "sample.jpeg"), cv2.IMREAD_GRAYSCALE),
        cv2.imread(str(tmp_path / "without" / "CNV" / "sample.jpeg"), cv2.IMREAD_GRAYSCALE),
    )


def test_cli_no_progress_and_empty_dataset(tmp_path, capsys):
    config_path = _config_file(tmp_path)
    audit_main(["--config", str(config_path), "--no-progress"])
    preprocess_main(["--config", str(config_path), "--no-progress"])
    assert (tmp_path / "qc" / "preprocessing_report.csv").exists()
    assert "Se encontraron 0 imagenes" in capsys.readouterr().out


def test_split_cli_accepts_no_progress_before_empty_dataset_error(tmp_path):
    config_path = _config_file(tmp_path)
    with pytest.raises(FileNotFoundError):
        create_splits_main(["--config", str(config_path), "--no-progress"])


def test_visualization_accepts_no_progress_and_empty_manifest(tmp_path):
    config_path = _config_file(tmp_path)
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True)
    pd.DataFrame(columns=["image_path"]).to_csv(manifest_dir / "train.csv", index=False)
    assert visualize_main([
        "--config", str(config_path), "--no-progress",
        "--output", str(tmp_path / "samples.png"),
    ]) == 0
