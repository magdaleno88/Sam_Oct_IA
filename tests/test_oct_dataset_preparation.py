"""Synthetic end-to-end tests for the unified OCT preparation workflow."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from sam_ml.oct.config import OCTConfig
from sam_ml.oct.constants import CLASS_NAMES
from sam_ml.oct.preparation import PreparationError, prepare_oct_dataset


def _image(path: Path, value: int = 70) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((24, 36), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _dataset(root: Path, train_counts: dict[str, int] | None = None) -> None:
    train_counts = train_counts or {name: 8 for name in CLASS_NAMES}
    for class_index, name in enumerate(CLASS_NAMES):
        for index in range(train_counts[name]):
            _image(root / "train" / name / f"{name}-TR{class_index}{index // 2:03d}-{index % 2 + 1}.png", 50 + index)
        for index in range(3):
            _image(root / "test" / name / f"{name}-TE{class_index}{index:03d}-1.png", 80 + index)


def _config(tmp_path: Path, root: Path, output: Path, experiment: str = "synthetic") -> OCTConfig:
    config = OCTConfig()
    config.preprocessing.target_size = 32
    config.preprocessing.audit_sample_size = 2
    config.data.val_fraction = 0.5
    config.dataset_preparation.input_root = root
    config.dataset_preparation.output_root = output
    config.dataset_preparation.reports_root = tmp_path / "reports"
    config.dataset_preparation.experiment_name = experiment
    config.dataset_preparation.generated_config_dir = tmp_path / "generated"
    return config


def test_prepares_sampled_train_and_preserves_complete_test(tmp_path):
    root = tmp_path / "raw"; output = tmp_path / "prepared"; _dataset(root)
    config = _config(tmp_path, root, output)
    summary = prepare_oct_dataset(
        config, train_percentage=50, balance_mode="none", progress_disabled=True,
    )
    assert summary["selected_train"] == 16
    assert summary["final_test"] == 12
    assert summary["test_integrity"]["passed"] is True
    assert len(list((output / "train").rglob("*.png"))) == 16
    assert len(list((output / "test").rglob("*.png"))) == 12
    for name in CLASS_NAMES:
        assert len(list((output / "test" / name).glob("*.png"))) == 3
    reports = tmp_path / "reports" / "synthetic"
    for filename in (
        "selection_manifest.csv", "balance_manifest.csv", "preprocessing_report.csv",
        "preprocessing_errors.csv", "test_integrity_report.json", "preparation_summary.json",
        "preparation_summary.csv", "class_weights.json", "original_distribution.csv",
        "selected_train_distribution.csv", "balanced_train_distribution.csv",
        "final_distribution.csv",
    ):
        assert (reports / filename).exists()
    generated = yaml.safe_load((tmp_path / "generated" / "synthetic.yaml").read_text())
    assert Path(generated["data"]["root"]) == output.resolve()
    assert (reports / "manifests" / "val.csv").exists()
    from sam_ml.oct.dataset import OCTManifestDataset
    dataset = OCTManifestDataset(
        reports / "manifests" / "test.csv", preprocessing=config.preprocessing.model_copy(update={"enabled": False}),
    )
    assert len(dataset) == 12
    image, label = dataset[0]
    assert image.shape == (3, 224, 224)
    assert 0 <= label < 4


def test_selection_is_reproducible_and_keeps_patients_complete(tmp_path):
    root = tmp_path / "raw"; _dataset(root)
    first = _config(tmp_path, root, tmp_path / "one", "one")
    second = _config(tmp_path, root, tmp_path / "two", "two")
    prepare_oct_dataset(
        first, train_percentage=50, seed=9, balance_mode="none",
        generate_training_config=False, progress_disabled=True,
    )
    prepare_oct_dataset(
        second, train_percentage=50, seed=9, balance_mode="none",
        generate_training_config=False, progress_disabled=True,
    )
    one = pd.read_csv(tmp_path / "reports" / "one" / "selection_manifest.csv")
    two = pd.read_csv(tmp_path / "reports" / "two" / "selection_manifest.csv")
    assert one.loc[one.selected, "source_path"].tolist() == two.loc[two.selected, "source_path"].tolist()
    assert one.groupby("patient_id")["selected"].nunique().max() == 1


def test_physical_balance_only_changes_train_and_names_repetitions(tmp_path):
    root = tmp_path / "raw"; output = tmp_path / "prepared"
    _dataset(root, {"CNV": 12, "DME": 8, "DRUSEN": 4, "NORMAL": 10})
    config = _config(tmp_path, root, output)
    summary = prepare_oct_dataset(
        config, train_percentage=100, balance_mode="moderate-physical",
        generate_training_config=False, progress_disabled=True,
    )
    assert summary["balance"]["after_counts"] != summary["balance"]["before_counts"]
    assert len(list((output / "test").rglob("*.png"))) == 12
    assert list((output / "train").rglob("*__repeat_001.png"))


def test_rejects_silent_image_level_fallback(tmp_path):
    root = tmp_path / "raw"; output = tmp_path / "prepared"
    for name in CLASS_NAMES:
        _image(root / "train" / name / f"unknown_{name}.png")
        _image(root / "test" / name / f"test_{name}.png")
    config = _config(tmp_path, root, output)
    with pytest.raises(PreparationError, match="allow-image-level"):
        prepare_oct_dataset(config, balance_mode="none", generate_training_config=False)
    assert not output.exists()


def test_corrupt_test_prevents_publication_and_writes_error_report(tmp_path):
    root = tmp_path / "raw"; output = tmp_path / "prepared"; _dataset(root)
    corrupt = next((root / "test" / "CNV").glob("*.png"))
    corrupt.write_bytes(b"not an image")
    config = _config(tmp_path, root, output)
    with pytest.raises(PreparationError, match="integrity"):
        prepare_oct_dataset(
            config, train_percentage=100, balance_mode="none",
            generate_training_config=False, progress_disabled=True,
        )
    assert not output.exists()
    reports = tmp_path / "reports" / "synthetic"
    assert (reports / "preparation_error.json").exists()
    assert len(pd.read_csv(reports / "preprocessing_errors.csv")) == 1
