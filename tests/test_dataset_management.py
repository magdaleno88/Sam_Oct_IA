"""Path-only tests for OCT sampling and moderate class balancing."""

from pathlib import Path

import pandas as pd
import pytest

from sam_ml.oct.config import DatasetManagementConfig
from sam_ml.oct.constants import CLASS_NAMES
from sam_ml.oct.dataset_management import (
    balance_dataset, distribution_summary, moderate_class_weights,
    sample_dataset, scan_dataset,
)


def _build_dataset(root: Path, train_counts: dict[str, int] | None = None) -> None:
    train_counts = train_counts or {name: 20 for name in CLASS_NAMES}
    for split in ("train", "val", "test"):
        for class_index, name in enumerate(CLASS_NAMES):
            count = train_counts[name] if split == "train" else 4
            if count % 2:
                raise ValueError("synthetic counts must be even")
            for image_index in range(count):
                patient = image_index // 2
                path = root / split / name / f"{name}-{split}{class_index}{patient:03d}-{image_index % 2 + 1}.jpeg"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"{split}-{name}-{image_index}".encode())


def _config(tmp_path: Path) -> DatasetManagementConfig:
    return DatasetManagementConfig(reports_dir=tmp_path / "reports", seed=42)


def test_distribution_counts_split_class_and_total(tmp_path):
    root = tmp_path / "raw"; _build_dataset(root)
    frame = scan_dataset(root, (".jpeg",), progress_disabled=True)
    summary = distribution_summary(frame)
    assert summary["total_images"] == 112
    assert summary["split_counts"]["train"]["CNV"] == 20
    assert summary["class_counts"]["NORMAL"] == 28
    assert summary["majority_minority_ratio"] == 1.0


def test_ten_percent_patient_sample_is_stratified_and_reproducible(tmp_path):
    root = tmp_path / "raw"; _build_dataset(root)
    config = _config(tmp_path)
    first, details = sample_dataset(root, tmp_path / "out1", 10, config, mode="manifest", seed=7)
    second, _ = sample_dataset(root, tmp_path / "out2", 10, config, mode="manifest", seed=7)
    selected = first[first["selected"]]
    assert selected.groupby(["split", "class_name"]).size().nunique() == 1
    assert selected["source_path"].tolist() == second[second["selected"]]["source_path"].tolist()
    assert details["actual_percentage"] == pytest.approx(21.4286, rel=1e-3)  # one whole patient per small stratum
    for _, group in first.groupby(["split", "patient_id"]):
        assert group["selected"].nunique() == 1


def test_different_seeds_select_different_patients(tmp_path):
    root = tmp_path / "raw"; _build_dataset(root)
    config = _config(tmp_path)
    one, _ = sample_dataset(root, tmp_path / "one", 25, config, mode="manifest", seed=1)
    two, _ = sample_dataset(root, tmp_path / "two", 25, config, mode="manifest", seed=2)
    assert set(one.loc[one.selected, "source_path"]) != set(two.loc[two.selected, "source_path"])


def test_manifest_mode_does_not_copy_and_copy_preserves_structure_content(tmp_path):
    root = tmp_path / "raw"; _build_dataset(root)
    config = _config(tmp_path)
    manifest_root = tmp_path / "manifest_only"
    sample_dataset(root, manifest_root, 10, config, mode="manifest")
    assert not manifest_root.exists()
    copy_root = tmp_path / "copied"
    manifest, _ = sample_dataset(root, copy_root, 10, config, mode="copy")
    row = manifest[manifest.selected].iloc[0]
    destination = Path(row.destination_path)
    assert destination.exists()
    assert destination.read_bytes() == Path(row.source_path).read_bytes()
    assert Path(row.source_path).exists()


@pytest.mark.parametrize("percentage", [0, -1, 100.1])
def test_invalid_percentage_rejected(tmp_path, percentage):
    root = tmp_path / "raw"; _build_dataset(root)
    with pytest.raises(ValueError, match="percentage"):
        sample_dataset(root, tmp_path / "out", percentage, _config(tmp_path), mode="manifest")


def test_output_equal_or_inside_input_rejected(tmp_path):
    root = tmp_path / "raw"; _build_dataset(root)
    with pytest.raises(ValueError):
        sample_dataset(root, root, 10, _config(tmp_path), mode="copy")
    with pytest.raises(ValueError):
        sample_dataset(root, root / "sample", 10, _config(tmp_path), mode="copy")


def test_moderate_weights_are_bounded_and_near_unit_mean():
    weights = moderate_class_weights({"CNV": 100, "DME": 30, "DRUSEN": 10, "NORMAL": 80})
    assert all(0.5 <= value <= 2.0 for value in weights.values())
    assert sum(weights.values()) / len(weights) == pytest.approx(1.0, abs=0.2)


def test_logical_balance_preserves_val_test_and_generates_reports(tmp_path):
    root = tmp_path / "raw"
    _build_dataset(root, {"CNV": 20, "DME": 10, "DRUSEN": 6, "NORMAL": 18})
    config = _config(tmp_path)
    before = scan_dataset(root, config.extensions)
    val_test_before = before[before.split.isin(["val", "test"])].groupby(["split", "class_name"]).size().to_dict()
    result = balance_dataset(root, config, balance_mode="class-weights")
    after = scan_dataset(root, config.extensions)
    val_test_after = after[after.split.isin(["val", "test"])].groupby(["split", "class_name"]).size().to_dict()
    assert val_test_before == val_test_after
    assert result["before_counts"] == result["after_counts"]
    assert (config.reports_dir / "class_weights.json").exists()
    assert (config.reports_dir / "balanced_distribution.csv").exists()


def test_physical_moderate_balance_respects_caps_and_only_changes_train(tmp_path):
    root = tmp_path / "raw"
    original_counts = {"CNV": 20, "DME": 10, "DRUSEN": 6, "NORMAL": 18}
    _build_dataset(root, original_counts)
    config = _config(tmp_path)
    output = tmp_path / "balanced"
    result = balance_dataset(root, config, output_root=output, balance_mode="physical")
    after = scan_dataset(output, config.extensions)
    for name, original in original_counts.items():
        count = result["after_counts"][name]
        assert count >= original * (1 - config.balancing.max_undersample_fraction)
        assert count <= original * config.balancing.max_oversample_factor
    for split in ("val", "test"):
        assert after[after.split == split].groupby("class_name").size().to_dict() == {name: 4 for name in CLASS_NAMES}


def test_empty_class_rejected_for_balancing(tmp_path):
    root = tmp_path / "raw"
    for name in CLASS_NAMES[:-1]:
        path = root / "train" / name / f"{name}-1-1.jpeg"
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"x")
    with pytest.raises(ValueError, match="empty classes"):
        balance_dataset(root, _config(tmp_path), balance_mode="class-weights")
