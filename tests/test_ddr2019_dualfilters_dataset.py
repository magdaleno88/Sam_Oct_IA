"""Unit tests for DDR2019 dual-filters dataset loader."""

import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from sam_ml.datasets import DDR2019DualFiltersDataset


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def processed_ddr2019_dual_dir(temp_dir):
    """Create a processed dual-filter layout with synchronized CLAHE and CECED files."""
    root = Path(temp_dir) / "processed_ddr2019_dualfilters"
    root.mkdir(parents=True)
    clahe_dir = root / "images_clahe"
    ceced_dir = root / "images_ceced"
    clahe_dir.mkdir()
    ceced_dir.mkdir()

    filenames = ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg", "f.jpg", "g.jpg", "h.jpg"]
    labels = [0, 0, 1, 1, 2, 2, 0, 1]

    for fname, label in zip(filenames, labels):
        clahe_img = Image.new("RGB", (299, 299), color=(label * 80, 100, 150))
        ceced_img = Image.new("RGB", (224, 224), color=(50, label * 60, 180))
        clahe_img.save(clahe_dir / fname, "JPEG")
        ceced_img.save(ceced_dir / fname, "JPEG")

    df = pd.DataFrame({
        "clahe_path": [f"images_clahe/{f}" for f in filenames],
        "ceced_path": [f"images_ceced/{f}" for f in filenames],
        "label": labels,
    })
    df.to_csv(root / "labels_dual.csv", index=False)
    return str(root)


class TestDDR2019DualFiltersDatasetInit:
    """Tests for dataset initialization and layout validation."""

    def test_init_split_all(self, processed_ddr2019_dual_dir):
        ds = DDR2019DualFiltersDataset(processed_ddr2019_dual_dir, split="all")
        assert len(ds) == 8

    def test_init_split_train_val(self, processed_ddr2019_dual_dir):
        ds_train = DDR2019DualFiltersDataset(
            processed_ddr2019_dual_dir, split="train", train_ratio=0.75, val_ratio=0.25
        )
        ds_val = DDR2019DualFiltersDataset(
            processed_ddr2019_dual_dir, split="val", train_ratio=0.75, val_ratio=0.25
        )
        assert len(ds_train) + len(ds_val) == 8
        assert len(ds_train) >= 1 and len(ds_val) >= 1

    def test_init_accepts_path(self, processed_ddr2019_dual_dir):
        ds = DDR2019DualFiltersDataset(Path(processed_ddr2019_dual_dir), split="all")
        assert len(ds) == 8

    def test_init_labels_csv_missing_raises(self, temp_dir):
        root = Path(temp_dir) / "no_labels"
        root.mkdir()
        (root / "images_clahe").mkdir()
        (root / "images_ceced").mkdir()
        with pytest.raises(FileNotFoundError, match="Labels file not found"):
            DDR2019DualFiltersDataset(str(root), split="all")

    def test_init_csv_wrong_columns_raises(self, processed_ddr2019_dual_dir):
        root = Path(processed_ddr2019_dual_dir)
        df_bad = pd.DataFrame({"x": [1], "y": [2], "label": [0]})
        df_bad.to_csv(root / "labels_dual.csv", index=False)
        with pytest.raises(ValueError, match="labels_dual.csv must have columns"):
            DDR2019DualFiltersDataset(str(root), split="all")


class TestDDR2019DualFiltersDatasetGetItem:
    """Tests for __getitem__ data contract."""

    def test_getitem_returns_pair_and_int(self, processed_ddr2019_dual_dir):
        ds = DDR2019DualFiltersDataset(processed_ddr2019_dual_dir, split="all")
        (clahe_img, ceced_img), label = ds[0]
        assert isinstance(clahe_img, torch.Tensor)
        assert isinstance(ceced_img, torch.Tensor)
        assert isinstance(label, int)
        assert clahe_img.shape == (3, 299, 299)
        assert ceced_img.shape == (3, 224, 224)

    def test_getitem_missing_clahe_image_raises(self, processed_ddr2019_dual_dir):
        ds = DDR2019DualFiltersDataset(processed_ddr2019_dual_dir, split="all")
        first_clahe_path = Path(processed_ddr2019_dual_dir) / "images_clahe" / "a.jpg"
        first_clahe_path.unlink()
        with pytest.raises(FileNotFoundError, match="CLAHE image not found"):
            _ = ds[0]

    def test_getitem_missing_ceced_image_raises(self, processed_ddr2019_dual_dir):
        ds = DDR2019DualFiltersDataset(processed_ddr2019_dual_dir, split="all")
        first_ceced_path = Path(processed_ddr2019_dual_dir) / "images_ceced" / "a.jpg"
        first_ceced_path.unlink()
        with pytest.raises(FileNotFoundError, match="CECED image not found"):
            _ = ds[0]


class TestDDR2019DualFiltersDatasetDataLoader:
    """Tests for DataLoader integration."""

    def test_dataloader_batch_shapes(self, processed_ddr2019_dual_dir):
        ds = DDR2019DualFiltersDataset(processed_ddr2019_dual_dir, split="all")
        loader = DataLoader(ds, batch_size=3, shuffle=False, num_workers=0)
        (clahe_batch, ceced_batch), labels = next(iter(loader))
        assert clahe_batch.shape == (3, 3, 299, 299)
        assert ceced_batch.shape == (3, 3, 224, 224)
        assert labels.shape == (3,)
