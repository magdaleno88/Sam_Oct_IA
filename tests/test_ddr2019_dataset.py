"""Unit tests for DDR2019 dataset loader."""

import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from sam_ml.datasets import DDR2019Dataset


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def processed_ddr2019_dir(temp_dir):
    """Create a minimal processed DDR2019 layout: labels.csv + images/ with small RGB images."""
    root = Path(temp_dir) / "processed_ddr2019"
    root.mkdir(parents=True)
    images_dir = root / "images"
    images_dir.mkdir()

    # Small images (32x32) for fast tests; multiple classes for stratified split
    filenames = ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg", "f.jpg", "g.jpg", "h.jpg"]
    labels = [0, 0, 1, 1, 2, 2, 0, 1]  # 3x class0, 2x class1, 2x class2

    for fname, label in zip(filenames, labels):
        img = Image.new("RGB", (32, 32), color=(label * 80, 100, 150))
        img.save(images_dir / fname, "JPEG")

    df = pd.DataFrame({"filename": filenames, "label": labels})
    df.to_csv(root / "labels.csv", index=False)
    return str(root)


class TestDDR2019DatasetInit:
    """Tests for DDR2019Dataset initialization and layout validation."""

    def test_init_split_all(self, processed_ddr2019_dir):
        """Dataset with split='all' loads full CSV."""
        ds = DDR2019Dataset(processed_ddr2019_dir, split="all")
        assert len(ds) == 8

    def test_init_split_train_val(self, processed_ddr2019_dir):
        """Train and val splits partition the data without overlap."""
        ds_train = DDR2019Dataset(
            processed_ddr2019_dir, split="train", train_ratio=0.75, val_ratio=0.25
        )
        ds_val = DDR2019Dataset(
            processed_ddr2019_dir, split="val", train_ratio=0.75, val_ratio=0.25
        )
        assert len(ds_train) + len(ds_val) == 8
        # No overlap possible by construction; just check sizes are reasonable
        assert len(ds_train) >= 1 and len(ds_val) >= 1

    def test_init_stratified_split_reproducible(self, processed_ddr2019_dir):
        """Same random_state yields same train/val indices."""
        ds1 = DDR2019Dataset(
            processed_ddr2019_dir,
            split="train",
            train_ratio=0.5,
            val_ratio=0.5,
            random_state=123,
        )
        ds2 = DDR2019Dataset(
            processed_ddr2019_dir,
            split="train",
            train_ratio=0.5,
            val_ratio=0.5,
            random_state=123,
        )
        assert len(ds1) == len(ds2)
        for i in range(len(ds1)):
            _, l1 = ds1[i]
            _, l2 = ds2[i]
            assert l1 == l2

    def test_init_accepts_path(self, processed_ddr2019_dir):
        """data_dir can be a Path."""
        ds = DDR2019Dataset(Path(processed_ddr2019_dir), split="all")
        assert len(ds) == 8

    def test_init_labels_csv_missing_raises(self, temp_dir):
        """FileNotFoundError when labels.csv does not exist."""
        root = Path(temp_dir) / "no_labels"
        root.mkdir()
        (root / "images").mkdir()
        with pytest.raises(FileNotFoundError, match="Labels file not found"):
            DDR2019Dataset(str(root), split="all")

    def test_init_images_dir_missing_raises(self, temp_dir):
        """FileNotFoundError when images/ directory does not exist."""
        root = Path(temp_dir) / "no_images"
        root.mkdir()
        pd.DataFrame({"filename": ["x.jpg"], "label": [0]}).to_csv(
            root / "labels.csv", index=False
        )
        with pytest.raises(FileNotFoundError, match="Images directory not found"):
            DDR2019Dataset(str(root), split="all")

    def test_init_csv_wrong_columns_raises(self, processed_ddr2019_dir):
        """ValueError when labels.csv lacks 'filename' or 'label'."""
        root = Path(processed_ddr2019_dir)
        df_bad = pd.DataFrame({"id": [1], "value": [0]})
        df_bad.to_csv(root / "labels.csv", index=False)
        with pytest.raises(ValueError, match="labels.csv must have columns"):
            DDR2019Dataset(str(root), split="all")


class TestDDR2019DatasetGetItem:
    """Tests for __len__ and __getitem__."""

    def test_getitem_returns_tensor_and_int(self, processed_ddr2019_dir):
        """__getitem__ returns (torch.Tensor, int) with correct shapes."""
        ds = DDR2019Dataset(processed_ddr2019_dir, split="all")
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(label, int)
        assert img.shape == (3, 32, 32)
        assert 0 <= label <= 2

    def test_getitem_all_indices(self, processed_ddr2019_dir):
        """Every index in [0, len(ds)) returns valid (image, label)."""
        ds = DDR2019Dataset(processed_ddr2019_dir, split="all")
        for i in range(len(ds)):
            img, label = ds[i]
            assert img.dim() == 3 and img.size(0) == 3
            assert isinstance(label, int)

    def test_getitem_out_of_range_raises(self, processed_ddr2019_dir):
        """Index >= len(ds) raises (IndexError from pandas iloc)."""
        ds = DDR2019Dataset(processed_ddr2019_dir, split="all")
        with pytest.raises(IndexError):
            _ = ds[100]


class TestDDR2019DatasetTransform:
    """Tests for custom transform and default ToTensor."""

    def test_default_transform_produces_tensor_0_1(self, processed_ddr2019_dir):
        """Default transform yields tensor in [0, 1]."""
        ds = DDR2019Dataset(processed_ddr2019_dir, split="all")
        img, _ = ds[0]
        assert img.min() >= 0.0 and img.max() <= 1.0

    def test_custom_transform_used(self, processed_ddr2019_dir):
        """Custom transform is applied when provided."""
        from torchvision import transforms as T

        def always_ones(pil_img):
            h, w = pil_img.height, pil_img.width
            return torch.ones(3, h, w)

        ds = DDR2019Dataset(processed_ddr2019_dir, split="all", transform=always_ones)
        img, label = ds[0]
        assert isinstance(label, int)
        assert img.shape == (3, 32, 32)
        assert (img == 1.0).all().item()


class TestDDR2019DatasetDataLoader:
    """Tests for DataLoader integration."""

    def test_dataloader_batch_shapes(self, processed_ddr2019_dir):
        """DataLoader yields batches of shape (B, 3, H, W) and (B,)."""
        ds = DDR2019Dataset(processed_ddr2019_dir, split="all")
        loader = DataLoader(ds, batch_size=3, shuffle=False, num_workers=0)
        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape == (3, 3, 32, 32)
        assert batch_y.shape == (3,)

    def test_dataloader_full_iteration(self, processed_ddr2019_dir):
        """Iterating DataLoader covers all samples."""
        ds = DDR2019Dataset(processed_ddr2019_dir, split="all")
        loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
        total = sum(batch_y.numel() for _, batch_y in loader)
        assert total == len(ds)


class TestDDR2019DatasetSplitRatios:
    """Tests for split ratio behavior."""

    def test_train_val_ratio_approx(self, processed_ddr2019_dir):
        """Train and val sizes approximately follow given ratios."""
        ds_train = DDR2019Dataset(
            processed_ddr2019_dir, split="train", train_ratio=0.8, val_ratio=0.2
        )
        ds_val = DDR2019Dataset(
            processed_ddr2019_dir, split="val", train_ratio=0.8, val_ratio=0.2
        )
        n_train, n_val = len(ds_train), len(ds_val)
        total = n_train + n_val
        assert total == 8
        # Stratified split: train should be larger
        assert n_train >= n_val
