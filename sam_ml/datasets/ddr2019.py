"""PyTorch dataset loader for the processed DDR2019 (Diabetic Retinopathy) dataset."""

from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    from torchvision import transforms
except ImportError:
    transforms = None  # type: ignore[assignment]


def _stratified_split(
    df: pd.DataFrame,
    split: Literal["train", "val"],
    train_ratio: float,
    val_ratio: float,
    random_state: int,
) -> pd.DataFrame:
    """Split DataFrame by label so train/val preserve class proportions."""
    train_parts = []
    val_parts = []
    for _, group in df.groupby("label", group_keys=False):
        n = len(group)
        n_train = max(1, int(n * train_ratio)) if train_ratio > 0 else 0
        n_val = n - n_train
        if n_val == 0:
            n_train, n_val = n, 0
        shuffled = group.sample(frac=1, random_state=random_state)
        train_parts.append(shuffled.iloc[:n_train])
        if n_val > 0:
            val_parts.append(shuffled.iloc[n_train:])
    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame()
    return train_df if split == "train" else val_df


def _default_transform():
    """Default transform: PIL RGB to tensor (C, H, W), scale [0, 1]."""
    if transforms is None:
        raise ImportError("torchvision is required for default transform")
    return transforms.Compose([
        transforms.ToTensor(),
    ])


class DDR2019Dataset(Dataset):
    """PyTorch Dataset for the processed DDR2019 dataset.

    Expects the processed layout:
        data_dir/
            labels.csv   # columns: filename, label
            images/      # <filename>.jpg (e.g. 512x512 RGB)

    Supports train/val splits via stratified random splitting.
    """

    LABELS_CSV = "labels.csv"
    IMAGES_SUBDIR = "images"

    def __init__(
        self,
        data_dir: str | Path,
        split: Literal["train", "val", "all"] = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
        transform: Optional[object] = None,
        random_state: int = 42,
    ) -> None:
        """Initialize the DDR2019 dataset.

        Args:
            data_dir: Path to the processed DDR2019 directory (contains labels.csv and images/).
            split: "train", "val", or "all". "all" uses the full dataset.
            train_ratio: Fraction of data for train when split in ("train", "val"). Ignored if split=="all".
            val_ratio: Fraction of data for val when split in ("train", "val"). Ignored if split=="all".
            transform: Optional callable (e.g. torchvision.transforms). If None, uses ToTensor().
            random_state: Random seed for reproducible train/val split.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.random_state = random_state

        labels_path = self.data_dir / self.LABELS_CSV
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        self.images_dir = self.data_dir / self.IMAGES_SUBDIR
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        df = pd.read_csv(labels_path)
        if not {"filename", "label"}.issubset(df.columns):
            raise ValueError(
                f"labels.csv must have columns 'filename' and 'label'; got {list(df.columns)}"
            )
        df = df.astype({"filename": str, "label": int})

        if split == "all":
            self._rows = df.reset_index(drop=True)
        else:
            self._rows = _stratified_split(
                df, split, train_ratio, val_ratio, random_state
            )

        self.transform = transform if transform is not None else _default_transform()

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self._rows.iloc[idx]
        filename = row["filename"]
        label = int(row["label"])

        image_path = self.images_dir / filename
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, label
