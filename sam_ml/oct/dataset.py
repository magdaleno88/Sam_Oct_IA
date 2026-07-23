"""Manifest-backed OCT dataset and clinically conservative transforms."""

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from sam_ml.oct.config import OCTPreprocessingConfig
from sam_ml.oct.preprocessing import load_oct_image, preprocess_oct_image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_oct_transform(
    training: bool,
    image_size: int = 224,
    crop_scale: tuple[float, float] = (0.9, 1.0),
    gaussian_blur_probability: float = 0.0,
) -> transforms.Compose:
    """Build train or deterministic evaluation preprocessing."""
    common = [transforms.Grayscale(num_output_channels=3)]
    if training:
        common.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(
                image_size, scale=crop_scale, ratio=(0.95, 1.05),
                interpolation=InterpolationMode.BILINEAR,
            ),
        ])
        if gaussian_blur_probability > 0:
            common.append(transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
                p=gaussian_blur_probability,
            ))
    else:
        resize_size = round(image_size / 0.875)
        common.extend([
            transforms.Resize(resize_size, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
        ])
    common.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(common)


class OCTManifestDataset(Dataset):
    """Read OCT images and labels from a generated manifest."""

    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        transform: Callable | None = None,
        return_metadata: bool = False,
        preprocessing: OCTPreprocessingConfig | None = None,
    ) -> None:
        self.rows = pd.read_csv(manifest) if not isinstance(manifest, pd.DataFrame) else manifest.copy()
        required = {"image_path", "label", "class_index", "split"}
        if not required.issubset(self.rows.columns):
            raise ValueError(f"Manifest missing columns: {required - set(self.rows.columns)}")
        self.transform = transform or build_oct_transform(training=False)
        self.return_metadata = return_metadata
        self.preprocessing = preprocessing

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        path = Path(str(row["image_path"]))
        try:
            if self.preprocessing is not None and self.preprocessing.enabled:
                cleaned = preprocess_oct_image(
                    load_oct_image(path), self.preprocessing,
                ).image
                image = Image.fromarray(cleaned, mode="L")
            else:
                with Image.open(path) as source:
                    image = source.convert("L")
            tensor = self.transform(image)
        except Exception as exc:
            raise RuntimeError(f"Unable to read OCT image {path}: {exc}") from exc
        label = int(row["class_index"])
        if self.return_metadata:
            return tensor, label, row.to_dict()
        return tensor, label


def denormalize_oct(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)
