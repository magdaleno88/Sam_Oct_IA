"""PyTorch Lightning wrapper for the dual-channel diabetic retinopathy model."""

from __future__ import annotations

from typing import Any, Sequence, Tuple

import torch
from torchvision.models import Inception_V3_Weights, VGG16_Weights

from sam_ml.modeling.models.base import BaseLightningModel
from sam_ml.modeling.models.dual_channel_model import DualChannelDiabeticRetinopathyModel
from sam_ml.modeling.models.registry import register_model


IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def _normalize_chw(
    x: torch.Tensor,
    *,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    """Normalize a BCHW tensor by per-channel mean/std."""
    if x.dim() != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(x.shape)}")
    mean_t = torch.as_tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    std_t = torch.as_tensor(std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return (x - mean_t) / std_t


@register_model("dual_channel")
def create_dual_channel_model(
    num_classes: int | None = None,
    learning_rate: float | None = None,
    optimizer: str | None = None,
    weight_decay: float | None = None,
    use_pretrained: bool = True,
    **kwargs: Any,
) -> "DualChannelModelLightning":
    """Factory registered under key `dual_channel`."""
    return DualChannelModelLightning(
        num_classes=num_classes,
        learning_rate=learning_rate,
        optimizer=optimizer,
        weight_decay=weight_decay,
        use_pretrained=use_pretrained,
        **kwargs,
    )


class DualChannelModelLightning(BaseLightningModel):
    """Lightning wrapper with internal per-branch normalization."""

    def __init__(
        self,
        num_classes: int | None = None,
        learning_rate: float | None = None,
        optimizer: str | None = None,
        weight_decay: float | None = None,
        use_pretrained: bool = True,
        **kwargs: Any,
    ) -> None:
        self.use_pretrained = bool(use_pretrained)

        super().__init__(
            num_classes=num_classes,
            learning_rate=learning_rate,
            optimizer=optimizer,
            weight_decay=weight_decay,
            **kwargs,
        )

    def _create_model(self) -> None:
        self.model = DualChannelDiabeticRetinopathyModel(
            num_classes=self.num_classes,
            use_pretrained=self.use_pretrained,
        )

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model not initialized. Call _create_model() first.")

        x_clahe, x_ceced = x

        # Inputs come from datasets as [0, 1] float tensors (ToTensor).
        x_clahe = x_clahe.float().clamp(0.0, 1.0)
        x_ceced = x_ceced.float().clamp(0.0, 1.0)

        # Normalize using torchvision weights metadata (falls back to ImageNet constants).
        if self.use_pretrained:
            clahe_mean = Inception_V3_Weights.DEFAULT.meta.get("mean", IMAGENET_MEAN)
            clahe_std = Inception_V3_Weights.DEFAULT.meta.get("std", IMAGENET_STD)
            ceced_mean = VGG16_Weights.DEFAULT.meta.get("mean", IMAGENET_MEAN)
            ceced_std = VGG16_Weights.DEFAULT.meta.get("std", IMAGENET_STD)
        else:
            clahe_mean, clahe_std = IMAGENET_MEAN, IMAGENET_STD
            ceced_mean, ceced_std = IMAGENET_MEAN, IMAGENET_STD

        x_clahe = _normalize_chw(x_clahe, mean=clahe_mean, std=clahe_std)
        x_ceced = _normalize_chw(x_ceced, mean=ceced_mean, std=ceced_std)

        return self.model((x_clahe, x_ceced))

