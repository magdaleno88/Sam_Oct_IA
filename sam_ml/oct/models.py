"""Baseline and dilation-aware ResNet50 models for four-class OCT classification."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class ImprovedResNet50(nn.Module):
    """ImageNet-compatible ResNet50 with optional official stride replacement.

    The torchvision backbone already contains global average pooling, batch normalization,
    and residual shortcuts. Softmax is intentionally excluded from forward().
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        replace_stride_with_dilation: tuple[bool, bool, bool] = (False, True, True),
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(
            weights=weights,
            replace_stride_with_dilation=list(replace_stride_with_dilation),
        )
        features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(features, num_classes))
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            for parameter in self.backbone.fc.parameters():
                parameter.requires_grad = True

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs)

    def predict_proba(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self(inputs), dim=1)

    def unfreeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True


def baseline_resnet50(
    num_classes: int = 4,
    pretrained: bool = True,
    dropout: float = 0.2,
    freeze_backbone: bool = False,
) -> ImprovedResNet50:
    return ImprovedResNet50(
        num_classes=num_classes,
        pretrained=pretrained,
        replace_stride_with_dilation=(False, False, False),
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )


def improved_resnet50(**kwargs) -> ImprovedResNet50:
    return ImprovedResNet50(**kwargs)


def create_oct_model(name: str, **kwargs) -> ImprovedResNet50:
    factories = {"baseline_resnet50": baseline_resnet50, "improved_resnet50": improved_resnet50}
    if name not in factories:
        raise KeyError(f"Unknown OCT model {name!r}; available: {sorted(factories)}")
    return factories[name](**kwargs)
