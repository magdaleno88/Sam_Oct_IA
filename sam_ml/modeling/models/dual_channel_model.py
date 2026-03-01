"""Dual-channel model architecture for diabetic retinopathy classification.

This file contains *only* the torch.nn.Module architecture.

Pipeline integration (PyTorch Lightning + normalization + registry) lives in a separate
Lightning wrapper module to match SAM-AI's model patterns.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import Inception_V3_Weights, VGG16_Weights


class Channel1Branch(nn.Module):
    """CLAHE branch: InceptionV3 backbone -> projection to 500-dim."""

    def __init__(
        self,
        *,
        embedding_dim: int = 500,
        use_pretrained: bool = True,
        freeze_backbone: bool = True,
        unfreeze_mixed7: bool = True,
    ) -> None:
        super().__init__()

        weights = Inception_V3_Weights.DEFAULT if use_pretrained else None
        # torchvision expects aux_logits=True when pretrained Inception weights are requested.
        aux_logits = bool(use_pretrained)
        init_weights = False if use_pretrained else True
        inception = models.inception_v3(
            weights=weights,
            aux_logits=aux_logits,
            transform_input=False,
            # Torchvision expects init_weights=False when loading pretrained weights.
            init_weights=init_weights,
        )
        if use_pretrained:
            # Disable auxiliary classifier after loading weights to keep single-logit output.
            inception.AuxLogits = None
            inception.aux_logits = False
        inception.fc = nn.Identity()

        if freeze_backbone:
            for p in inception.parameters():
                p.requires_grad = False

        if unfreeze_mixed7:
            # Unfreeze top-level blocks (Mixed_7a, Mixed_7b, Mixed_7c).
            for name, p in inception.named_parameters():
                if "Mixed_7" in name:
                    p.requires_grad = True

        self.backbone = inception
        self.proj = nn.Linear(2048, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)  # (B, 2048)
        feats = self.proj(feats)  # (B, embedding_dim)
        return F.relu(feats)


class Channel2Branch(nn.Module):
    """CECED branch: VGG16 conv features -> global avg pool -> projection to 500-dim."""

    def __init__(
        self,
        *,
        embedding_dim: int = 500,
        use_pretrained: bool = True,
        freeze_backbone: bool = True,
        unfreeze_block5: bool = True,
    ) -> None:
        super().__init__()

        weights = VGG16_Weights.DEFAULT if use_pretrained else None
        vgg = models.vgg16(weights=weights)
        self.backbone = vgg.features

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if unfreeze_block5:
            # VGG16 features indices: 0-4 block1, 5-9 block2, 10-16 block3, 17-23 block4, 24-30 block5
            for i, layer in enumerate(self.backbone):
                if i >= 24:
                    for p in layer.parameters():
                        p.requires_grad = True

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(512, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        feats = self.avgpool(feats)
        feats = torch.flatten(feats, 1)
        feats = self.proj(feats)
        return F.relu(feats)


class WeightedFusionLayer(nn.Module):
    """Learned scalar fusion: F = α·F1 + (1-α)·F2, with α = sigmoid(w)."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.0))  # sigmoid -> 0.5 at init

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.w)
        return alpha * f1 + (1.0 - alpha) * f2


class DualChannelDiabeticRetinopathyModel(nn.Module):
    """Two-branch feature extractor with weighted fusion and classifier head."""

    def __init__(
        self,
        *,
        num_classes: int = 5,
        embedding_dim: int = 500,
        dropout_p: float = 0.3,
        use_pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.c1 = Channel1Branch(embedding_dim=embedding_dim, use_pretrained=use_pretrained)
        self.c2 = Channel2Branch(embedding_dim=embedding_dim, use_pretrained=use_pretrained)
        self.fusion = WeightedFusionLayer()
        self.post_fusion_dropout = nn.Dropout(p=float(dropout_p))
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x1, x2 = inputs
        f1 = self.c1(x1)
        f2 = self.c2(x2)
        fused = self.fusion(f1, f2)
        fused = self.post_fusion_dropout(fused)
        return self.classifier(fused)