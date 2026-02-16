from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam_ml.modeling.models.base import BaseLightningModel
from sam_ml.modeling.models.registry import register_model


class _ConvBNReLUPool(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, padding: int, pool: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class PCNNFeatureExtractor(nn.Module):
    '''
    Lightweight Parallel CNN inspired by the paper:
    - 4 parallel conv blocks with 64 filters and kernels 9,7,5,3 (same padding)
    - concat -> conv(32,k=3,valid) -> pool -> conv(16,k=3,valid) -> pool -> dropout
    - flatten -> dense(250) -> dropout -> dense(120)  (features)

    Uses AdaptiveAvgPool2d to avoid dependence on input image size (your pipeline may be 512x512).
    '''

    def __init__(self, feature_dim: int = 120, dropout_p: float = 0.5) -> None:
        super().__init__()
        self.p9 = _ConvBNReLUPool(3, 64, k=9, padding=9 // 2, pool=True)
        self.p7 = _ConvBNReLUPool(3, 64, k=7, padding=7 // 2, pool=True)
        self.p5 = _ConvBNReLUPool(3, 64, k=5, padding=5 // 2, pool=True)
        self.p3 = _ConvBNReLUPool(3, 64, k=3, padding=3 // 2, pool=True)

        self.conv5 = nn.Conv2d(256, 32, kernel_size=3, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 16, kernel_size=3, padding=0, bias=False)
        self.bn6 = nn.BatchNorm2d(16)

        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dp_after_conv = nn.Dropout(p=dropout_p)

        # Stabilize FC input size (independent of input resolution)
        self.adapt = nn.AdaptiveAvgPool2d((29, 29))

        self.fc1 = nn.Linear(16 * 29 * 29, 250)
        self.bn_fc1 = nn.BatchNorm1d(250)
        self.dp_after_fc1 = nn.Dropout(p=dropout_p)
        self.fc_features = nn.Linear(250, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.p9(x)
        b = self.p7(x)
        c = self.p5(x)
        d = self.p3(x)

        x = torch.cat([a, b, c, d], dim=1)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.act(x)
        x = self.pool(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.act(x)
        x = self.pool(x)

        x = self.dp_after_conv(x)
        x = self.adapt(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.act(x)
        x = self.dp_after_fc1(x)

        feats = self.fc_features(x)
        return feats


@dataclass
class ELMState:
    w: torch.Tensor
    b: torch.Tensor
    beta: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor


def _fit_elm_closed_form(
    x: torch.Tensor,
    y: torch.Tensor,
    hidden_dim: int,
    num_classes: int,
    ridge_lambda: float,
    device: torch.device,
    seed: int = 42,
) -> ELMState:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    x = x.to(device)
    y = y.to(device)

    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-6)
    x_std = (x - mean) / std

    w = torch.randn(x_std.shape[1], hidden_dim, generator=g, device=device) * 0.1
    b = torch.randn(hidden_dim, generator=g, device=device) * 0.1

    h = F.relu(x_std @ w + b)
    t = F.one_hot(y, num_classes=num_classes).float()

    ht_h = h.T @ h
    ht_t = h.T @ t
    eye = torch.eye(hidden_dim, device=device, dtype=ht_h.dtype)
    beta = torch.linalg.solve(ht_h + ridge_lambda * eye, ht_t)

    return ELMState(w=w, b=b, beta=beta, mean=mean, std=std)


def _elm_predict_logits(state: ELMState, x: torch.Tensor) -> torch.Tensor:
    x_std = (x - state.mean) / state.std
    h = F.relu(x_std @ state.w + state.b)
    return h @ state.beta


class PCNNELMLightning(BaseLightningModel):
    '''
    Train CNN features with a linear head (CrossEntropy).
    Fit ELM at the end of each epoch using detached train features.
    Validation/Test uses ELM logits if available, otherwise uses linear head.
    '''

    def __init__(
        self,
        num_classes: int = 5,
        learning_rate: float = 1e-4,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        elm_hidden_dim: int = 1000,
        elm_ridge_lambda: float = 1e-3,
        elm_seed: int = 42,
        **kwargs: Any,
    ) -> None:
        self.elm_hidden_dim = elm_hidden_dim
        self.elm_ridge_lambda = elm_ridge_lambda
        self.elm_seed = elm_seed

        self._elm_state: Optional[ELMState] = None
        self._train_feats: List[torch.Tensor] = []
        self._train_labels: List[torch.Tensor] = []

        super().__init__(
            num_classes=num_classes,
            learning_rate=learning_rate,
            optimizer=optimizer,
            weight_decay=weight_decay,
            **kwargs,
        )

    def _create_model(self) -> None:
        self.backbone = PCNNFeatureExtractor(feature_dim=120, dropout_p=0.5)
        self.linear_head = nn.Linear(120, self.num_classes)
        self.model = nn.Sequential(self.backbone, self.linear_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if self._elm_state is not None:
            return _elm_predict_logits(self._elm_state, feats)
        return self.linear_head(feats)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        feats = self.backbone(x)
        logits = self.linear_head(feats)
        loss = self.criterion(logits, y)

        metrics = self._compute_metrics(logits.detach(), y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", metrics["accuracy"], on_step=True, on_epoch=True, prog_bar=True)

        self._train_feats.append(feats.detach().cpu())
        self._train_labels.append(y.detach().cpu())
        return loss

    def on_train_epoch_end(self) -> None:
        if not self._train_feats:
            return

        x = torch.cat(self._train_feats, dim=0)
        y = torch.cat(self._train_labels, dim=0)

        self._elm_state = _fit_elm_closed_form(
            x=x,
            y=y,
            hidden_dim=self.elm_hidden_dim,
            num_classes=self.num_classes,
            ridge_lambda=self.elm_ridge_lambda,
            device=self.device,
            seed=self.elm_seed,
        )

        self._train_feats.clear()
        self._train_labels.clear()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)

        metrics = self._compute_metrics(logits.detach(), y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", metrics["accuracy"], on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)

        metrics = self._compute_metrics(logits.detach(), y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_accuracy", metrics["accuracy"], on_step=False, on_epoch=True)


@register_model("pcnn_elm")
def create_pcnn_elm_model(**kwargs: Any) -> BaseLightningModel:
    return PCNNELMLightning(**kwargs)
