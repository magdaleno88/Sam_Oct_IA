"""Reproducible PyTorch Lightning training for OCT experiments."""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, WeightedRandomSampler

from sam_ml.oct.config import OCTConfig
from sam_ml.oct.data import load_dataset_splits, manifest_sha256
from sam_ml.oct.dataset import OCTManifestDataset, build_oct_transform
from sam_ml.oct.models import create_oct_model


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma, self.weight = gamma, weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()


class OCTLightningModule(pl.LightningModule):
    def __init__(self, config: OCTConfig, class_weights: list[float] | None = None) -> None:
        super().__init__()
        self.config_data = config.model_dump(mode="json")
        self.save_hyperparameters(self.config_data)
        self.model = create_oct_model(
            config.model.name, num_classes=config.model.num_classes,
            pretrained=config.model.pretrained, dropout=config.model.dropout,
            freeze_backbone=config.model.freeze_backbone,
            **({"replace_stride_with_dilation": config.model.replace_stride_with_dilation}
               if config.model.name == "improved_resnet50" else {}),
        )
        weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.register_buffer("class_weights", weights)
        self.criterion = FocalLoss(config.training.focal_gamma, weights) if config.training.balance_mode == "focal" else torch.nn.CrossEntropyLoss(weight=weights)
        self.learning_rate = config.training.learning_rate
        self.weight_decay = config.training.weight_decay

    def forward(self, inputs):
        return self.model(inputs)

    def _step(self, batch, stage):
        inputs, labels = batch
        logits = self(inputs)
        loss = self.criterion(logits, labels)
        accuracy = (logits.argmax(1) == labels).float().mean()
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_accuracy", accuracy, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}


def _class_weights(frame: pd.DataFrame) -> list[float]:
    counts = frame["class_index"].value_counts().reindex(range(4), fill_value=0).to_numpy()
    if (counts == 0).any():
        raise ValueError("Every OCT class must occur in the training manifest")
    return (len(frame) / (4 * counts)).tolist()


def _environment(
    config: OCTConfig,
    manifest_dir: Path,
    split_source: str,
) -> dict[str, object]:
    packages = {}
    for package in ("torch", "torchvision", "pytorch-lightning", "pandas", "scikit-learn"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = None
    manifest_hashes = {
        name: manifest_sha256(manifest_dir / f"{name}.csv")
        for name in ("train", "val", "test")
        if (manifest_dir / f"{name}.csv").exists()
    }
    return {
        "created_at": datetime.now(timezone.utc).isoformat(), "seed": config.training.seed,
        "python": sys.version, "platform": platform.platform(), "packages": packages,
        "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": commit,
        "split_source": split_source,
        "manifest_hashes": manifest_hashes,
    }


def train_experiment(config: OCTConfig, experiment: str, resume: str | None = None, seed: int | None = None):
    if seed is not None:
        config = config.model_copy(deep=True)
        config.training.seed = seed
    pl.seed_everything(config.training.seed, workers=True)
    run_dir = Path("runs") / experiment
    for folder in ("checkpoints", "predictions", "figures", "logs"):
        (run_dir / folder).mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
    splits, split_source = load_dataset_splits(config)
    train_frame = splits["train"]
    val_frame = splits["val"]
    (run_dir / "environment.json").write_text(
        json.dumps(_environment(config, config.data.manifest_dir, split_source), indent=2),
        encoding="utf-8",
    )
    weights = _class_weights(train_frame) if config.training.balance_mode in {"class_weights", "focal"} else None
    train_ds = OCTManifestDataset(
        train_frame, build_oct_transform(True, config.data.image_size),
        preprocessing=config.preprocessing,
    )
    val_ds = OCTManifestDataset(
        val_frame, build_oct_transform(False, config.data.image_size),
        preprocessing=config.preprocessing,
    )
    sampler = None
    shuffle = True
    if config.training.balance_mode == "weighted_sampler":
        class_weights = _class_weights(train_frame)
        sampler = WeightedRandomSampler([class_weights[i] for i in train_frame["class_index"]], len(train_frame), replacement=True)
        shuffle = False
    loader_args = {"batch_size": config.training.batch_size, "num_workers": config.training.num_workers,
                   "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(train_ds, shuffle=shuffle, sampler=sampler, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    accumulation = max(1, math.ceil(config.training.effective_batch_size / config.training.batch_size))
    checkpoint = ModelCheckpoint(dirpath=run_dir / "checkpoints", filename="best-{epoch:02d}-{val_loss:.4f}", monitor="val_loss", mode="min", save_last=True, save_top_k=1)
    trainer = pl.Trainer(
        max_steps=config.training.max_steps, max_epochs=config.training.max_epochs,
        accelerator="auto", devices=1, precision="16-mixed" if config.training.mixed_precision and torch.cuda.is_available() else "32-true",
        accumulate_grad_batches=accumulation, deterministic=True,
        callbacks=[checkpoint, EarlyStopping("val_loss", patience=config.training.early_stopping_patience, mode="min")],
        logger=CSVLogger(run_dir / "logs", name="training"),
    )
    model = OCTLightningModule(config, weights)
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume)
    metrics_file = Path(trainer.logger.log_dir) / "metrics.csv"
    if metrics_file.exists():
        (run_dir / "training_history.csv").write_bytes(metrics_file.read_bytes())
    summary = {
        "split_source": split_source,
        **{name: len(frame) for name, frame in splits.items()},
    }
    (run_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run_dir, checkpoint.best_model_path
