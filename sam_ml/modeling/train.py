"""Unified training script for all models using the model registry."""

import argparse
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from sam_ml.config import get_model_config, get_training_config
from sam_ml.modeling.models import get_model, list_models


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train a model using the model registry",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=f"Model to train. Available: {', '.join(list_models())}",
    )
    
    # Get defaults from config
    model_config = get_model_config()
    training_config = get_training_config()
    
    # Model hyperparameters
    parser.add_argument(
        "--num-classes",
        type=int,
        default=model_config.num_classes,
        help=f"Number of output classes (default: {model_config.num_classes})",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=model_config.learning_rate,
        help=f"Learning rate for optimizer (default: {model_config.learning_rate})",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=model_config.optimizer,
        choices=["adam", "sgd"],
        help=f"Optimizer to use (default: {model_config.optimizer})",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=model_config.weight_decay,
        help=f"Weight decay (L2 regularization) coefficient (default: {model_config.weight_decay})",
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=training_config.batch_size,
        help=f"Batch size for training (default: {training_config.batch_size})",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=training_config.num_epochs,
        help=f"Number of training epochs (default: {training_config.num_epochs})",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(training_config.data_dir),
        help=f"Directory containing processed dataset (default: {training_config.data_dir})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(training_config.output_dir),
        help=f"Directory to save model checkpoints and logs (default: {training_config.output_dir})",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=training_config.gpus,
        help=f"Number of GPUs to use (None for CPU) (default: {training_config.gpus})",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=training_config.patience,
        help=f"Early stopping patience (default: {training_config.patience})",
    )
    
    args = parser.parse_args()
    
    # Validate model key
    available_models = list_models()
    if args.model not in available_models:
        parser.error(
            f"Model '{args.model}' not found. Available models: {', '.join(available_models)}"
        )
    
    # Create output directory
    output_path = Path(args.output_dir) / args.model
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Training model: {args.model}")
    print(f"Output directory: {output_path}")
    print(f"Hyperparameters:")
    print(f"  - num_classes: {args.num_classes}")
    print(f"  - learning_rate: {args.learning_rate}")
    print(f"  - optimizer: {args.optimizer}")
    print(f"  - weight_decay: {args.weight_decay}")
    print(f"  - batch_size: {args.batch_size}")
    print(f"  - num_epochs: {args.num_epochs}")
    
    # Get model from registry
    try:
        model = get_model(
            args.model,
            num_classes=args.num_classes,
            learning_rate=args.learning_rate,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
        )
        print(f"✓ Model '{args.model}' loaded successfully")
    except KeyError as e:
        print(f"✗ Error loading model: {e}")
        return
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        return
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path / "checkpoints",
        filename=f"{args.model}-{{epoch:02d}}-{{val_loss:.2f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=args.patience,
        verbose=True,
    )
    
    # Setup logger
    logger = CSVLogger(
        save_dir=output_path,
        name="logs",
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        accelerator="gpu" if args.gpus else "cpu",
        devices=args.gpus if args.gpus else 1,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    
    # TODO: Load datasets when dataset classes are available
    # For now, print a message
    print("\n" + "=" * 60)
    print("NOTE: Dataset loading is not yet implemented.")
    print("When dataset classes are available, uncomment the following:")
    print("  - Create train_dataset and val_dataset")
    print("  - Create train_loader and val_loader")
    print("  - Call trainer.fit(model, train_loader, val_loader)")
    print("=" * 60)
    
    # Uncomment when datasets are ready:
    # from sam_ml.datasets import DiabeticRetinopathyDataset
    # from torch.utils.data import DataLoader
    # 
    # train_dataset = DiabeticRetinopathyDataset(
    #     data_dir=args.data_dir,
    #     split="train",
    # )
    # val_dataset = DiabeticRetinopathyDataset(
    #     data_dir=args.data_dir,
    #     split="val",
    # )
    # 
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=4,
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=4,
    # )
    # 
    # trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
