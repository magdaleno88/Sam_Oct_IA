"""Training script for diabetic retinopathy detection models."""

import argparse
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from sam_ml.datasets.eyepacs import load_eyepacs_dual_channel
from sam_ml.modeling.models.dual_channel_model import DualChannelDiabeticRetinopathyModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_cuda_availability() -> bool:
    """
    Check if CUDA is available and properly configured.
    
    Returns:
        True if CUDA is available and can be used, False otherwise
        
    Example:
        >>> if check_cuda_availability():
        ...     print("CUDA is available!")
        ... else:
        ...     print("Using CPU")
    """
    try:
        # Check if TensorFlow can see any GPUs
        gpus = tf.config.list_physical_devices("GPU")
        
        if len(gpus) == 0:
            logger.info("No GPU devices found. Using CPU.")
            return False
        
        # Check if CUDA is built and available
        if not tf.test.is_built_with_cuda():
            logger.warning(
                "GPU devices found but TensorFlow was not built with CUDA support. "
                "Using CPU."
            )
            return False
        
        # Configure GPU memory growth to avoid allocating all memory at once
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Configured GPU: {gpu.name}")
            except RuntimeError as e:
                logger.warning(f"Could not configure GPU {gpu.name}: {e}")
        
        logger.info(f"CUDA is available. Found {len(gpus)} GPU(s).")
        return True
        
    except Exception as e:
        logger.warning(f"Error checking CUDA availability: {e}. Using CPU.")
        return False


def create_model(
    num_classes: int = 5,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
) -> DualChannelDiabeticRetinopathyModel:
    """
    Create a dual-channel diabetic retinopathy detection model.
    
    Args:
        num_classes: Number of output classes (default: 5)
        input_shape: Shape of input images (height, width, channels) (default: (224, 224, 3))
        
    Returns:
        Model instance (not compiled)
        
    Example:
        >>> model = create_model(num_classes=5)
        >>> model.summary()
    """
    model = DualChannelDiabeticRetinopathyModel(
        num_classes=num_classes,
        input_shape=input_shape,
    )
    
    return model


def load_dataset(
    base_path: Optional[Path] = None,
    dataset_name: str = "eyepacs_dataset",
    batch_size: int = 32,
    image_size_clahe: Tuple[int, int] = (299, 299),
    image_size_ceced: Tuple[int, int] = (224, 224),
    label_mode: str = "categorical",
    shuffle: bool = True,
    seed: Optional[int] = 42,
    cache: bool = True,
    prefetch: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Load EyePACS dataset for dual-channel model training.
    
    Args:
        base_path: Base path to processed datasets. If None, uses default:
                   data/processed/{dataset_name}
        dataset_name: Name of the dataset directory (default: "eyepacs_dataset")
        batch_size: Batch size for datasets (default: 32)
        image_size_clahe: Image size for CLAHE channel (default: (299, 299))
        image_size_ceced: Image size for CECED channel (default: (224, 224))
        label_mode: Label format - "categorical", "int", or "binary" (default: "categorical")
        shuffle: Whether to shuffle the training data (default: True)
        seed: Random seed for shuffling (default: 42)
        cache: Whether to cache datasets in memory (default: True)
        prefetch: Whether to prefetch batches (default: True)
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
        
    Example:
        >>> train, val, test = load_dataset(batch_size=64)
    """
    train, val, test = load_eyepacs_dual_channel(
        base_path=base_path,
        dataset_name=dataset_name,
        batch_size=batch_size,
        image_size_clahe=image_size_clahe,
        image_size_ceced=image_size_ceced,
        label_mode=label_mode,
        shuffle=shuffle,
        seed=seed,
        cache=cache,
        prefetch=prefetch,
    )
    
    return train, val, test


def create_adam_optimizer(
    learning_rate: float = 0.001,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1e-7,
    weight_decay: float = 0.0,
) -> keras.optimizers.Adam:
    """
    Create an Adam optimizer with specified hyperparameters.
    
    Args:
        learning_rate: Learning rate (default: 0.001)
        beta_1: Exponential decay rate for the first moment estimates (default: 0.9)
        beta_2: Exponential decay rate for the second moment estimates (default: 0.999)
        epsilon: Small constant for numerical stability (default: 1e-7)
        weight_decay: Weight decay coefficient (default: 0.0)
                     Note: Weight decay in TensorFlow/Keras is typically handled via
                     kernel_regularizer in layers. This parameter is included for
                     future compatibility or custom optimizer wrappers.
        
    Returns:
        Configured Adam optimizer
        
    Example:
        >>> optimizer = create_adam_optimizer(learning_rate=0.0001, weight_decay=1e-4)
    """
    # Note: TensorFlow/Keras Adam optimizer doesn't directly support weight_decay
    # Weight decay is typically implemented via kernel_regularizer in layers
    # or using optimizer wrappers. For now, we'll create the optimizer with
    # standard parameters and document weight_decay for future use.
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
    )
    
    if weight_decay > 0.0:
        logger.warning(
            f"Weight decay ({weight_decay}) is specified but not directly supported "
            "by Adam optimizer. Consider using kernel_regularizer in model layers "
            "or an optimizer wrapper for weight decay."
        )
    
    return optimizer


def create_callbacks(
    checkpoint_dir: Optional[Path] = None,
    checkpoint_filename: str = "best_model.keras",
    monitor: str = "val_loss",
    mode: str = "min",
    patience: int = 10,
    restore_best_weights: bool = True,
    reduce_lr_patience: int = 5,
    reduce_lr_factor: float = 0.5,
    reduce_lr_min_lr: float = 1e-7,
    verbose: int = 1,
) -> List[keras.callbacks.Callback]:
    """
    Create training callbacks for model training.
    
    Args:
        checkpoint_dir: Directory to save model checkpoints. If None, uses "models/checkpoints"
        checkpoint_filename: Filename for model checkpoint (default: "best_model.keras")
        monitor: Metric to monitor for early stopping and checkpointing (default: "val_loss")
        mode: One of "min" or "max" (default: "min")
        patience: Number of epochs to wait before early stopping (default: 10)
        restore_best_weights: Whether to restore best weights after training (default: True)
        reduce_lr_patience: Patience for learning rate reduction (default: 5)
        reduce_lr_factor: Factor by which learning rate is reduced (default: 0.5)
        reduce_lr_min_lr: Minimum learning rate (default: 1e-7)
        verbose: Verbosity level (default: 1)
        
    Returns:
        List of Keras callbacks
        
    Example:
        >>> callbacks = create_callbacks(checkpoint_dir=Path("models"))
    """
    callbacks: List[keras.callbacks.Callback] = []
    
    # Model checkpoint callback
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / checkpoint_filename
        
        checkpoint_callback = ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            save_weights_only=False,
            verbose=verbose,
        )
        callbacks.append(checkpoint_callback)
        logger.info(f"Model checkpoint will be saved to: {checkpoint_path}")
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=patience,
        restore_best_weights=restore_best_weights,
        verbose=verbose,
    )
    callbacks.append(early_stopping)
    
    # Learning rate reduction callback
    reduce_lr = ReduceLROnPlateau(
        monitor=monitor,
        mode=mode,
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        min_lr=reduce_lr_min_lr,
        verbose=verbose,
    )
    callbacks.append(reduce_lr)
    
    return callbacks


def train(
    model: Optional[keras.Model] = None,
    train_dataset: Optional[tf.data.Dataset] = None,
    val_dataset: Optional[tf.data.Dataset] = None,
    test_dataset: Optional[tf.data.Dataset] = None,
    epochs: int = 50,
    optimizer: Optional[Union[str, keras.optimizers.Optimizer]] = None,
    loss: Union[str, Callable] = "categorical_crossentropy",
    metrics: Optional[List[Union[str, Callable]]] = None,
    callbacks: Optional[List[keras.callbacks.Callback]] = None,
    validation_freq: int = 1,
    verbose: int = 1,
    use_cuda: Optional[bool] = None,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_filename: str = "best_model.keras",
    learning_rate: float = 0.001,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1e-7,
    weight_decay: float = 0.0,
    num_classes: int = 5,
    **kwargs,
) -> keras.callbacks.History:
    """
    Train a diabetic retinopathy detection model.
    
    This function handles model creation, dataset loading, CUDA validation,
    and training with proper Keras callbacks.
    
    Args:
        model: Keras model to train. If None, creates default DualChannelDiabeticRetinopathyModel
        train_dataset: Training dataset. If None, loads default EyePACS dataset
        val_dataset: Validation dataset. If None, loads default EyePACS dataset
        test_dataset: Test dataset (optional, for final evaluation)
        epochs: Number of training epochs (default: 50)
        optimizer: Optimizer to use. If None, creates Adam optimizer with hyperparameters (default: None)
        loss: Loss function to use (default: "categorical_crossentropy")
        metrics: List of metrics to track (default: ["accuracy"])
        callbacks: List of Keras callbacks. If None, creates default callbacks
        validation_freq: Frequency of validation (default: 1)
        verbose: Verbosity level (default: 1)
        use_cuda: Whether to use CUDA. If None, auto-detects CUDA availability
        checkpoint_dir: Directory to save model checkpoints (default: "models/checkpoints")
        checkpoint_filename: Filename for model checkpoint (default: "best_model.keras")
        learning_rate: Learning rate for Adam optimizer (default: 0.001)
        beta_1: Exponential decay rate for first moment estimates (default: 0.9)
        beta_2: Exponential decay rate for second moment estimates (default: 0.999)
        epsilon: Small constant for numerical stability (default: 1e-7)
        weight_decay: Weight decay coefficient (default: 0.0)
        **kwargs: Additional arguments passed to dataset loading function
        
    Returns:
        Training history object
        
    Example:
        >>> history = train(epochs=100, batch_size=64)
        >>> # Or with custom model and dataset:
        >>> model = create_model()
        >>> train, val, test = load_dataset()
        >>> history = train(model=model, train_dataset=train, val_dataset=val)
    """
    # Check CUDA availability
    if use_cuda is None:
        use_cuda = check_cuda_availability()
    elif use_cuda:
        if not check_cuda_availability():
            logger.warning(
                "CUDA was requested but not available. Falling back to CPU."
            )
            use_cuda = False
    
    # Create or use provided model
    if model is None:
        logger.info("Creating default dual-channel model...")
        model = create_model(num_classes=num_classes)
    else:
        logger.info(f"Using provided model: {model.name}")
    
    # Load datasets if not provided
    if train_dataset is None or val_dataset is None:
        logger.info("Loading default EyePACS dataset...")
        train_ds, val_ds, test_ds = load_dataset(**kwargs)
        
        if train_dataset is None:
            train_dataset = train_ds
        if val_dataset is None:
            val_dataset = val_ds
        if test_dataset is None:
            test_dataset = test_ds
    
    # Set default metrics
    if metrics is None:
        metrics = ["accuracy"]
    
    # Create optimizer if not provided
    if optimizer is None:
        logger.info("Creating Adam optimizer with specified hyperparameters...")
        optimizer = create_adam_optimizer(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            weight_decay=weight_decay,
        )
    elif isinstance(optimizer, str) and optimizer.lower() == "adam":
        # If optimizer is string "adam", create Adam optimizer with hyperparameters
        logger.info("Creating Adam optimizer with specified hyperparameters...")
        optimizer = create_adam_optimizer(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            weight_decay=weight_decay,
        )
    
    # Compile model
    logger.info("Compiling model...")
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )
    
    # Create callbacks if not provided
    if callbacks is None:
        if checkpoint_dir is None:
            checkpoint_dir = Path("models/checkpoints")
        callbacks = create_callbacks(
            checkpoint_dir=checkpoint_dir,
            checkpoint_filename=checkpoint_filename,
        )
    
    # Log training configuration
    logger.info("=" * 60)
    logger.info("Training Configuration:")
    logger.info(f"  Model: {model.name}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Optimizer: {optimizer}")
    if isinstance(optimizer, keras.optimizers.Adam):
        logger.info(f"    Learning Rate: {optimizer.learning_rate.numpy() if hasattr(optimizer.learning_rate, 'numpy') else optimizer.learning_rate}")
        logger.info(f"    Beta 1: {optimizer.beta_1.numpy() if hasattr(optimizer.beta_1, 'numpy') else optimizer.beta_1}")
        logger.info(f"    Beta 2: {optimizer.beta_2.numpy() if hasattr(optimizer.beta_2, 'numpy') else optimizer.beta_2}")
        logger.info(f"    Epsilon: {optimizer.epsilon}")
        if weight_decay > 0.0:
            logger.info(f"    Weight Decay: {weight_decay}")
    logger.info(f"  Loss: {loss}")
    logger.info(f"  Metrics: {metrics}")
    logger.info(f"  CUDA: {use_cuda}")
    logger.info(f"  Callbacks: {len(callbacks)}")
    logger.info("=" * 60)
    
    # Train model
    logger.info("Starting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        validation_freq=validation_freq,
        verbose=verbose,
    )
    
    logger.info("Training completed!")
    
    # Evaluate on test set if provided
    if test_dataset is not None:
        logger.info("Evaluating on test set...")
        test_results = model.evaluate(test_dataset, verbose=verbose)
        logger.info(f"Test results: {dict(zip(model.metrics_names, test_results))}")
    
    return history


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for training script.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Train diabetic retinopathy detection model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Path to processed dataset directory. If not provided, uses default: data/processed/eyepacs_dataset",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="eyepacs_dataset",
        help="Name of the dataset directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--validation-freq",
        type=int,
        default=1,
        help="Frequency of validation (validate every N epochs)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)",
    )
    
    # Optimizer hyperparameters
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for Adam optimizer",
    )
    parser.add_argument(
        "--beta-1",
        type=float,
        default=0.9,
        help="Exponential decay rate for first moment estimates",
    )
    parser.add_argument(
        "--beta-2",
        type=float,
        default=0.999,
        help="Exponential decay rate for second moment estimates",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-7,
        help="Small constant for numerical stability",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay coefficient (note: not directly supported by Adam, use kernel_regularizer for weight decay)",
    )
    
    # Model arguments
    parser.add_argument(
        "--num-classes",
        type=int,
        default=5,
        help="Number of output classes",
    )
    
    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("models/checkpoints"),
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--checkpoint-filename",
        type=str,
        default="best_model.keras",
        help="Filename for model checkpoint",
    )
    
    # CUDA arguments
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        help="Force use of CUDA (if available). If not specified, auto-detects CUDA availability",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Force use of CPU (disable CUDA)",
    )
    
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for training script.
    
    Parses command-line arguments and starts training.
    """
    args = parse_args()
    
    # Determine CUDA usage
    use_cuda: Optional[bool] = None
    if args.use_cuda:
        use_cuda = True
    elif args.no_cuda:
        use_cuda = False
    
    # Prepare dataset loading kwargs
    dataset_kwargs: Dict[str, Any] = {
        "batch_size": args.batch_size,
    }
    if args.dataset_path is not None:
        dataset_kwargs["base_path"] = args.dataset_path
        dataset_kwargs["dataset_name"] = args.dataset_name
    
    # Start training
    logger.info("Starting training with command-line arguments...")
    history = train(
        epochs=args.epochs,
        validation_freq=args.validation_freq,
        verbose=args.verbose,
        use_cuda=use_cuda,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_filename=args.checkpoint_filename,
        learning_rate=args.learning_rate,
        beta_1=args.beta_1,
        beta_2=args.beta_2,
        epsilon=args.epsilon,
        weight_decay=args.weight_decay,
        num_classes=args.num_classes,
        **dataset_kwargs,
    )
    
    logger.info("Training script completed successfully.")


if __name__ == "__main__":
    main()
