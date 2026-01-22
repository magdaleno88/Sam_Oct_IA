"""Unit tests for BaseLightningModel."""

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from sam_ml.modeling.models.base import BaseLightningModel


class ConcreteTestModel(BaseLightningModel):
    """Concrete implementation of BaseLightningModel for testing."""

    def _create_model(self) -> None:
        """Create a simple test model architecture."""
        self.model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_classes),
        )


@pytest.fixture
def sample_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Create a sample batch for testing."""
    batch_size = 4
    input_size = 10
    num_classes = 5
    
    inputs = torch.randn(batch_size, input_size)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    return (inputs, targets)


@pytest.fixture
def model_default() -> ConcreteTestModel:
    """Create a model with default parameters."""
    return ConcreteTestModel()


@pytest.fixture
def model_custom() -> ConcreteTestModel:
    """Create a model with custom parameters."""
    return ConcreteTestModel(
        num_classes=3,
        learning_rate=1e-3,
        optimizer="sgd",
        weight_decay=0.01,
    )


class TestInitialization:
    """Tests for model initialization."""

    def test_init_default_parameters(self) -> None:
        """Test initialization with default parameters."""
        model = ConcreteTestModel()
        
        assert model.num_classes == 5
        assert model.learning_rate == 1e-4
        assert model.optimizer_name == "adam"
        assert model.weight_decay == 0.0
        assert model.model is not None
        assert isinstance(model.criterion, nn.CrossEntropyLoss)

    def test_init_custom_parameters(self) -> None:
        """Test initialization with custom parameters."""
        model = ConcreteTestModel(
            num_classes=3,
            learning_rate=1e-3,
            optimizer="SGD",  # Test case insensitivity
            weight_decay=0.01,
        )
        
        assert model.num_classes == 3
        assert model.learning_rate == 1e-3
        assert model.optimizer_name == "sgd"  # Should be lowercased
        assert model.weight_decay == 0.01
        assert model.model is not None

    def test_init_hyperparameters_saved(self) -> None:
        """Test that hyperparameters are saved."""
        model = ConcreteTestModel(num_classes=3, learning_rate=1e-3)
        
        # Check that hyperparameters are stored
        assert hasattr(model, "hparams")
        assert model.hparams["num_classes"] == 3
        assert model.hparams["learning_rate"] == 1e-3

    def test_init_model_created(self) -> None:
        """Test that model architecture is created during initialization."""
        model = ConcreteTestModel()
        
        assert model.model is not None
        assert isinstance(model.model, nn.Module)

    def test_init_abstract_class_raises_error(self) -> None:
        """Test that instantiating the abstract base class raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Subclasses must implement"):
            BaseLightningModel()


class TestForward:
    """Tests for forward pass."""

    def test_forward_basic(self, model_default: ConcreteTestModel, sample_batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Test basic forward pass."""
        inputs, _ = sample_batch
        
        output = model_default.forward(inputs)
        
        assert output.shape == (4, 5)  # (batch_size, num_classes)
        assert isinstance(output, torch.Tensor)

    def test_forward_different_batch_sizes(self, model_default: ConcreteTestModel) -> None:
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 4, 8, 16]:
            inputs = torch.randn(batch_size, 10)
            output = model_default.forward(inputs)
            
            assert output.shape == (batch_size, 5)

    def test_forward_without_model_raises_error(self) -> None:
        """Test that forward raises error if model is not initialized."""
        # Create a model but manually set model to None
        model = ConcreteTestModel()
        model.model = None
        
        inputs = torch.randn(4, 10)
        
        with pytest.raises(RuntimeError, match="Model not initialized"):
            model.forward(inputs)


class TestComputeLoss:
    """Tests for loss computation."""

    def test_compute_loss_basic(self, model_default: ConcreteTestModel) -> None:
        """Test basic loss computation."""
        batch_size = 4
        num_classes = 5
        
        # Create predictions with gradients by passing through model
        inputs = torch.randn(batch_size, 10, requires_grad=True)
        predictions = model_default.forward(inputs)
        targets = torch.randint(0, num_classes, (batch_size,))
        
        loss = model_default._compute_loss(predictions, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0  # Loss should be non-negative
        assert loss.requires_grad  # Loss should have gradients

    def test_compute_loss_perfect_predictions(self, model_default: ConcreteTestModel) -> None:
        """Test loss computation with perfect predictions."""
        batch_size = 4
        num_classes = 5
        
        # Create perfect predictions (one-hot encoded)
        predictions = torch.zeros(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        for i, target in enumerate(targets):
            predictions[i, target] = 10.0  # High confidence for correct class
        
        loss = model_default._compute_loss(predictions, targets)
        
        # Loss should be very small (but not exactly zero due to numerical precision)
        assert loss.item() < 0.1

    def test_compute_loss_wrong_predictions(self, model_default: ConcreteTestModel) -> None:
        """Test loss computation with wrong predictions."""
        batch_size = 4
        num_classes = 5
        
        # Create wrong predictions (all confidence on wrong class)
        predictions = torch.zeros(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        for i, target in enumerate(targets):
            wrong_class = (target + 1) % num_classes
            predictions[i, wrong_class] = 10.0  # High confidence for wrong class
        
        loss = model_default._compute_loss(predictions, targets)
        
        # Loss should be high
        assert loss.item() > 1.0


class TestComputeMetrics:
    """Tests for metrics computation."""

    def test_compute_metrics_basic(self, model_default: ConcreteTestModel) -> None:
        """Test basic metrics computation."""
        batch_size = 4
        num_classes = 5
        
        predictions = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        
        metrics = model_default._compute_metrics(predictions, targets)
        
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert isinstance(metrics["accuracy"], torch.Tensor)
        assert 0.0 <= metrics["accuracy"].item() <= 1.0

    def test_compute_metrics_perfect_accuracy(self, model_default: ConcreteTestModel) -> None:
        """Test metrics computation with perfect predictions."""
        batch_size = 4
        num_classes = 5
        
        # Create perfect predictions
        predictions = torch.zeros(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        for i, target in enumerate(targets):
            predictions[i, target] = 10.0
        
        metrics = model_default._compute_metrics(predictions, targets)
        
        assert metrics["accuracy"].item() == 1.0

    def test_compute_metrics_zero_accuracy(self, model_default: ConcreteTestModel) -> None:
        """Test metrics computation with all wrong predictions."""
        batch_size = 4
        num_classes = 5
        
        # Create all wrong predictions
        predictions = torch.zeros(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        for i, target in enumerate(targets):
            wrong_class = (target + 1) % num_classes
            predictions[i, wrong_class] = 10.0
        
        metrics = model_default._compute_metrics(predictions, targets)
        
        assert metrics["accuracy"].item() == 0.0

    def test_compute_metrics_partial_accuracy(self, model_default: ConcreteTestModel) -> None:
        """Test metrics computation with partial accuracy."""
        batch_size = 4
        num_classes = 5
        
        predictions = torch.zeros(batch_size, num_classes)
        targets = torch.tensor([0, 1, 2, 3])
        
        # Make 2 correct predictions
        predictions[0, 0] = 10.0  # Correct
        predictions[1, 1] = 10.0  # Correct
        predictions[2, 0] = 10.0  # Wrong (should be 2)
        predictions[3, 0] = 10.0  # Wrong (should be 3)
        
        metrics = model_default._compute_metrics(predictions, targets)
        
        assert metrics["accuracy"].item() == 0.5  # 2 out of 4 correct


class TestTrainingStep:
    """Tests for training step."""

    def test_training_step_basic(self, model_default: ConcreteTestModel, sample_batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Test basic training step."""
        batch = sample_batch
        
        # Mock the log method to avoid Lightning trainer dependency
        with patch.object(model_default, "log") as mock_log:
            loss = model_default.training_step(batch, batch_idx=0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        
        # Verify logging was called
        assert mock_log.called
        # Check that train_loss was logged
        log_calls = [call[0][0] for call in mock_log.call_args_list]
        assert "train_loss" in log_calls

    def test_training_step_logs_metrics(self, model_default: ConcreteTestModel, sample_batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Test that training step logs all metrics."""
        batch = sample_batch
        
        with patch.object(model_default, "log") as mock_log:
            model_default.training_step(batch, batch_idx=0)
        
        # Get all logged metric names
        logged_metrics = [call[0][0] for call in mock_log.call_args_list]
        
        assert "train_loss" in logged_metrics
        assert "train_accuracy" in logged_metrics

    def test_training_step_logs_with_correct_flags(self, model_default: ConcreteTestModel, sample_batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Test that training step logs with correct flags."""
        batch = sample_batch
        
        with patch.object(model_default, "log") as mock_log:
            model_default.training_step(batch, batch_idx=0)
        
        # Check that train_loss is logged with on_step=True, on_epoch=True, prog_bar=True
        for call in mock_log.call_args_list:
            metric_name = call[0][0]
            kwargs = call[1]
            
            if metric_name == "train_loss":
                assert kwargs.get("on_step") is True
                assert kwargs.get("on_epoch") is True
                assert kwargs.get("prog_bar") is True

    def test_training_step_returns_loss(self, model_default: ConcreteTestModel, sample_batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Test that training step returns loss tensor."""
        batch = sample_batch
        
        with patch.object(model_default, "log"):
            loss = model_default.training_step(batch, batch_idx=0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss


class TestValidationStep:
    """Tests for validation step."""

    def test_validation_step_basic(self, model_default: ConcreteTestModel, sample_batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Test basic validation step."""
        batch = sample_batch
        
        with patch.object(model_default, "log") as mock_log:
            model_default.validation_step(batch, batch_idx=0)
        
        # Verify logging was called
        assert mock_log.called
        log_calls = [call[0][0] for call in mock_log.call_args_list]
        assert "val_loss" in log_calls

    def test_validation_step_logs_metrics(self, model_default: ConcreteTestModel, sample_batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Test that validation step logs all metrics."""
        batch = sample_batch
        
        with patch.object(model_default, "log") as mock_log:
            model_default.validation_step(batch, batch_idx=0)
        
        logged_metrics = [call[0][0] for call in mock_log.call_args_list]
        
        assert "val_loss" in logged_metrics
        assert "val_accuracy" in logged_metrics

    def test_validation_step_logs_with_correct_flags(self, model_default: ConcreteTestModel, sample_batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Test that validation step logs with correct flags."""
        batch = sample_batch
        
        with patch.object(model_default, "log") as mock_log:
            model_default.validation_step(batch, batch_idx=0)
        
        # Check that val_loss is logged with on_step=False, on_epoch=True, prog_bar=True
        for call in mock_log.call_args_list:
            metric_name = call[0][0]
            kwargs = call[1]
            
            if metric_name == "val_loss":
                assert kwargs.get("on_step") is False
                assert kwargs.get("on_epoch") is True
                assert kwargs.get("prog_bar") is True

    def test_validation_step_no_return(self, model_default: ConcreteTestModel, sample_batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Test that validation step returns None."""
        batch = sample_batch
        
        with patch.object(model_default, "log"):
            result = model_default.validation_step(batch, batch_idx=0)
        
        assert result is None


class TestTestStep:
    """Tests for test step."""

    def test_test_step_basic(self, model_default: ConcreteTestModel, sample_batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Test basic test step."""
        batch = sample_batch
        
        with patch.object(model_default, "log") as mock_log:
            model_default.test_step(batch, batch_idx=0)
        
        # Verify logging was called
        assert mock_log.called
        log_calls = [call[0][0] for call in mock_log.call_args_list]
        assert "test_loss" in log_calls

    def test_test_step_logs_metrics(self, model_default: ConcreteTestModel, sample_batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Test that test step logs all metrics."""
        batch = sample_batch
        
        with patch.object(model_default, "log") as mock_log:
            model_default.test_step(batch, batch_idx=0)
        
        logged_metrics = [call[0][0] for call in mock_log.call_args_list]
        
        assert "test_loss" in logged_metrics
        assert "test_accuracy" in logged_metrics

    def test_test_step_logs_with_correct_flags(self, model_default: ConcreteTestModel, sample_batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Test that test step logs with correct flags."""
        batch = sample_batch
        
        with patch.object(model_default, "log") as mock_log:
            model_default.test_step(batch, batch_idx=0)
        
        # Check that test_loss is logged with on_step=False, on_epoch=True
        for call in mock_log.call_args_list:
            metric_name = call[0][0]
            kwargs = call[1]
            
            if metric_name == "test_loss":
                assert kwargs.get("on_step") is False
                assert kwargs.get("on_epoch") is True
                # Test step should not have prog_bar
                assert kwargs.get("prog_bar") is None or kwargs.get("prog_bar") is False

    def test_test_step_no_return(self, model_default: ConcreteTestModel, sample_batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Test that test step returns None."""
        batch = sample_batch
        
        with patch.object(model_default, "log"):
            result = model_default.test_step(batch, batch_idx=0)
        
        assert result is None


class TestConfigureOptimizers:
    """Tests for optimizer configuration."""

    def test_configure_optimizers_adam(self, model_default: ConcreteTestModel) -> None:
        """Test optimizer configuration with Adam."""
        config = model_default.configure_optimizers()
        
        assert isinstance(config, dict)
        assert "optimizer" in config
        assert "lr_scheduler" in config
        
        optimizer = config["optimizer"]
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]["lr"] == 1e-4
        assert optimizer.param_groups[0]["weight_decay"] == 0.0

    def test_configure_optimizers_sgd(self, model_custom: ConcreteTestModel) -> None:
        """Test optimizer configuration with SGD."""
        config = model_custom.configure_optimizers()
        
        assert isinstance(config, dict)
        optimizer = config["optimizer"]
        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]["lr"] == 1e-3
        assert optimizer.param_groups[0]["weight_decay"] == 0.01
        assert optimizer.param_groups[0]["momentum"] == 0.9

    def test_configure_optimizers_scheduler(self, model_default: ConcreteTestModel) -> None:
        """Test that learning rate scheduler is configured correctly."""
        config = model_default.configure_optimizers()
        
        assert "lr_scheduler" in config
        scheduler_config = config["lr_scheduler"]
        assert isinstance(scheduler_config, dict)
        assert "scheduler" in scheduler_config
        assert "monitor" in scheduler_config
        
        scheduler = scheduler_config["scheduler"]
        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert scheduler_config["monitor"] == "val_loss"

    def test_configure_optimizers_unsupported_raises_error(self) -> None:
        """Test that unsupported optimizer raises ValueError."""
        model = ConcreteTestModel(optimizer="unsupported")
        
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            model.configure_optimizers()

    def test_configure_optimizers_case_insensitive(self) -> None:
        """Test that optimizer name is case insensitive."""
        # Test with uppercase
        model1 = ConcreteTestModel(optimizer="ADAM")
        config1 = model1.configure_optimizers()
        
        # Test with lowercase
        model2 = ConcreteTestModel(optimizer="adam")
        config2 = model2.configure_optimizers()
        
        # Both should create Adam optimizer
        assert isinstance(config1["optimizer"], torch.optim.Adam)
        assert isinstance(config2["optimizer"], torch.optim.Adam)


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_full_training_workflow(self, model_default: ConcreteTestModel) -> None:
        """Test a complete training workflow with multiple batches."""
        batch_size = 4
        num_batches = 3
        
        batches = [
            (torch.randn(batch_size, 10), torch.randint(0, 5, (batch_size,)))
            for _ in range(num_batches)
        ]
        
        with patch.object(model_default, "log"):
            losses = []
            for batch_idx, batch in enumerate(batches):
                loss = model_default.training_step(batch, batch_idx)
                losses.append(loss)
        
        assert len(losses) == num_batches
        assert all(isinstance(loss, torch.Tensor) for loss in losses)

    def test_training_and_validation_workflow(self, model_default: ConcreteTestModel) -> None:
        """Test alternating training and validation steps."""
        batch = (torch.randn(4, 10), torch.randint(0, 5, (4,)))
        
        with patch.object(model_default, "log"):
            # Training step
            train_loss = model_default.training_step(batch, batch_idx=0)
            
            # Validation step
            model_default.validation_step(batch, batch_idx=0)
        
        assert isinstance(train_loss, torch.Tensor)

    def test_model_parameters_updated(self, model_default: ConcreteTestModel) -> None:
        """Test that model parameters can be updated (simulating training)."""
        # Get initial parameters
        initial_params = [p.clone() for p in model_default.parameters()]
        
        # Create a simple training step
        batch = (torch.randn(4, 10), torch.randint(0, 5, (4,)))
        
        with patch.object(model_default, "log"):
            loss = model_default.training_step(batch, batch_idx=0)
        
        # Manually update parameters (simulating optimizer step)
        loss.backward()
        optimizer = torch.optim.Adam(model_default.parameters(), lr=1e-4)
        optimizer.step()
        optimizer.zero_grad()
        
        # Check that parameters changed
        final_params = [p for p in model_default.parameters()]
        for initial, final in zip(initial_params, final_params):
            assert not torch.equal(initial, final), "Parameters should have changed after optimization"
