"""Unit tests for DualChannelDiabeticRetinopathyModel."""

from typing import List

import pytest
import tensorflow as tf
import numpy as np

from mlops_project.modeling.models.dual_channel_model import DualChannelDiabeticRetinopathyModel


class TestDualChannelDiabeticRetinopathyModel:
    """Test cases for DualChannelDiabeticRetinopathyModel."""
    
    @pytest.fixture
    def model(self) -> DualChannelDiabeticRetinopathyModel:
        """Create a model instance with default parameters."""
        return DualChannelDiabeticRetinopathyModel(num_classes=5)
    
    @pytest.fixture
    def dummy_inputs(self) -> List[tf.Tensor]:
        """Create dummy preprocessed images for testing."""
        batch_size = 2
        height, width, channels = 224, 224, 3
        
        clahe_images = tf.random.uniform(
            (batch_size, height, width, channels),
            minval=0.0,
            maxval=1.0
        )
        ceced_images = tf.random.uniform(
            (batch_size, height, width, channels),
            minval=0.0,
            maxval=1.0
        )
        
        return [clahe_images, ceced_images]
    
    @pytest.fixture
    def dummy_labels(self) -> tf.Tensor:
        """Create dummy one-hot encoded labels."""
        labels = tf.one_hot([0, 2], depth=5)
        return labels
    
    def test_initialization(self, model: DualChannelDiabeticRetinopathyModel) -> None:
        """Test that model initializes correctly."""
        assert model.num_classes == 5
        assert model.input_shape == (224, 224, 3)
    
    def test_call_output_shape(
        self, 
        model: DualChannelDiabeticRetinopathyModel, 
        dummy_inputs: List[tf.Tensor]
    ) -> None:
        """Test that model call returns correct output shape."""
        predictions = model(dummy_inputs, training=False)
        
        batch_size = dummy_inputs[0].shape[0]
        expected_shape = (batch_size, model.num_classes)
        
        assert predictions.shape == expected_shape
    
    def test_call_output_is_probabilities(
        self, 
        model: DualChannelDiabeticRetinopathyModel, 
        dummy_inputs: List[tf.Tensor]
    ) -> None:
        """Test that model output is a probability distribution."""
        predictions = model(dummy_inputs, training=False)
        
        row_sums = tf.reduce_sum(predictions, axis=1)
        np.testing.assert_array_almost_equal(
            row_sums.numpy(),
            np.ones(row_sums.shape),
            decimal=5
        )
        
        assert tf.reduce_all(predictions >= 0)
        assert tf.reduce_all(predictions <= 1)
    
    def test_model_compilation(
        self, 
        model: DualChannelDiabeticRetinopathyModel
    ) -> None:
        """Test that model can be compiled."""
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        assert model.optimizer is not None
        assert model.loss is not None
    
    def test_model_training_step(
        self, 
        model: DualChannelDiabeticRetinopathyModel, 
        dummy_inputs: List[tf.Tensor],
        dummy_labels: tf.Tensor
    ) -> None:
        """Test that model can perform a training step."""
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        from tensorflow.keras.losses import get as get_loss
        loss_fn = get_loss("categorical_crossentropy")
        
        with tf.GradientTape() as tape:
            predictions = model(dummy_inputs, training=True)
            loss = loss_fn(dummy_labels, predictions)
        
        trainable_vars = model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        assert len(gradients) == len(trainable_vars)
        assert all(g is not None for g in gradients)
