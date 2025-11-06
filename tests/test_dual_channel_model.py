"""Unit tests for DualChannelDiabeticRetinopathyModel."""

from typing import List

import pytest
import tensorflow as tf
import numpy as np

from sam_ml.modeling.models.dual_channel_model import (
    DualChannelDiabeticRetinopathyModel,
    Channel1Branch,
    Channel2Branch,
    WeightedFusionLayer,
)


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
        assert model.name == "dual_channel_dr_model"
        assert model.channel1_branch is not None
        assert model.channel2_branch is not None
        assert model.fusion_layer is not None
        assert model.classifier is not None
    
    def test_call_output_shape(
        self, 
        model: DualChannelDiabeticRetinopathyModel, 
        dummy_inputs: List[tf.Tensor]
    ) -> None:
        """Test that model call returns correct output shape."""
        predictions = model(dummy_inputs, training=False)
        
        batch_size = dummy_inputs[0].shape[0]
        expected_shape = (batch_size, 5)  # num_classes = 5
        
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

    def test_channel1_branch_output_shape(
        self,
        dummy_inputs: List[tf.Tensor]
    ) -> None:
        """Test that Channel1Branch outputs shape (batch, 500)."""
        branch = Channel1Branch(input_shape=(224, 224, 3))
        output = branch(dummy_inputs[0], training=False)
        
        batch_size = dummy_inputs[0].shape[0]
        expected_shape = (batch_size, 500)
        
        assert output.shape == expected_shape

    def test_channel2_branch_output_shape(
        self,
        dummy_inputs: List[tf.Tensor]
    ) -> None:
        """Test that Channel2Branch outputs shape (batch, 500)."""
        branch = Channel2Branch(input_shape=(224, 224, 3))
        output = branch(dummy_inputs[1], training=False)
        
        batch_size = dummy_inputs[1].shape[0]
        expected_shape = (batch_size, 500)
        
        assert output.shape == expected_shape

    def test_weighted_fusion_layer_output_shape(
        self,
        dummy_inputs: List[tf.Tensor]
    ) -> None:
        """Test that WeightedFusionLayer outputs shape (batch, 500) when given two tensors of shape (batch, 500)."""
        fusion_layer = WeightedFusionLayer()
        
        # Create two 500-dim feature vectors
        batch_size = dummy_inputs[0].shape[0]
        fc1_2 = tf.random.uniform((batch_size, 500))
        fc2_2 = tf.random.uniform((batch_size, 500))
        
        fused = fusion_layer([fc1_2, fc2_2])
        
        expected_shape = (batch_size, 500)
        assert fused.shape == expected_shape

    def test_weighted_fusion_layer_shape_mismatch_raises_error(self) -> None:
        """Test that WeightedFusionLayer raises error for mismatched shapes."""
        fusion_layer = WeightedFusionLayer()
        
        fc1_2 = tf.random.uniform((2, 500))
        fc2_2 = tf.random.uniform((2, 300))  # Different shape
        
        with pytest.raises(ValueError, match="Channel shapes must match"):
            _ = fusion_layer([fc1_2, fc2_2])

    def test_weighted_fusion_layer_weight_initialization(self) -> None:
        """Test that fusion weight is properly initialized."""
        fusion_layer = WeightedFusionLayer()
        
        # Build the layer with dummy input
        dummy_input = tf.random.uniform((2, 500))
        _ = fusion_layer([dummy_input, dummy_input])  # Build the layer
        
        assert fusion_layer.fusion_weight is not None
        assert fusion_layer.fusion_weight.shape == (1,)
        # Weight should be initialized to 0.5
        assert tf.abs(fusion_layer.fusion_weight - 0.5) < 1e-6

    def test_weighted_fusion_layer_weight_constraint(self) -> None:
        """Test that fusion weight is constrained to [0, 1]."""
        fusion_layer = WeightedFusionLayer()
        
        dummy_input = tf.random.uniform((2, 500))
        _ = fusion_layer([dummy_input, dummy_input])
        
        weight_value = fusion_layer.fusion_weight.numpy()[0]
        assert 0.0 <= weight_value <= 1.0

    def test_weighted_fusion_layer_custom_name(self) -> None:
        """Test WeightedFusionLayer with custom name."""
        fusion_layer = WeightedFusionLayer(name="custom_fusion")
        assert fusion_layer.name == "custom_fusion"

    def test_channel1_branch_initialization(self) -> None:
        """Test Channel1Branch initialization."""
        branch = Channel1Branch(input_shape=(224, 224, 3))
        
        assert branch.name == "channel1_branch"
        assert branch.resize is not None
        assert branch.backbone is not None
        assert branch.fc1_1 is not None
        assert branch.fc1_2 is not None

    def test_channel1_branch_custom_input_shape(self) -> None:
        """Test Channel1Branch with custom input shape."""
        branch = Channel1Branch(input_shape=(256, 256, 3))
        dummy_input = tf.random.uniform((1, 256, 256, 3))
        output = branch(dummy_input, training=False)
        
        assert output.shape == (1, 500)

    def test_channel1_branch_training_mode(self) -> None:
        """Test Channel1Branch in training mode."""
        branch = Channel1Branch(input_shape=(224, 224, 3))
        dummy_input = tf.random.uniform((2, 224, 224, 3))
        
        output_training = branch(dummy_input, training=True)
        output_inference = branch(dummy_input, training=False)
        
        assert output_training.shape == (2, 500)
        assert output_inference.shape == (2, 500)

    def test_channel2_branch_initialization(self) -> None:
        """Test Channel2Branch initialization."""
        branch = Channel2Branch(input_shape=(224, 224, 3))
        
        assert branch.name == "channel2_branch"
        assert branch.backbone is not None
        assert branch.fc2_1 is not None
        assert branch.fc2_2 is not None

    def test_channel2_branch_custom_input_shape(self) -> None:
        """Test Channel2Branch with custom input shape."""
        branch = Channel2Branch(input_shape=(256, 256, 3))
        dummy_input = tf.random.uniform((1, 256, 256, 3))
        output = branch(dummy_input, training=False)
        
        assert output.shape == (1, 500)

    def test_channel2_branch_training_mode(self) -> None:
        """Test Channel2Branch in training mode."""
        branch = Channel2Branch(input_shape=(224, 224, 3))
        dummy_input = tf.random.uniform((2, 224, 224, 3))
        
        output_training = branch(dummy_input, training=True)
        output_inference = branch(dummy_input, training=False)
        
        assert output_training.shape == (2, 500)
        assert output_inference.shape == (2, 500)

    def test_model_custom_num_classes(self) -> None:
        """Test model with custom num_classes."""
        model = DualChannelDiabeticRetinopathyModel(num_classes=3)
        dummy_input1 = tf.random.uniform((1, 224, 224, 3))
        dummy_input2 = tf.random.uniform((1, 224, 224, 3))
        
        predictions = model([dummy_input1, dummy_input2], training=False)
        assert predictions.shape == (1, 3)

    def test_model_custom_input_shape(self) -> None:
        """Test model with custom input_shape."""
        model = DualChannelDiabeticRetinopathyModel(
            num_classes=5,
            input_shape=(256, 256, 3)
        )
        dummy_input1 = tf.random.uniform((1, 256, 256, 3))
        dummy_input2 = tf.random.uniform((1, 256, 256, 3))
        
        predictions = model([dummy_input1, dummy_input2], training=False)
        assert predictions.shape == (1, 5)

    def test_model_training_vs_inference_mode(
        self,
        model: DualChannelDiabeticRetinopathyModel,
        dummy_inputs: List[tf.Tensor]
    ) -> None:
        """Test model behavior in training vs inference mode."""
        output_training = model(dummy_inputs, training=True)
        output_inference = model(dummy_inputs, training=False)
        
        assert output_training.shape == output_inference.shape
        assert output_training.shape == (2, 5)

    def test_model_wrong_input_format_raises_error(
        self,
        model: DualChannelDiabeticRetinopathyModel
    ) -> None:
        """Test that model raises error for wrong input format."""
        # Single tensor instead of list
        dummy_input = tf.random.uniform((1, 224, 224, 3))
        
        with pytest.raises((ValueError, TypeError)):
            _ = model(dummy_input, training=False)

    def test_model_fusion_weight_is_trainable(
        self,
        model: DualChannelDiabeticRetinopathyModel,
        dummy_inputs: List[tf.Tensor],
        dummy_labels: tf.Tensor
    ) -> None:
        """Test that fusion weight is trainable."""
        # Build the model first
        _ = model(dummy_inputs, training=False)
        
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy"
        )
        
        # Get initial fusion weight
        initial_weight = model.fusion_layer.fusion_weight.numpy()[0]
        
        # Perform a training step
        model.fit(
            dummy_inputs,
            dummy_labels,
            epochs=1,
            verbose=0
        )
        
        # Get weight after training
        final_weight = model.fusion_layer.fusion_weight.numpy()[0]
        
        # Weight should be trainable
        assert model.fusion_layer.fusion_weight.trainable is True
        # Weight should be constrained to [0, 1]
        assert 0.0 <= final_weight <= 1.0

    def test_model_different_batch_sizes(
        self,
        model: DualChannelDiabeticRetinopathyModel
    ) -> None:
        """Test model with different batch sizes."""
        for batch_size in [1, 4, 8]:
            dummy_input1 = tf.random.uniform((batch_size, 224, 224, 3))
            dummy_input2 = tf.random.uniform((batch_size, 224, 224, 3))
            
            predictions = model([dummy_input1, dummy_input2], training=False)
            assert predictions.shape == (batch_size, 5)

    def test_weighted_fusion_formula(self) -> None:
        """Test that weighted fusion formula is correct."""
        fusion_layer = WeightedFusionLayer()
        
        # Create deterministic inputs
        fc1_2 = tf.ones((2, 500)) * 2.0  # All 2.0
        fc2_2 = tf.ones((2, 500)) * 3.0  # All 3.0
        
        fused = fusion_layer([fc1_2, fc2_2])
        
        # With weight = 0.5: fused = 0.5 * 2.0 + 0.5 * 3.0 = 2.5
        expected = tf.ones((2, 500)) * 2.5
        np.testing.assert_array_almost_equal(
            fused.numpy(),
            expected.numpy(),
            decimal=5
        )
