"""Dual-channel model for diabetic retinopathy detection."""

from typing import List, Optional, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import InceptionV3, VGG16


class WeightedFusionLayer(layers.Layer):
    """Combines features from two channels using a learnable weight."""
    
    def __init__(self, name: str = "weighted_fusion", **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.fusion_weight: Optional[tf.Variable] = None
        
    def build(self, input_shape: Tuple[int, ...]) -> None:
        self.fusion_weight = self.add_weight(
            name="fusion_weight",
            shape=(1,),
            initializer=keras.initializers.Constant(0.5),
            trainable=True,
            constraint=keras.constraints.MinMaxNorm(min_value=0.0, max_value=1.0)
        )
        super().build(input_shape)
        
    def call(
        self, 
        inputs: List[tf.Tensor], 
        training: Optional[bool] = None
    ) -> tf.Tensor:
        channel1, channel2 = inputs
        
        if channel1.shape != channel2.shape:
            raise ValueError(
                f"Channel shapes must match. Got {channel1.shape} and {channel2.shape}"
            )
        
        fused = self.fusion_weight * channel1 + (1 - self.fusion_weight) * channel2
        return fused


class Channel1Branch(keras.Model):
    """
    Channel 1: CLAHE images → Inception V3 → fc1_1 → fc1_2.
    
    fv1 = feature vector from Inception V3 (2048 dimensions)
    fc1_1, fc1_2 = fully connected layers (dimensions should be verified from paper)
    """
    
    def __init__(
        self, 
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        fc1_1_dim: int = 512,
        fc1_2_dim: int = 256
    ) -> None:
        super().__init__(name="channel1_branch")
        
        self.resize = layers.Resizing(299, 299, name="channel1_resize")
        self.backbone = InceptionV3(
            include_top=False,
            weights="imagenet",
            input_shape=(299, 299, input_shape[2]),
            pooling="avg",
            name="channel1_inception_v3"
        )
        # FC layers after feature extraction (as per paper diagram)
        # fv1 (2048) → fc1_1 → fc1_2
        # Note: Dimensions should be verified from the paper's methodology section
        self.fc1_1 = layers.Dense(fc1_1_dim, activation="relu", name="fc1_1")
        self.fc1_2 = layers.Dense(fc1_2_dim, activation="relu", name="fc1_2")
    
    def call(
        self,
        inputs: tf.Tensor,
        training: Optional[bool] = None
    ) -> tf.Tensor:
        x = self.resize(inputs)
        fv1 = self.backbone(x, training=training)  # 2048 features
        x = self.fc1_1(fv1, training=training)    # 512 features
        x = self.fc1_2(x, training=training)       # 256 features
        return x


class Channel2Branch(keras.Model):
    """
    Channel 2: CECED images → VGG-16 → fc2_1 → fc2_2.
    
    fv2 = feature vector from VGG-16 (512 dimensions)
    fc2_1, fc2_2 = fully connected layers (dimensions should be verified from paper)
    """
    
    def __init__(
        self, 
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        fc2_1_dim: int = 512,
        fc2_2_dim: int = 256
    ) -> None:
        super().__init__(name="channel2_branch")
        
        self.backbone = VGG16(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling="avg",
            name="channel2_vgg16"
        )
        # FC layers after feature extraction (as per paper diagram)
        # fv2 (512) → fc2_1 → fc2_2
        # Note: Dimensions should be verified from the paper's methodology section
        # Note: fc2_1 and fc2_2 can have different dimensions than fc1_1 and fc1_2
        self.fc2_1 = layers.Dense(fc2_1_dim, activation="relu", name="fc2_1")
        self.fc2_2 = layers.Dense(fc2_2_dim, activation="relu", name="fc2_2")
    
    def call(
        self,
        inputs: tf.Tensor,
        training: Optional[bool] = None
    ) -> tf.Tensor:
        fv2 = self.backbone(inputs, training=training)  # 512 features
        x = self.fc2_1(fv2, training=training)          # 512 features
        x = self.fc2_2(x, training=training)            # 256 features
        return x


class DualChannelDiabeticRetinopathyModel(keras.Model):
    """
    Dual-channel model for diabetic retinopathy detection.
    
    Architecture (as per paper diagram):
    - Channel 1 (CLAHE): Inception V3 (fv1) → fc1_1 → fc1_2
    - Channel 2 (CECED): VGG-16 (fv2) → fc2_1 → fc2_2
    - Weighted fusion of fc1_2 and fc2_2
    - Final FC layer (f1) → Softmax
    
    Note: fv1 and fv2 are feature vectors extracted from the backbones.
    FC layer dimensions should be verified from the paper's methodology section.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        fc1_1_dim: int = 512,
        fc1_2_dim: int = 256,
        fc2_1_dim: int = 512,
        fc2_2_dim: int = 256,
        f1_dim: int = 256
    ) -> None:
        super().__init__(name="dual_channel_dr_model")
        
        self.num_classes: int = num_classes
        self.input_shape: Tuple[int, int, int] = input_shape
        
        self.channel1_branch = Channel1Branch(
            input_shape=input_shape,
            fc1_1_dim=fc1_1_dim,
            fc1_2_dim=fc1_2_dim
        )
        self.channel2_branch = Channel2Branch(
            input_shape=input_shape,
            fc2_1_dim=fc2_1_dim,
            fc2_2_dim=fc2_2_dim
        )
        self.fusion_layer = WeightedFusionLayer()
        
        # Final FC layer (f1) before softmax (as per paper diagram)
        # Dimension should be verified from the paper
        self.f1 = layers.Dense(f1_dim, activation="relu", name="f1")
        self.classifier = layers.Dense(num_classes, name="classifier")
        self.softmax = layers.Activation("softmax", name="softmax")
    
    def call(
        self,
        inputs: List[tf.Tensor],
        training: Optional[bool] = None
    ) -> tf.Tensor:
        channel1_image, channel2_image = inputs
        
        # Process through each channel: fv → fc1 → fc2
        fc1_2 = self.channel1_branch(channel1_image, training=training)  # 256 features
        fc2_2 = self.channel2_branch(channel2_image, training=training)  # 256 features
        
        # Weighted fusion of fc1_2 and fc2_2
        fused = self.fusion_layer([fc1_2, fc2_2])  # 256 features
        
        # Final FC layer (f1) before classification
        x = self.f1(fused, training=training)      # 256 features
        x = self.classifier(x)                     # num_classes
        predictions = self.softmax(x)              # softmax probabilities
        
        return predictions
