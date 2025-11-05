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
        
    def build(self, input_shape) -> None:
        # input_shape may be a list/tuple of shapes (one per channel)
        # We don't need to use it, just keep the weight definition
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
    Channel 1: CLAHE images → Inception V3 → GlobalAvgPool (fc1_1) → Dense(500) (fc1_2).
    
    Matches Figure 7 of the paper:
    CLAHE image → resized to 299×299 → InceptionV3 → GlobalAveragePooling2D → Dense(500)
    """
    
    def __init__(
        self, 
        input_shape: Tuple[int, int, int] = (224, 224, 3)
    ) -> None:
        super().__init__(name="channel1_branch")
        
        self.resize = layers.Resizing(299, 299, name="channel1_resize")
        self.backbone = InceptionV3(
            include_top=False,
            weights="imagenet",
            input_shape=(299, 299, input_shape[2]),
            pooling=None,
            name="channel1_inception_v3"
        )
        self.fc1_1 = layers.GlobalAveragePooling2D(name="fc1_1")
        self.fc1_2 = layers.Dense(500, activation="relu", name="fc1_2")
    
    def call(
        self,
        inputs: tf.Tensor,
        training: Optional[bool] = None
    ) -> tf.Tensor:
        x = self.resize(inputs)
        x = self.backbone(x, training=training)
        x = self.fc1_1(x)
        x = self.fc1_2(x, training=training)
        return x


class Channel2Branch(keras.Model):
    """
    Channel 2: CECED images → VGG-16 → GlobalAvgPool (fc2_1) → Dense(500) (fc2_2).
    
    Matches Figure 7 of the paper:
    CECED image → VGG16 → GlobalAveragePooling2D → Dense(500)
    """
    
    def __init__(
        self, 
        input_shape: Tuple[int, int, int] = (224, 224, 3)
    ) -> None:
        super().__init__(name="channel2_branch")
        
        self.backbone = VGG16(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling=None,
            name="channel2_vgg16"
        )
        self.fc2_1 = layers.GlobalAveragePooling2D(name="fc2_1")
        self.fc2_2 = layers.Dense(500, activation="relu", name="fc2_2")
    
    def call(
        self,
        inputs: tf.Tensor,
        training: Optional[bool] = None
    ) -> tf.Tensor:
        x = self.backbone(inputs, training=training)
        x = self.fc2_1(x)
        x = self.fc2_2(x, training=training)
        return x


class DualChannelDiabeticRetinopathyModel(keras.Model):
    """
    Dual-channel model for diabetic retinopathy detection.
    
    Architecture (as per paper Figure 7):
    - Channel 1 (CLAHE): InceptionV3 → GlobalAvgPool (fc1_1) → Dense(500) (fc1_2)
    - Channel 2 (CECED): VGG16 → GlobalAvgPool (fc2_1) → Dense(500) (fc2_2)
    - Weighted fusion: f1 = w * fc1_2 + (1 - w) * fc2_2
    - Classifier: Dense(num_classes, activation="softmax")(f1)
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        input_shape: Tuple[int, int, int] = (224, 224, 3)
    ) -> None:
        super().__init__(name="dual_channel_dr_model")
        
        self.channel1_branch = Channel1Branch(input_shape=input_shape)
        self.channel2_branch = Channel2Branch(input_shape=input_shape)
        self.fusion_layer = WeightedFusionLayer()
        self.classifier = layers.Dense(num_classes, activation="softmax", name="classifier")
    
    def call(
        self,
        inputs: List[tf.Tensor],
        training: Optional[bool] = None
    ) -> tf.Tensor:
        channel1_image, channel2_image = inputs
        
        fc1_2 = self.channel1_branch(channel1_image, training=training)
        fc2_2 = self.channel2_branch(channel2_image, training=training)
        fused = self.fusion_layer([fc1_2, fc2_2])
        predictions = self.classifier(fused)
        
        return predictions
