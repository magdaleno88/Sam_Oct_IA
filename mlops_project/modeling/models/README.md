# Model Architecture

This module contains a simple implementation of the dual-channel weighted fusion model for diabetic retinopathy detection.

## Quick Start

Create and use the model in just a few lines:

```python
from mlops_project.modeling.models import DualChannelDiabeticRetinopathyModel

# Create the model
model = DualChannelDiabeticRetinopathyModel(num_classes=5)

# Build it (initialize all layers)
dummy_input1 = tf.zeros((1, 224, 224, 3))  # CLAHE preprocessed image
dummy_input2 = tf.zeros((1, 224, 224, 3))  # CECED preprocessed image
_ = model([dummy_input1, dummy_input2])  # Build the model

# Compile for training
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train (example)
# model.fit([channel1_images, channel2_images], labels, epochs=50)
```

## How It Works

### 1. Dual-Channel Input
The model takes **two preprocessed images** from the same fundus scan:
- **Channel 1: CLAHE images** - Contrast-Limited Adaptive Histogram Equalization enhances image contrast
- **Channel 2: CECED images** - Contrast-Enhanced Canny Edge Detection highlights edges and boundaries

Both channels are derived from the same original fundus image using different preprocessing techniques.
**Note:** Preprocessing should be done before feeding images to the model.

### 2. Feature Extraction and Processing
Each preprocessed channel is processed through different pre-trained CNNs followed by fully connected layers (as per paper Figure 7):
- **Channel 1 (CLAHE)**: 
  - Input resized to **299×299** (for Inception V3)
  - **Inception V3** backbone extracts feature maps (conv only, no pooling)
  - **fc1_1**: GlobalAveragePooling2D (pools feature maps to vector)
  - **fc1_2**: Dense(500, relu) layer
  - Output: 500-dimensional feature vector
- **Channel 2 (CECED)**: 
  - **VGG-16** backbone extracts feature maps (conv only, no pooling)
  - **fc2_1**: GlobalAveragePooling2D (pools feature maps to vector)
  - **fc2_2**: Dense(500, relu) layer
  - Output: 500-dimensional feature vector
- Both backbones use weights pre-trained on ImageNet
- Both channels output 500-dimensional features for fusion

### 3. Weighted Fusion
The features from both channels are combined using a **learnable weight**:
- The model learns how much to trust channel 1 vs channel 2
- Formula: `fused = w * channel1_features + (1-w) * channel2_features`
- The weight `w` is learned during training

### 4. Final Classification
After weighted fusion, the features go through:
- **Classifier**: Dense(num_classes, activation="softmax")
- Output: probabilities for 5 DR severity levels (one-hot encoded)

## Code Structure

All model code is consolidated in a single file: `dual_channel_model.py` (~144 lines)

The file contains four simple classes:
1. `WeightedFusionLayer` - Combines features from both channels using a learnable weight
2. `Channel1Branch` - Processes CLAHE images with Inception V3
3. `Channel2Branch` - Processes CECED images with VGG-16
4. `DualChannelDiabeticRetinopathyModel` - Main model combining all components

This simple structure directly implements the paper's architecture without unnecessary complexity, configuration options, or multiple files.

## Model Components

### `DualChannelDiabeticRetinopathyModel`
Main model class. Inherits from `keras.Model`.

**Parameters:**
- `num_classes`: Number of DR severity levels (default: 5)
- `input_shape`: Image size (default: (224, 224, 3))
  - Note: The model automatically resizes Channel 1 inputs to 299×299 for Inception V3

**Architecture:** Matches Figure 7 of the paper exactly:
- Channel 1 outputs 500-dim features
- Channel 2 outputs 500-dim features
- Weighted fusion combines them
- Direct classification with softmax (no intermediate FC layer)

### `Channel1Branch`
Processes CLAHE preprocessed images using Inception V3 (as per paper Figure 7).
- Resizes input to 299×299
- Extracts feature maps using Inception V3 (no pooling in backbone)
- fc1_1: GlobalAveragePooling2D (pools feature maps to vector)
- fc1_2: Dense(500, relu)
- Outputs 500-dimensional features

### `Channel2Branch`
Processes CECED preprocessed images using VGG-16 (as per paper Figure 7).
- Extracts feature maps using VGG-16 (no pooling in backbone)
- fc2_1: GlobalAveragePooling2D (pools feature maps to vector)
- fc2_2: Dense(500, relu)
- Outputs 500-dimensional features

### `WeightedFusionLayer`
Custom layer that combines features from two channels using a learnable weight.

## Architecture Diagram

```
CHANNEL 1 (CLAHE)                          CHANNEL 2 (CECED)
─────────────────                          ─────────────────

CLAHE Image                                CECED Image
(224×224×3)                                (224×224×3)
    │                                           │
    ├─→ Resize to 299×299                      ├─→ VGG-16
    │                                           │   (feature maps)
    └─→ Inception V3                            │
        (feature maps)                           │
            │                                   │
            ├─→ fc1_1 (GlobalAvgPool)            ├─→ fc2_1 (GlobalAvgPool)
            │                                   │
            └─→ fc1_2 (Dense 500)                └─→ fc2_2 (Dense 500)
                │                                   │
                └───────────┬───────────────────────┘
                            │
                    Weighted Fusion
                    (500 features)
                            │
                            └─→ Classifier (Dense 5, softmax)
                                │
                                └─→ DR Severity Classification
```

**Note:** The architecture directly matches Figure 7 of the paper:
- Channel 1: CLAHE → Resize(299×299) → Inception V3 → GlobalAvgPool (fc1_1) → Dense(500) (fc1_2)
- Channel 2: CECED → VGG-16 → GlobalAvgPool (fc2_1) → Dense(500) (fc2_2)
- Both channels output 500-dimensional features
- Weighted fusion: f1 = w * fc1_2 + (1 - w) * fc2_2
- Direct classification: Dense(num_classes, softmax) on fused features

## Example Usage

```python
import tensorflow as tf
from mlops_project.modeling.models import DualChannelDiabeticRetinopathyModel

# Create model (matches paper Figure 7 exactly)
model = DualChannelDiabeticRetinopathyModel(num_classes=5)

# Prepare your data
# clahe_images: CLAHE preprocessed images, shape (batch_size, 224, 224, 3)
# ceced_images: CECED preprocessed images, shape (batch_size, 224, 224, 3)
# labels: one-hot encoded, shape (batch_size, 5)

# Build the model
_ = model([clahe_images[:1], ceced_images[:1]])

# Compile
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
history = model.fit(
    [clahe_images, ceced_images],  # Two inputs: CLAHE and CECED
    labels,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

# Predict
predictions = model.predict([clahe_test, ceced_test])
```

**Note:** The model expects preprocessed images. You need to apply CLAHE and CECED preprocessing to your fundus images before feeding them to the model.

## Understanding the Code

### For Beginners

**What is a CNN?**
- CNN (Convolutional Neural Network) = a type of neural network good at understanding images
- Inception V3 and VGG-16 are well-known CNN architectures used in this model
- Inception V3 is good at capturing complex patterns, VGG-16 is good at edge detection

**What is feature extraction?**
- Instead of using raw pixels, we extract meaningful patterns (edges, shapes, textures)
- These patterns are represented as numbers (feature vectors)

**What is weighted fusion?**
- A way to combine information from two sources
- The model learns which source is more important
- Like deciding how much to trust each of two opinions

**What is dropout?**
- A technique to prevent the model from memorizing the training data
- Randomly "turns off" some neurons during training
- Helps the model generalize better to new data

## References

- [Identification of Diabetic Retinopathy Using Weighted Fusion Deep Learning Based on Dual-Channel Fundus Scans](https://www.mdpi.com/2075-4418/12/2/540)
