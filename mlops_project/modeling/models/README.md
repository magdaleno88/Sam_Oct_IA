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
Each preprocessed channel is processed through different pre-trained CNNs followed by fully connected layers (as per paper diagram):
- **Channel 1 (CLAHE)**: 
  - **Inception V3** extracts feature vector **fv1** (2048 dimensions)
  - **fc1_1**: First fully connected layer (dimensions per paper)
  - **fc1_2**: Second fully connected layer (dimensions per paper)
- **Channel 2 (CECED)**: 
  - **VGG-16** extracts feature vector **fv2** (512 dimensions)
  - **fc2_1**: First fully connected layer (dimensions per paper)
  - **fc2_2**: Second fully connected layer (dimensions per paper)
- Both backbones use weights pre-trained on ImageNet
- The model automatically resizes inputs to match each backbone's requirements (Inception V3: 299x299, VGG-16: 224x224)

**Note:** 
- **fv1** and **fv2** are the feature vectors (outputs) from the CNN backbones
- The FC layer dimensions (fc1_1, fc1_2, fc2_1, fc2_2, f1) should be verified from the paper's methodology section
- Current implementation uses reasonable defaults (512, 256) but these can be adjusted via constructor parameters

### 3. Weighted Fusion
The features from both channels are combined using a **learnable weight**:
- The model learns how much to trust channel 1 vs channel 2
- Formula: `fused = w * channel1_features + (1-w) * channel2_features`
- The weight `w` is learned during training

### 4. Final Classification
After fusion, the features go through:
- **f1**: Final fully connected layer (256 dim)
- **Softmax**: Classification layer
- Output: probabilities for 5 DR severity levels

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
  - Note: The model automatically resizes inputs internally (Inception V3 needs 299x299, VGG-16 uses 224x224)
- `fc1_1_dim`, `fc1_2_dim`: FC layer dimensions for Channel 1 (default: 512, 256)
- `fc2_1_dim`, `fc2_2_dim`: FC layer dimensions for Channel 2 (default: 512, 256)
- `f1_dim`: Final FC layer dimension before softmax (default: 256)
  
**Note:** FC layer dimensions should be verified from the paper's methodology section. Defaults are reasonable but may not match the paper exactly.

### `Channel1Branch`
Processes CLAHE preprocessed images using Inception V3.
- Resizes input to 299x299
- Extracts features using Inception V3 (fv1: 2048 dim)
- Processes through fc1_1 (512 dim) → fc1_2 (256 dim)
- Outputs 256-dimensional features

### `Channel2Branch`
Processes CECED preprocessed images using VGG-16.
- Extracts features using VGG-16 (fv2: 512 dim)
- Processes through fc2_1 (512 dim) → fc2_2 (256 dim)
- Outputs 256-dimensional features

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
    │                                           │   (fv2: 512)
    └─→ Inception V3                            │
        (fv1: 2048)                             │
            │                                   │
            ├─→ fc1_1 (512)                     ├─→ fc2_1 (512)
            │                                   │
            └─→ fc1_2 (256)                     └─→ fc2_2 (256)
                │                                   │
                └───────────┬───────────────────────┘
                            │
                    Weighted Fusion
                    (256 features)
                            │
                            ├─→ f1 (256)
                            │
                            └─→ Softmax (5 classes)
                                │
                                └─→ DR Severity Classification
```

**Note:** The architecture directly matches the paper diagram:
- Channel 1: CLAHE → Inception V3 → fc1_1 → fc1_2
- Channel 2: CECED → VGG-16 → fc2_1 → fc2_2
- Each channel has its own FC layers before fusion (fc1_1, fc1_2 vs fc2_1, fc2_2)
- FC layers can have different configurations per channel (fv1 starts at 2048, fv2 starts at 512)
- After fusion: f1 → Softmax for final classification

**Important:** 
- **fv1** = feature vector from Inception V3 (2048 dim) - this is the backbone output
- **fv2** = feature vector from VGG-16 (512 dim) - this is the backbone output
- The FC layer dimensions (fc1_1, fc1_2, fc2_1, fc2_2, f1) shown in the diagram are **reasonable defaults** (512, 256)
- **The paper should be consulted** for the exact dimensions specified in the methodology section
- These dimensions can be adjusted via constructor parameters if needed

## Example Usage

```python
import tensorflow as tf
from mlops_project.modeling.models import DualChannelDiabeticRetinopathyModel

# Create model
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
