# Dataset Structure

This folder contains the dataset for training and evaluating the dual-channel diabetic retinopathy detection model.

## Expected Dataset Structure

The dataset should be organized in the following structure to enable easy loading using TensorFlow's `image_dataset_from_directory`:

```
data/
└── processed/
    └── dataset_v1/
        ├── CLAHE/
        │   ├── train/
        │   │   ├── 0_no_DR/
        │   │   ├── 1_mild/
        │   │   ├── 2_moderate/
        │   │   ├── 3_severe/
        │   │   └── 4_proliferative/
        │   ├── val/
        │   │   ├── 0_no_DR/
        │   │   ├── 1_mild/
        │   │   ├── 2_moderate/
        │   │   ├── 3_severe/
        │   │   └── 4_proliferative/
        │   └── test/
        │       ├── 0_no_DR/
        │       ├── 1_mild/
        │       ├── 2_moderate/
        │       ├── 3_severe/
        │       └── 4_proliferative/
        │
        └── CECED/
            ├── train/
            │   ├── 0_no_DR/
            │   ├── 1_mild/
            │   ├── 2_moderate/
            │   ├── 3_severe/
            │   └── 4_proliferative/
            ├── val/
            │   ├── 0_no_DR/
            │   ├── 1_mild/
            │   ├── 2_moderate/
            │   ├── 3_severe/
            │   └── 4_proliferative/
            └── test/
                ├── 0_no_DR/
                ├── 1_mild/
                ├── 2_moderate/
                ├── 3_severe/
                └── 4_proliferative/
```

## Dataset Details

- **CLAHE**: Contrast-Limited Adaptive Histogram Equalization preprocessed images
- **CECED**: Contrast-Enhanced Canny Edge Detection preprocessed images
- **5 Classes**: 
  - `0_no_DR`: No diabetic retinopathy
  - `1_mild`: Mild diabetic retinopathy
  - `2_moderate`: Moderate diabetic retinopathy
  - `3_severe`: Severe diabetic retinopathy
  - `4_proliferative`: Proliferative diabetic retinopathy
- **Splits**: train, validation (val), and test sets

## Loading the Dataset

Example code to load the dataset using TensorFlow:

```python
import tensorflow as tf

base_path = "data/processed/dataset_v1"
img_size_clahe = (299, 299)   # InceptionV3 input size
img_size_ceced = (224, 224)   # VGG16 input size
batch_size = 32

# CLAHE dataset
clahe_train = tf.keras.utils.image_dataset_from_directory(
    f"{base_path}/CLAHE/train",
    image_size=img_size_clahe,
    batch_size=batch_size,
    label_mode="categorical"
)

clahe_val = tf.keras.utils.image_dataset_from_directory(
    f"{base_path}/CLAHE/val",
    image_size=img_size_clahe,
    batch_size=batch_size,
    label_mode="categorical"
)

clahe_test = tf.keras.utils.image_dataset_from_directory(
    f"{base_path}/CLAHE/test",
    image_size=img_size_clahe,
    batch_size=batch_size,
    label_mode="categorical"
)

# CECED dataset
ceced_train = tf.keras.utils.image_dataset_from_directory(
    f"{base_path}/CECED/train",
    image_size=img_size_ceced,
    batch_size=batch_size,
    label_mode="categorical"
)

ceced_val = tf.keras.utils.image_dataset_from_directory(
    f"{base_path}/CECED/val",
    image_size=img_size_ceced,
    batch_size=batch_size,
    label_mode="categorical"
)

ceced_test = tf.keras.utils.image_dataset_from_directory(
    f"{base_path}/CECED/test",
    image_size=img_size_ceced,
    batch_size=batch_size,
    label_mode="categorical"
)

# Combine datasets for dual-channel model
# Note: Ensure both datasets have matching samples and order
train_dataset = tf.data.Dataset.zip((clahe_train, ceced_train))
val_dataset = tf.data.Dataset.zip((clahe_val, ceced_val))
test_dataset = tf.data.Dataset.zip((clahe_test, ceced_test))
```

## Status

⚠️ **Pending Implementation**: This dataset structure is planned but not yet implemented. The dataset needs to be organized according to this structure before training can begin.

## Notes

- Image files should be placed in the corresponding class folders
- Both CLAHE and CECED datasets should have the same number of images and matching order
- The folder names (`0_no_DR`, `1_mild`, etc.) are used as class labels by TensorFlow
- Labels are automatically converted to categorical (one-hot encoded) format with `label_mode="categorical"`

