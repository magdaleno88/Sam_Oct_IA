"""Model architectures for diabetic retinopathy detection."""

from sam_ml.modeling.models.dual_channel_model import (
    DualChannelDiabeticRetinopathyModel,
    WeightedFusionLayer,
)

__all__ = [
    "DualChannelDiabeticRetinopathyModel",
    "WeightedFusionLayer",
]
