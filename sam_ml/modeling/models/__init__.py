"""Model architectures for diabetic retinopathy detection."""

from sam_ml.modeling.models.base import BaseLightningModel
from sam_ml.modeling.models.registry import get_model, list_models, register_model

# Import models to register them
from sam_ml.modeling.models import simple_cnn_lightning  # noqa: F401

__all__ = [
    "BaseLightningModel",
    "get_model",
    "list_models",
    "register_model",
]
