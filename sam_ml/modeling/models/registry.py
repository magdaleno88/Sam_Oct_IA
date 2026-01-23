"""Model registry for managing all available models."""

from typing import Any, Callable, Dict

from sam_ml.modeling.models.base import BaseLightningModel


# Model registry dictionary
# Key: unique model identifier (string)
# Value: function that creates the model instance
MODEL_REGISTRY: Dict[str, Callable[..., BaseLightningModel]] = {}


def register_model(key: str) -> Callable:
    """
    Decorator to register a model in the registry.
    
    Usage:
        @register_model("simple_cnn")
        def create_simple_cnn(**kwargs) -> BaseLightningModel:
            return SimpleCNNLightning(**kwargs)
    
    Args:
        key: Unique identifier for the model
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., BaseLightningModel]) -> Callable:
        if key in MODEL_REGISTRY:
            raise ValueError(f"Model key '{key}' is already registered!")
        MODEL_REGISTRY[key] = func
        return func
    return decorator


def get_model(key: str, **kwargs: Any) -> BaseLightningModel:
    """
    Get a model instance from the registry.
    
    Args:
        key: Model identifier
        **kwargs: Arguments to pass to model constructor
        
    Returns:
        Model instance
        
    Raises:
        KeyError: If model key is not found
    """
    if key not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys()) if MODEL_REGISTRY else "none"
        raise KeyError(
            f"Model '{key}' not found in registry. "
            f"Available models: {available}"
        )
    
    return MODEL_REGISTRY[key](**kwargs)


def list_models() -> list[str]:
    """
    List all registered model keys.
    
    Returns:
        List of model keys
    """
    return list(MODEL_REGISTRY.keys())
