"""Middleware for per-image preprocessing (BGR in, list of (output_key, BGR) out)."""

from abc import ABC, abstractmethod
from typing import Any, Callable, TypedDict

import numpy as np

from sam_ml.preprocessing.utils import add_padding_to_square_bgr, resize_bgr


class MiddlewareContext(TypedDict, total=False):
    """Context passed to middleware process(); all keys optional."""

    min_size: int
    target_size: tuple[int, int]
    clahe_target_size: tuple[int, int]
    ceced_target_size: tuple[int, int]


# Registry: key -> middleware class (not instance)
MIDDLEWARE_REGISTRY: dict[str, type["BaseMiddleware"]] = {}


def register_middleware(key: str) -> Callable[[type["BaseMiddleware"]], type["BaseMiddleware"]]:
    """Decorator to register a middleware class in the registry."""

    def decorator(cls: type["BaseMiddleware"]) -> type["BaseMiddleware"]:
        if key in MIDDLEWARE_REGISTRY:
            raise ValueError(f"Middleware key '{key}' is already registered")
        MIDDLEWARE_REGISTRY[key] = cls
        return cls

    return decorator


def get_middleware(key: str, **kwargs: Any) -> "BaseMiddleware":
    """Get a middleware instance from the registry."""
    if key not in MIDDLEWARE_REGISTRY:
        available = ", ".join(MIDDLEWARE_REGISTRY.keys()) if MIDDLEWARE_REGISTRY else "none"
        raise KeyError(
            f"Middleware '{key}' not found in registry. Available: {available}"
        )
    return MIDDLEWARE_REGISTRY[key](**kwargs)


def list_middlewares() -> list[str]:
    """List all registered middleware keys."""
    return list(MIDDLEWARE_REGISTRY.keys())


class BaseMiddleware(ABC):
    """Abstract base for per-image transformation middleware.

    Contract: receive BGR image (numpy array), filename, and context;
    return a list of (output_key, BGR array) pairs. Each output_key
    maps to a subdirectory under processed_dir (e.g. "images", "paper_orig").
    """

    @abstractmethod
    def process(
        self,
        img_bgr: np.ndarray,
        filename: str,
        context: MiddlewareContext,
    ) -> list[tuple[str, np.ndarray]]:
        """Process one image and return zero or more (output_key, BGR image) pairs.

        Args:
            img_bgr: Input image in BGR format (H, W, C).
            filename: Original filename (for logging or naming).
            context: Optional min_size, target_size, etc.

        Returns:
            List of (output_key, img_bgr). output_key is the subdir name
            under processed_dir (e.g. "images"). Empty list means skip this image.
        """
        ...


@register_middleware("default")
class DefaultMiddleware(BaseMiddleware):
    """Default pipeline: min-size filter, pad to square, resize to target_size.

    Returns a single output key (config default_output_subdir, typically "images").
    Skips images that are too small or would require upscaling (returns []).
    """

    def __init__(
        self,
        min_size: int = 512,
        target_size: tuple[int, int] = (512, 512),
        output_key: str = "images",
    ) -> None:
        self.min_size = min_size
        self.target_size = target_size
        self.output_key = output_key

    def process(
        self,
        img_bgr: np.ndarray,
        filename: str,
        context: MiddlewareContext,
    ) -> list[tuple[str, np.ndarray]]:
        min_size = context.get("min_size", self.min_size)
        target_size = context.get("target_size", self.target_size)
        h, w = img_bgr.shape[:2]
        if w < min_size or h < min_size:
            return []
        img_square = add_padding_to_square_bgr(img_bgr)
        ch, cw = img_square.shape[:2]
        if ch < target_size[0] or cw < target_size[1]:
            return []
        resized = resize_bgr(img_square, target_size[0], target_size[1])
        return [(self.output_key, resized)]


@register_middleware("paper_dual")
class PaperDualMiddleware(BaseMiddleware):
    """Produce resized original + CLAHE + CECED (BGR) for paper replication.

    Returns three outputs: output_key (resized original), output_key + '_clahe',
    output_key + '_ceced'. Uses apply_clahe_bgr and apply_ceced_bgr from
    sam_ml.preprocessing.filters (requires OpenCV).
    """

    def __init__(
        self,
        min_size: int = 512,
        target_size: tuple[int, int] = (512, 512),
        output_key: str = "images",
    ) -> None:
        self.min_size = min_size
        self.target_size = target_size
        self.output_key = output_key

    def process(
        self,
        img_bgr: np.ndarray,
        filename: str,
        context: MiddlewareContext,
    ) -> list[tuple[str, np.ndarray]]:
        min_size = context.get("min_size", self.min_size)
        target_size = context.get("target_size", self.target_size)
        h, w = img_bgr.shape[:2]
        if w < min_size or h < min_size:
            return []
        img_square = add_padding_to_square_bgr(img_bgr)
        ch, cw = img_square.shape[:2]
        if ch < target_size[0] or cw < target_size[1]:
            return []
        resized = resize_bgr(img_square, target_size[0], target_size[1])

        from sam_ml.preprocessing.filters import apply_clahe_bgr, apply_ceced_bgr

        clahe_bgr = apply_clahe_bgr(resized)
        ceced_bgr = apply_ceced_bgr(resized)
        return [
            (self.output_key, resized),
            (f"{self.output_key}_clahe", clahe_bgr),
            (f"{self.output_key}_ceced", ceced_bgr),
        ]


@register_middleware("dual_filters_multisize")
class DualFiltersMultisizeMiddleware(BaseMiddleware):
    """Produce synchronized CLAHE and CECED outputs with different target sizes."""

    def __init__(
        self,
        min_size: int = 512,
        clahe_target_size: tuple[int, int] = (299, 299),
        ceced_target_size: tuple[int, int] = (224, 224),
        clahe_output_key: str = "images_clahe",
        ceced_output_key: str = "images_ceced",
    ) -> None:
        self.min_size = min_size
        self.clahe_target_size = clahe_target_size
        self.ceced_target_size = ceced_target_size
        self.clahe_output_key = clahe_output_key
        self.ceced_output_key = ceced_output_key

    def process(
        self,
        img_bgr: np.ndarray,
        filename: str,
        context: MiddlewareContext,
    ) -> list[tuple[str, np.ndarray]]:
        min_size = context.get("min_size", self.min_size)
        clahe_target_size = context.get("clahe_target_size", self.clahe_target_size)
        ceced_target_size = context.get("ceced_target_size", self.ceced_target_size)

        # Enforce no-upscaling for either branch.
        max_target = max(
            clahe_target_size[0],
            clahe_target_size[1],
            ceced_target_size[0],
            ceced_target_size[1],
        )

        h, w = img_bgr.shape[:2]
        if w < min_size or h < min_size:
            return []

        img_square = add_padding_to_square_bgr(img_bgr)
        ch, cw = img_square.shape[:2]
        if ch < max_target or cw < max_target:
            return []

        from sam_ml.preprocessing.filters import apply_clahe_bgr, apply_ceced_bgr

        clahe_resized = resize_bgr(img_square, clahe_target_size[0], clahe_target_size[1])
        ceced_resized = resize_bgr(img_square, ceced_target_size[0], ceced_target_size[1])

        clahe_bgr = apply_clahe_bgr(clahe_resized)
        ceced_bgr = apply_ceced_bgr(ceced_resized)
        return [
            (self.clahe_output_key, clahe_bgr),
            (self.ceced_output_key, ceced_bgr),
        ]


@register_middleware("resize_norm")
class ResizeNormMiddleware(BaseMiddleware):
    """Resize to target_size then normalize to [0, 1] and back to uint8 BGR for saving.

    Single output (default_output_subdir). Same min_size/pad/resize as default;
    no extra normalization step that changes the saved image (output remains BGR uint8).
    """

    def __init__(
        self,
        min_size: int = 512,
        target_size: tuple[int, int] = (512, 512),
        output_key: str = "images",
    ) -> None:
        self.min_size = min_size
        self.target_size = target_size
        self.output_key = output_key

    def process(
        self,
        img_bgr: np.ndarray,
        filename: str,
        context: MiddlewareContext,
    ) -> list[tuple[str, np.ndarray]]:
        min_size = context.get("min_size", self.min_size)
        target_size = context.get("target_size", self.target_size)
        h, w = img_bgr.shape[:2]
        if w < min_size or h < min_size:
            return []
        img_square = add_padding_to_square_bgr(img_bgr)
        ch, cw = img_square.shape[:2]
        if ch < target_size[0] or cw < target_size[1]:
            return []
        resized = resize_bgr(img_square, target_size[0], target_size[1])
        # Normalize to [0,1] then back to uint8 BGR for consistent I/O
        out = (resized.astype(np.float64) / 255.0)
        out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
        return [(self.output_key, out)]
