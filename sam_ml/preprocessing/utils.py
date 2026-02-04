"""Utility functions for image preprocessing (padding, CLAHE, CECED)."""

from pathlib import Path
from typing import Union

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = hasattr(cv2, "imread") and callable(getattr(cv2, "imread", None))
except Exception:
    cv2 = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False


def load_image_bgr(path: Union[str, Path]) -> np.ndarray | None:
    """Load an image as BGR numpy array (H, W, 3). Uses OpenCV when available, else PIL."""
    path = Path(path)
    if _CV2_AVAILABLE and cv2 is not None:
        img = cv2.imread(str(path))
        if img is not None:
            return img
    try:
        from PIL import Image
        pil_img = Image.open(path).convert("RGB")
        arr = np.array(pil_img)
        return arr[:, :, ::-1].copy()  # RGB -> BGR
    except Exception:
        return None


def save_image_bgr(path: Union[str, Path], img_bgr: np.ndarray) -> bool:
    """Save a BGR numpy array as image. Uses OpenCV when available, else PIL."""
    path = Path(path)
    if _CV2_AVAILABLE and cv2 is not None:
        return cv2.imwrite(str(path), img_bgr)
    try:
        from PIL import Image
        rgb = img_bgr[:, :, ::-1].copy()
        Image.fromarray(rgb).save(path, "JPEG", quality=95)
        return True
    except Exception:
        return False


def resize_bgr(img_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize BGR image to (width, height). Uses OpenCV when available, else PIL."""
    if _CV2_AVAILABLE and cv2 is not None:
        return cv2.resize(
            img_bgr,
            (width, height),
            interpolation=cv2.INTER_LANCZOS4,
        )
    from PIL import Image
    pil_rgb = Image.fromarray(img_bgr[:, :, ::-1].copy())
    pil_resized = pil_rgb.resize((width, height), Image.Resampling.LANCZOS)
    return np.array(pil_resized)[:, :, ::-1].copy()


def add_padding_to_square_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """Add padding to make a BGR image square.

    Pads the smaller dimension (width or height) to match the larger one.
    Uses black padding (BGR 0, 0, 0). Input and output are OpenCV BGR (H, W, C).

    Args:
        img_bgr: Input image in BGR format (numpy array, H x W x 3).

    Returns:
        Square BGR image with padding added (numpy array).
    """
    h, w = img_bgr.shape[:2]
    if w == h:
        return img_bgr
    target = max(w, h)
    if img_bgr.ndim == 2:
        square = np.zeros((target, target), dtype=img_bgr.dtype)
    else:
        square = np.zeros((target, target, img_bgr.shape[2]), dtype=img_bgr.dtype)
    y_offset = (target - h) // 2
    x_offset = (target - w) // 2
    square[y_offset : y_offset + h, x_offset : x_offset + w] = img_bgr
    return square
