"""Utility functions for image preprocessing (CLAHE and CECED)."""

import cv2
import numpy as np


def apply_clahe_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast-Limited Adaptive Histogram Equalization) to BGR image.
    
    Args:
        img_bgr: Input image in BGR format (numpy array)
        
    Returns:
        Processed image in BGR format
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return img_clahe


def apply_ceced_bgr_3ch(
    img_bgr: np.ndarray,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: tuple[int, int] = (8, 8),
    blur_kernel_size: int = 7,
    auto_canny_sigma: float = 0.20,
    dilate_iterations: int = 2,
) -> np.ndarray:
    """
    Apply a softer, retina-friendly version of CECED (Contrast-Enhanced Canny Edge Detection)
    to a BGR fundus image.

    This variant:
    - Works on the green channel (better vessel contrast).
    - Uses CLAHE but avoids over-normalization.
    - Uses auto Canny thresholds based on image statistics (median).
    - Optionally dilates edges to make vessels more visible.
    
    Args:
        img_bgr: Input image in BGR format (H, W, 3).
        clahe_clip_limit: CLAHE clip limit.
        clahe_tile_grid_size: CLAHE tile grid size.
        blur_kernel_size: Gaussian blur kernel size (must be odd).
        auto_canny_sigma: Sigma for auto Canny threshold computation.
        dilate_iterations: Number of dilation iterations to thicken edges (0 = no dilation).

    Returns:
        3-channel edge image in BGR format (edges replicated in all channels).
    """
    # 1) Use the green channel (usually best contrast for vessels)
    b, g, r = cv2.split(img_bgr)
    gray = g

    # 2) Apply CLAHE to enhance local contrast
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit,
        tileGridSize=clahe_tile_grid_size
    )
    gray_eq = clahe.apply(gray)

    # 3) Gentle smoothing to reduce noise but keep thin vessels
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1  # ensure odd
    blurred = cv2.GaussianBlur(gray_eq, (blur_kernel_size, blur_kernel_size), 0)

    # 4) Auto Canny thresholds based on the median intensity
    v = np.median(blurred)
    lower = int(max(0, (1.0 - auto_canny_sigma) * v))
    upper = int(min(255, (1.0 + auto_canny_sigma) * v))

    edges = cv2.Canny(blurred, lower, upper)

    # 5) Optionally dilate edges to make vessels more visible to the CNN
    if dilate_iterations > 0:
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=dilate_iterations)

    # 6) Convert to 3-channel (BGR) so it matches expected input shape
    edges_3ch = cv2.merge((edges, edges, edges))
    return edges_3ch

