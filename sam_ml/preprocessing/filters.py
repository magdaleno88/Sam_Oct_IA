"""Utility functions for image preprocessing (CLAHE and CECED).

For preprocessing/filters we recommend using CLAHE (apply_clahe_bgr) to improve
local contrast in fundus images with minimal risk of damaging prediction.
"""

import numpy as np
import cv2


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


def apply_ceced_bgr(
    img_bgr: np.ndarray,
    clahe_clip_limit: float = 9.0,
    clahe_tile_grid_size: tuple[int, int] = (16, 16),
    blur_kernel_size: int = 7,
    auto_canny_sigma: float = 0.05,
    dilate_iterations: int = 0,
    use_l2_gradient: bool = False,
) -> np.ndarray:
    """
    Apply CECED (Contrast-Enhanced Canny Edge Detection) to a BGR fundus image.
    
    Simplified & Robust Pipeline (Validated via Notebook):
    1. Grayscale Conversion (Standard).
    2. Strong CLAHE (Clip 9.0, Grid 16) to maximize vein contrast.
    3. Gaussian Blur (7x7) to reduce noise.
    4. Robust Auto-Canny (Sigma 0.05):
       - Uses masked median (ignoring black background) to determine thresholds.
       - Very tight sigma (0.05) for high sensitivity to the strong CLAHE edges.
    5. No Dilation (0) by default to keep veins thin.

    Args:
        img_bgr: Input image in BGR format (H, W, 3).
        clahe_clip_limit: CLAHE clip limit (Default: 9.0).
        clahe_tile_grid_size: CLAHE tile grid size (Default: (16, 16)).
        blur_kernel_size: Gaussian blur kernel size (Default: 7).
        auto_canny_sigma: Sigma for auto Canny (Default: 0.05).
        dilate_iterations: Dilation count (Default: 0).
        use_l2_gradient: Use L2 norm for Canny gradients (Default: False).

    Returns:
        3-channel edge image in BGR format.
    """
    # 1) Convert to Grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2) CLAHE
    # Strong contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit,
        tileGridSize=clahe_tile_grid_size
    )
    gray_eq = clahe.apply(gray)

    # 3) Gaussian Blur
    if blur_kernel_size > 1:
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
        blurred = cv2.GaussianBlur(gray_eq, (blur_kernel_size, blur_kernel_size), 0)
    else:
        blurred = gray_eq

    # 4) Robust Auto-Canny (Masked Median)
    # Calculate median of the retina ONLY (ignore black background)
    # This prevents the median from dropping to 0, ensuring correct thresholds.
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    if np.count_nonzero(mask) > 0:
        v = np.median(blurred[mask > 0])
    else:
        v = np.median(blurred)

    # Compute thresholds based on sigma
    lower = int(max(0, (1.0 - auto_canny_sigma) * v))
    upper = int(min(255, (1.0 + auto_canny_sigma) * v))

    edges = cv2.Canny(blurred, lower, upper, L2gradient=use_l2_gradient)

    # 5) Dilation
    if dilate_iterations > 0:
        kernel_dilate = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel_dilate, iterations=dilate_iterations)

    # 6) Convert to 3-channel
    edges_3ch = cv2.merge((edges, edges, edges))
    return edges_3ch