"""
Canny edge detection with filtering and contour extraction utilities.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple


def _red(img: np.ndarray) -> np.ndarray:
    """Extract red channel or convert to grayscale."""
    if img.ndim == 2:
        return img
    if img.shape[2] >= 3:
        return img[:, :, 0]  # Red channel
    return img[:, :, 0]


def compute_filtered_canny(
    base: np.ndarray,
    blur_sigma: float,
    canny_low: int,
    canny_high: int,
    min_area: int,
) -> Tuple[np.ndarray, int, int]:
    """
    Red channel -> blur -> Canny -> remove white components < min_area.
    Returns (edges_binary, n_white_px, n_removed).
    edges_binary: uint8, 0 or 255 (black background, white edges).
    """
    gray = _red(base)
    blurred = cv2.GaussianBlur(gray, (0, 0), max(0.1, blur_sigma))
    canny = cv2.Canny(blurred, canny_low, canny_high)
    
    # Remove small components
    n_white_before = int(np.sum(canny > 127))
    if min_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(canny, connectivity=8)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                canny[labels == i] = 0
    n_white_after = int(np.sum(canny > 127))
    n_removed = n_white_before - n_white_after
    
    return canny, n_white_after, n_removed


def edges_to_contours(binary_edges: np.ndarray, min_points: int = 3) -> list:
    """
    Contours that follow the edge pixels (outer boundary of each connected component).
    No closing; use edges_to_contours_closed for the recommended pipeline.
    """
    if np.all(binary_edges <= 0):
        return []
    contours, _ = cv2.findContours(
        binary_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if min_points > 0:
        contours = [c for c in contours if len(c) >= min_points]
    return contours


def edges_to_contours_closed(
    binary_edges: np.ndarray,
    close_kernel_size: int = 5,
    min_area: int = 200,
    keep_all_points: bool = True,
) -> list:
    """
    Best-practice pipeline: edges → threshold → close → findContours → filter by area.
    """
    if np.all(binary_edges <= 0):
        return []
    _, binary = cv2.threshold(binary_edges, 127, 255, cv2.THRESH_BINARY)
    if close_kernel_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (close_kernel_size, close_kernel_size)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    mode = cv2.CHAIN_APPROX_NONE if keep_all_points else cv2.CHAIN_APPROX_SIMPLE
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, mode)
    if min_area > 0:
        contours = [c for c in contours if cv2.contourArea(c) > min_area]
    return contours


def edges_to_region_contours(
    binary_edges: np.ndarray,
    min_area: int = 0,
    close_kernel_size: int = 0,
    dilate_before_close: int = 0,
) -> list:
    """
    Make contours that **enclose areas** (e.g. cells) from a binary edge image.
    """
    if np.all(binary_edges >= 255):
        return []
    edges = binary_edges.copy()
    if dilate_before_close > 0:
        k = 2 * dilate_before_close + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        edges = cv2.dilate(edges, kernel)
    if close_kernel_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size)
        )
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    inv = (255 - edges).astype(np.uint8)
    h, w = inv.shape[:2]
    # Flood-fill from border to mark background; remaining white = holes (enclosed regions)
    mask = inv.copy()
    cv2.floodFill(mask, None, (0, 0), 0)
    cv2.floodFill(mask, None, (w - 1, 0), 0)
    cv2.floodFill(mask, None, (0, h - 1), 0)
    cv2.floodFill(mask, None, (w - 1, h - 1), 0)
    for x in [0, w // 2, w - 1]:
        if mask[0, x] == 255:
            cv2.floodFill(mask, None, (x, 0), 0)
        if mask[h - 1, x] == 255:
            cv2.floodFill(mask, None, (x, h - 1), 0)
    for y in [0, h // 2, h - 1]:
        if mask[y, 0] == 255:
            cv2.floodFill(mask, None, (0, y), 0)
        if mask[y, w - 1] == 255:
            cv2.floodFill(mask, None, (w - 1, y), 0)
    holes = (mask == 255).astype(np.uint8) * 255
    if np.all(holes <= 0):
        return []
    contours, _ = cv2.findContours(
        holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if min_area > 0:
        contours = [c for c in contours if cv2.contourArea(c) > min_area]
    return contours
