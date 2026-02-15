"""
Edge detection variants for faint edges (slightly different from background).
Each function takes img (H,W) or (H,W,3), returns list of (name, uint8_image).
"""
from typing import List, Tuple

import cv2
import numpy as np

SCHARR_KSIZE = -1
SOBEL_KSIZE = 3


def _sobel_scharr_mag(ch: np.ndarray, ksize: int) -> np.ndarray:
    if ch.dtype != np.float64:
        ch = ch.astype(np.float64)
    gx = cv2.Sobel(ch, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(ch, cv2.CV_64F, 0, 1, ksize=ksize)
    return np.sqrt(gx * gx + gy * gy)


def _to_uint8_linear(mag: np.ndarray) -> np.ndarray:
    if mag.ndim == 2:
        lo, hi = mag.min(), mag.max()
        if hi > lo:
            return np.clip((mag - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        return np.zeros_like(mag, dtype=np.uint8)
    out = np.zeros(mag.shape, dtype=np.uint8)
    for c in range(mag.shape[-1]):
        ch = mag[:, :, c]
        lo, hi = ch.min(), ch.max()
        if hi > lo:
            out[:, :, c] = np.clip((ch - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        else:
            out[:, :, c] = 0
    return out


def _ensure_rgb(gray: np.ndarray) -> np.ndarray:
    if gray.ndim == 2:
        return np.stack([gray, gray, gray], axis=-1)
    return gray


def _red(img: np.ndarray) -> np.ndarray:
    """Use red channel instead of grayscale (R-only pipeline)."""
    if img.ndim == 3:
        return np.ascontiguousarray(img[:, :, 0])
    return img


# --- 1. Grayscale combined (Sobel/Scharr on red channel) ---
def grayscale_combined(img: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    gray = _red(img)
    sobel_mag = _sobel_scharr_mag(gray, SOBEL_KSIZE)
    scharr_mag = _sobel_scharr_mag(gray, SCHARR_KSIZE)
    sobel_u8 = _ensure_rgb(_to_uint8_linear(sobel_mag))
    scharr_u8 = _ensure_rgb(_to_uint8_linear(scharr_mag))
    return [("sobel", sobel_u8), ("scharr", scharr_u8)]


# --- 2. CLAHE then Sobel/Scharr ---
def clahe_then_edges(img: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    gray = _red(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    sobel_mag = _sobel_scharr_mag(enhanced, SOBEL_KSIZE)
    scharr_mag = _sobel_scharr_mag(enhanced, SCHARR_KSIZE)
    sobel_u8 = _ensure_rgb(_to_uint8_linear(sobel_mag))
    scharr_u8 = _ensure_rgb(_to_uint8_linear(scharr_mag))
    return [("sobel", sobel_u8), ("scharr", scharr_u8)]


# --- 3. Unsharp mask then Sobel/Scharr ---
def unsharp_then_edges(img: np.ndarray, sigma: float = 1.0, strength: float = 1.5) -> List[Tuple[str, np.ndarray]]:
    gray = _red(img).astype(np.float64)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    sharp = np.clip(gray + strength * (gray - blurred), 0, 255).astype(np.uint8)
    sobel_mag = _sobel_scharr_mag(sharp, SOBEL_KSIZE)
    scharr_mag = _sobel_scharr_mag(sharp, SCHARR_KSIZE)
    sobel_u8 = _ensure_rgb(_to_uint8_linear(sobel_mag))
    scharr_u8 = _ensure_rgb(_to_uint8_linear(scharr_mag))
    return [("sobel", sobel_u8), ("scharr", scharr_u8)]


# --- 4. Background subtract (image - gaussian blur) then edges ---
def background_subtract_then_edges(img: np.ndarray, sigma: int = 31) -> List[Tuple[str, np.ndarray]]:
    gray = _red(img).astype(np.float64)
    blurred = cv2.GaussianBlur(gray, (sigma | 1, sigma | 1), 0)
    diff = np.clip(gray - blurred + 128, 0, 255).astype(np.uint8)
    sobel_mag = _sobel_scharr_mag(diff, SOBEL_KSIZE)
    scharr_mag = _sobel_scharr_mag(diff, SCHARR_KSIZE)
    sobel_u8 = _ensure_rgb(_to_uint8_linear(sobel_mag))
    scharr_u8 = _ensure_rgb(_to_uint8_linear(scharr_mag))
    return [("sobel", sobel_u8), ("scharr", scharr_u8)]


# --- 5. Percentile rescale (post-process edge map so 95th pct -> 255) ---
def percentile_rescale_edges(img: np.ndarray, pct: float = 95.0) -> List[Tuple[str, np.ndarray]]:
    gray = _red(img)
    sobel_mag = _sobel_scharr_mag(gray, SOBEL_KSIZE)
    scharr_mag = _sobel_scharr_mag(gray, SCHARR_KSIZE)

    def rescale(mag: np.ndarray) -> np.ndarray:
        hi = np.percentile(mag, pct)
        lo = mag.min()
        if hi <= lo:
            return np.zeros_like(mag, dtype=np.uint8)
        out = np.clip((mag - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        return _ensure_rgb(out)

    sobel_u8 = rescale(sobel_mag)
    scharr_u8 = rescale(scharr_mag)
    return [("sobel", sobel_u8), ("scharr", scharr_u8)]


# --- 6. Gamma rescale (power-law to brighten mid-tones) ---
def gamma_rescale_edges(img: np.ndarray, gamma: float = 0.5) -> List[Tuple[str, np.ndarray]]:
    gray = _red(img)
    sobel_mag = _sobel_scharr_mag(gray, SOBEL_KSIZE)
    scharr_mag = _sobel_scharr_mag(gray, SCHARR_KSIZE)

    def rescale(mag: np.ndarray) -> np.ndarray:
        hi = mag.max()
        if hi <= 0:
            return _ensure_rgb(np.zeros_like(mag, dtype=np.uint8))
        norm = mag / hi
        out = np.power(norm, gamma)
        out = np.clip(out * 255, 0, 255).astype(np.uint8)
        return _ensure_rgb(out)

    sobel_u8 = rescale(sobel_mag)
    scharr_u8 = rescale(scharr_mag)
    return [("sobel", sobel_u8), ("scharr", scharr_u8)]


# --- 7. Canny ---
def canny_edges(img: np.ndarray, low: int = 30, high: int = 100) -> List[Tuple[str, np.ndarray]]:
    gray = _red(img)
    canny = cv2.Canny(gray, low, high)
    return [("canny", _ensure_rgb(canny))]


def canny_low_threshold(img: np.ndarray, low: int = 15, high: int = 50) -> List[Tuple[str, np.ndarray]]:
    """Canny with lower thresholds to pick up more (fainter) edges."""
    gray = _red(img)
    canny = cv2.Canny(gray, low, high)
    return [("canny", _ensure_rgb(canny))]


def canny_blur_then(
    img: np.ndarray, sigma: float = 1.0, low: int = 15, high: int = 50
) -> List[Tuple[str, np.ndarray]]:
    """Light Gaussian blur then Canny with lower thresholds: less noise, more faint edges."""
    gray = _red(img)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    canny = cv2.Canny(blurred, low, high)
    return [("canny", _ensure_rgb(canny))]


# --- 8. Morphological gradient (dilate - erode) ---
def morph_gradient_edges(img: np.ndarray, ksize: int = 3) -> List[Tuple[str, np.ndarray]]:
    gray = _red(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(gray, kernel)
    eroded = cv2.erode(gray, kernel)
    grad = np.clip(dilated.astype(np.int16) - eroded.astype(np.int16), 0, 255).astype(np.uint8)
    return [("morph_gradient", _ensure_rgb(grad))]


# --- 9. Laplacian / LoG ---
def laplacian_edges(img: np.ndarray, ksize: int = 3) -> List[Tuple[str, np.ndarray]]:
    gray = _red(img)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    lap = np.abs(lap)
    out = _to_uint8_linear(lap)
    return [("laplacian", _ensure_rgb(out))]


def log_edges(img: np.ndarray, sigma: float = 1.0, ksize: int = 5) -> List[Tuple[str, np.ndarray]]:
    gray = _red(img).astype(np.float64)
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    lap = np.abs(lap)
    out = _to_uint8_linear(lap)
    return [("log", _ensure_rgb(out))]


# --- 10. LAB L* channel edges (still use L* from RGB->LAB of red-only image) ---
def lab_edges(img: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    # Build RGB from red channel only so LAB L* is red-based
    r = _red(img)
    if r.ndim == 2:
        img_rgb = np.stack([r, r, r], axis=-1)
    else:
        img_rgb = img
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_ch = lab[:, :, 0]
    sobel_mag = _sobel_scharr_mag(l_ch, SOBEL_KSIZE)
    scharr_mag = _sobel_scharr_mag(l_ch, SCHARR_KSIZE)
    sobel_u8 = _ensure_rgb(_to_uint8_linear(sobel_mag))
    scharr_u8 = _ensure_rgb(_to_uint8_linear(scharr_mag))
    return [("sobel", sobel_u8), ("scharr", scharr_u8)]


# --- 11. Bilateral (denoise) then Sobel/Scharr ---
def bilateral_then_edges(img: np.ndarray, d: int = 5, sigma_color: int = 50, sigma_space: int = 50) -> List[Tuple[str, np.ndarray]]:
    gray = _red(img)
    denoised = cv2.bilateralFilter(gray, d, sigma_color, sigma_space)
    sobel_mag = _sobel_scharr_mag(denoised, SOBEL_KSIZE)
    scharr_mag = _sobel_scharr_mag(denoised, SCHARR_KSIZE)
    sobel_u8 = _ensure_rgb(_to_uint8_linear(sobel_mag))
    scharr_u8 = _ensure_rgb(_to_uint8_linear(scharr_mag))
    return [("sobel", sobel_u8), ("scharr", scharr_u8)]


# --- 12. Combined magnitude (max over R,G,B edge magnitudes) ---
def combined_magnitude_edges(img: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """Red channel only (R-only pipeline: same as grayscale_combined on R)."""
    gray = _red(img)
    sobel_mag = _sobel_scharr_mag(gray, SOBEL_KSIZE)
    scharr_mag = _sobel_scharr_mag(gray, SCHARR_KSIZE)
    sobel_u8 = _ensure_rgb(_to_uint8_linear(sobel_mag))
    scharr_u8 = _ensure_rgb(_to_uint8_linear(scharr_mag))
    return [("sobel", sobel_u8), ("scharr", scharr_u8)]


# --- 13. Light Gaussian blur then edges (reduce noise, then rescale) ---
def blur_then_edges(img: np.ndarray, sigma: float = 0.8) -> List[Tuple[str, np.ndarray]]:
    gray = _red(img)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    sobel_mag = _sobel_scharr_mag(blurred, SOBEL_KSIZE)
    scharr_mag = _sobel_scharr_mag(blurred, SCHARR_KSIZE)
    sobel_u8 = _ensure_rgb(_to_uint8_linear(sobel_mag))
    scharr_u8 = _ensure_rgb(_to_uint8_linear(scharr_mag))
    return [("sobel", sobel_u8), ("scharr", scharr_u8)]


# Registry: folder_name -> (description, function)
ALL_VARIANTS = [
    ("01_grayscale_combined", "Sobel/Scharr on luminance", grayscale_combined),
    ("02_clahe", "CLAHE then Sobel/Scharr", clahe_then_edges),
    ("03_unsharp", "Unsharp mask then Sobel/Scharr", unsharp_then_edges),
    ("04_background_subtract", "Image minus Gaussian blur then edges", background_subtract_then_edges),
    ("05_percentile_rescale", "Edge map rescaled by 95th percentile", percentile_rescale_edges),
    ("06_gamma_rescale", "Edge map gamma (0.5) to brighten mid-tones", gamma_rescale_edges),
    ("07_canny", "Canny edge detector", canny_edges),
    ("07b_canny_low_threshold", "Canny with lower thresholds (more edges)", canny_low_threshold),
    ("07c_canny_blur_then", "Blur then Canny with lower thresholds", canny_blur_then),
    ("08_morph_gradient", "Morphological gradient (dilate - erode)", morph_gradient_edges),
    ("09_laplacian", "Laplacian edge detector", laplacian_edges),
    ("10_log", "Laplacian of Gaussian", log_edges),
    ("11_lab", "Sobel/Scharr on LAB L* channel", lab_edges),
    ("12_bilateral_then_edges", "Bilateral denoise then Sobel/Scharr", bilateral_then_edges),
    ("13_combined_magnitude", "Max of R,G,B edge magnitudes", combined_magnitude_edges),
    ("14_blur_then_edges", "Light Gaussian blur then Sobel/Scharr", blur_then_edges),
]
