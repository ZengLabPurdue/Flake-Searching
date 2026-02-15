"""
Preprocessing to make slightly darker regions easier to discern from background.
Each function takes img (H,W) or (H,W,3), returns a single uint8 image (grayscale or RGB).
"""
import numpy as np
import cv2


def _to_uint8(x: np.ndarray, per_channel: bool = True) -> np.ndarray:
    """Normalize to 0-255 uint8."""
    if x.ndim == 2:
        lo, hi = x.min(), x.max()
        if hi > lo:
            return np.clip((x - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        return np.zeros_like(x, dtype=np.uint8)
    out = np.zeros(x.shape, dtype=np.uint8)
    for c in range(x.shape[-1]):
        ch = x[:, :, c]
        lo, hi = ch.min(), ch.max()
        if hi > lo:
            out[:, :, c] = np.clip((ch - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        else:
            out[:, :, c] = 0
    return out


def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def _red(img: np.ndarray) -> np.ndarray:
    """Use red channel instead of grayscale for full pipeline (R-only)."""
    if img.ndim == 3:
        return np.ascontiguousarray(img[:, :, 0])
    return img


# --- 1. CLAHE (contrast-limited adaptive histogram equalization) ---
def clahe(img: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    """Boost local contrast so slightly darker regions stand out (uses red channel)."""
    gray = _red(img)
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    out = clahe_obj.apply(gray)
    return np.stack([out, out, out], axis=-1)


# --- CLAHE + denoise (reduce noise while keeping contrast) ---
def clahe_then_bilateral(
    img: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: int = 8,
    d: int = 5,
    sigma_color: int = 50,
    sigma_space: int = 50,
) -> np.ndarray:
    """CLAHE then bilateral filter: edge-preserving denoise (uses red channel)."""
    gray = _red(img)
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    enhanced = clahe_obj.apply(gray)
    out = cv2.bilateralFilter(enhanced, d, sigma_color, sigma_space)
    return np.stack([out, out, out], axis=-1)


def clahe_then_gaussian(
    img: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: int = 8,
    sigma: float = 0.8,
) -> np.ndarray:
    """CLAHE then light Gaussian blur (uses red channel)."""
    gray = _red(img)
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    enhanced = clahe_obj.apply(gray)
    out = cv2.GaussianBlur(enhanced, (0, 0), sigma)
    return np.stack([out, out, out], axis=-1)


def clahe_then_nlmeans(
    img: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: int = 8,
    h: float = 10.0,
    template_window: int = 7,
    search_window: int = 21,
) -> np.ndarray:
    """CLAHE then non-local means denoising (uses red channel)."""
    gray = _red(img)
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    enhanced = clahe_obj.apply(gray)
    out = cv2.fastNlMeansDenoising(enhanced, None, h, template_window, search_window)
    return np.stack([out, out, out], axis=-1)


def clahe_gentle(img: np.ndarray, clip_limit: float = 1.2, grid_size: int = 8) -> np.ndarray:
    """Gentler CLAHE, uses red channel."""
    gray = _red(img)
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    out = clahe_obj.apply(gray)
    return np.stack([out, out, out], axis=-1)


# --- 2. Background flatten (divide by smoothed image) ---
def background_flatten(img: np.ndarray, sigma: int = 51) -> np.ndarray:
    """Remove uneven illumination (uses red channel)."""
    gray = _red(img).astype(np.float64) + 1.0
    blurred = cv2.GaussianBlur(gray, (sigma | 1, sigma | 1), 0) + 1.0
    flat = gray / blurred
    out = _to_uint8(flat)
    return np.stack([out, out, out], axis=-1)


# --- 3. Local normalization (z-score in a window) ---
def local_normalize(img: np.ndarray, radius: int = 15, eps: float = 1e-6) -> np.ndarray:
    """Local z-score (uses red channel)."""
    gray = _red(img).astype(np.float64)
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.float64) / ((2 * radius + 1) ** 2)
    local_mean = cv2.filter2D(gray, -1, kernel)
    local_sq = cv2.filter2D(gray * gray, -1, kernel)
    local_var = np.maximum(local_sq - local_mean * local_mean, 0)
    local_std = np.sqrt(local_var) + eps
    z = (gray - local_mean) / local_std
    # Map roughly [-2, 2] to [0, 255]
    z = np.clip(z * 64 + 128, 0, 255).astype(np.uint8)
    return np.stack([z, z, z], axis=-1)


# --- 4. Black top-hat (closing - image): isolates dark structures ---
def black_tophat(img: np.ndarray, radius: int = 11) -> np.ndarray:
    """Black top-hat (uses red channel)."""
    gray = _red(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    dark = np.clip(closed.astype(np.int16) - gray.astype(np.int16), 0, 255).astype(np.uint8)
    return np.stack([dark, dark, dark], axis=-1)


# --- 5. Percentile stretch (2nd–98th percentile -> 0–255) ---
def percentile_stretch(img: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
    """Percentile stretch (uses red channel)."""
    gray = _red(img)
    lo = np.percentile(gray, low_pct)
    hi = np.percentile(gray, high_pct)
    if hi <= lo:
        out = np.zeros_like(gray, dtype=np.uint8)
    else:
        out = np.clip((gray.astype(np.float64) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return np.stack([out, out, out], axis=-1)


# --- 6. Gamma < 1 (brighten mid-tones) ---
def gamma_brighten(img: np.ndarray, gamma: float = 0.7) -> np.ndarray:
    """Gamma brighten (uses red channel)."""
    gray = _red(img).astype(np.float64) / 255.0
    out = np.power(gray, gamma)
    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    return np.stack([out, out, out], axis=-1)


# --- 7. Subtract opening (image - opening): dark-on-light local contrast ---
def dark_on_light(img: np.ndarray, radius: int = 5) -> np.ndarray:
    """Image minus opening (uses red channel)."""
    gray = _red(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    diff = np.clip(gray.astype(np.int16) - opened.astype(np.int16), 0, 255).astype(np.uint8)
    return np.stack([diff, diff, diff], axis=-1)


# --- 8. Combined: background flatten then percentile stretch ---
def flatten_then_stretch(img: np.ndarray, sigma: int = 51, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    """Flatten then stretch (uses red channel)."""
    gray = _red(img).astype(np.float64) + 1.0
    blurred = cv2.GaussianBlur(gray, (sigma | 1, sigma | 1), 0) + 1.0
    flat = gray / blurred
    lo = np.percentile(flat, low_pct)
    hi = np.percentile(flat, high_pct)
    if hi <= lo:
        out = np.zeros_like(flat, dtype=np.uint8)
    else:
        out = np.clip((flat - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return np.stack([out, out, out], axis=-1)


# Registry: folder_name -> (description, function)
ALL_PREPROCESSORS = [
    ("01_clahe", "CLAHE local contrast", clahe),
    ("02_background_flatten", "Divide by Gaussian blur (flatten illumination)", background_flatten),
    ("03_local_normalize", "Local z-score (local_mean, local_std)", local_normalize),
    ("04_black_tophat", "Black top-hat: isolate slightly darker regions", black_tophat),
    ("05_percentile_stretch", "Rescale 2nd–98th percentile to 0–255", percentile_stretch),
    ("06_gamma_brighten", "Gamma 0.7 to brighten mid-tones", gamma_brighten),
    ("07_dark_on_light", "Image minus opening (dark vs local neighborhood)", dark_on_light),
    ("08_flatten_then_stretch", "Background flatten + percentile stretch", flatten_then_stretch),
    # CLAHE + denoise (less noisy than plain CLAHE)
    ("09_clahe_then_bilateral", "CLAHE then bilateral (edge-preserving denoise)", clahe_then_bilateral),
    ("10_clahe_then_gaussian", "CLAHE then light Gaussian blur", clahe_then_gaussian),
    ("11_clahe_then_nlmeans", "CLAHE then non-local means denoise", clahe_then_nlmeans),
    ("12_clahe_gentle", "Gentler CLAHE (lower clip = less noise)", clahe_gentle),
]
