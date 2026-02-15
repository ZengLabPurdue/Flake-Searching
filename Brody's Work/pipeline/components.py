"""
Register implementations for pipeline components.
Wraps existing preprocess, edge detection, etc. into the registry.
Import this module to populate PREPROCESSOR_REGISTRY, EDGE_DETECTOR_REGISTRY, etc.

To add a new preprocessor:
    @register_preprocessor("my_preproc")
    def my_preproc(img, params):
        return processed_img

To add a new edge detector:
    @register_edge_detector("my_edges")
    def my_edges(work, preprocessed, params):
        return edges_uint8
"""
import numpy as np
from pipeline.core import (
    register_preprocessor,
    register_edge_detector,
    PREPROCESSOR_REGISTRY,
    EDGE_DETECTOR_REGISTRY,
    BINNER_REGISTRY,
)


# --- Preprocessors ---

def _preprocess_none(img: np.ndarray, params: dict) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8) if img.dtype != np.uint8 else img


def _preprocess_clahe(img: np.ndarray, params: dict) -> np.ndarray:
    from preprocess import clahe
    return clahe(img)


def _preprocess_full(img: np.ndarray, params: dict) -> np.ndarray:
    from preprocess import clahe_then_nlmeans
    return clahe_then_nlmeans(img)


register_preprocessor("none")(_preprocess_none)
register_preprocessor("clahe")(_preprocess_clahe)
register_preprocessor("full")(_preprocess_full)


# --- Edge detector (combined: single or multiple methods) ---

def _edge_detector_combined(work: np.ndarray, preprocessed: np.ndarray, params: dict) -> np.ndarray:
    """Uses contour_tuner_ui's _compute_edges_combined (handles edge_methods list)."""
    from contour_tuner_ui import _compute_edges_combined
    return _compute_edges_combined(work, preprocessed, params)


register_edge_detector("combined")(_edge_detector_combined)


# --- Binners ---

def _binner_2x2(img: np.ndarray, factor: int) -> np.ndarray:
    from batch_robust_contours_and_masks import bin_image_by_2
    return bin_image_by_2(img)


def _binner_none(img: np.ndarray, factor: int) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8) if img.dtype != np.uint8 else img.copy()


BINNER_REGISTRY["2x2"] = _binner_2x2
BINNER_REGISTRY["none"] = _binner_none


# Ensure components are registered when module is imported
def _ensure_registered():
    pass


_ensure_registered()
