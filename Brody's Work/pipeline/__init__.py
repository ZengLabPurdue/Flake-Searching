"""
Modular contour detection pipeline with interchangeable components.
Use classes and registry pattern for swappable preprocessors, edge detectors, etc.
"""
from pipeline import components  # noqa: F401 - registers components
from pipeline.core import (
    ContourPipeline,
    Preprocessor,
    EdgeDetector,
    GapCloser,
    Binner,
    PREPROCESSOR_REGISTRY,
    EDGE_DETECTOR_REGISTRY,
)

__all__ = [
    "ContourPipeline",
    "Preprocessor",
    "EdgeDetector",
    "GapCloser",
    "Binner",
    "PREPROCESSOR_REGISTRY",
    "EDGE_DETECTOR_REGISTRY",
]
