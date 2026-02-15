# Pipeline Module

Modular, interchangeable contour detection pipeline. Components (preprocessors, edge detectors, binners) are registered by name and looked up from params.

## Usage

```python
from pipeline import ContourPipeline

pipeline = ContourPipeline()
params = {"preprocessing": "full", "edge_method": "canny_h", "binning": True, ...}
result = pipeline.run(img, params)
# result: contours, overlay, edges, mask_shapes, masked_background, n_contours, n_edges_px
```

## Adding New Components

### New Preprocessor

In `pipeline/components.py` or your own module (imported before use):

```python
from pipeline.core import register_preprocessor

@register_preprocessor("my_preproc")
def my_preproc(img: np.ndarray, params: dict) -> np.ndarray:
    # Your logic
    return processed_img
```

Set `params["preprocessing"] = "my_preproc"` to use it.

### New Edge Detector

```python
from pipeline.core import register_edge_detector

@register_edge_detector("my_edges")
def my_edges(work: np.ndarray, preprocessed: np.ndarray, params: dict) -> np.ndarray:
    # work = binned image, preprocessed = after CLAHE/denoise
    return edges_uint8  # 0-255
```

Set `params["edge_method"] = "my_edges"` or add to `params["edge_methods"]` for combination.

### Multiple Edge Methods

Use `params["edge_methods"] = ["canny_h", "sobel", "laplacian"]` to combine edges with pixel-wise max (OR).

## Architecture

- **ContourPipeline**: Orchestrates bin → preprocess → edges → gap close → contours
- **Registries**: `PREPROCESSOR_REGISTRY`, `EDGE_DETECTOR_REGISTRY`, `BINNER_REGISTRY`
- **Params-driven**: Component choice comes from `params["preprocessing"]`, `params["edge_method"]` / `params["edge_methods"]`
