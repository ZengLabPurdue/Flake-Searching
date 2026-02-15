#!/usr/bin/env python3
"""
Single-frame pipeline for microscope handoff.

Takes one image (file path or numpy array) and returns exactly two things:
  1. Full-frame mask overlay (from batch_robust_contours_and_masks) — original image with
     shapes masked (background visible, shapes blacked out)
  2. Contours (from batch_filtered_sensitive_overlays_2x2) — raw contour list, no overlay

Usage:
    from single_frame_pipeline import process_frame

    mask_overlay, contours = process_frame("path/to/image.png")
    # mask_overlay: HxWx3 RGB image (original with shapes blacked out)
    # contours: list of np.ndarray (Nx1x2), OpenCV contour format
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from PIL import Image

from batch_robust_contours_and_masks import process_image_with_masks
from pipeline import ContourPipeline

# Params for contour detection (from batch_filtered_sensitive_overlays_2x2)
SETTINGS_PATH = Path(__file__).resolve().parent / "batch_filtered_settings.json"
_DEFAULT_PARAMS = {
    "blur_sigma": 0.6,
    "canny_low": 10,
    "canny_high": 50,
    "min_area": 0,
    "edge_method": "canny_h",
    "edge_methods": ["canny_h"],
    "use_h_channel": True,
    "preprocessing": "full",
    "bridge_gap_factor": 0.15,
    "force_close_factor": 0.3,
    "line_thickness": 4,
    "close_kernel_divisor": 100,
    "morph_margin": 15,
    "corner_radius": 2,
    "min_area_contour": 200,
}


def _load_params() -> dict:
    """Load pipeline params from batch_filtered_settings.json if exists."""
    if not SETTINGS_PATH.exists():
        return _DEFAULT_PARAMS.copy()
    try:
        with open(SETTINGS_PATH) as f:
            loaded = json.load(f)
        out = _DEFAULT_PARAMS.copy()
        for k, v in loaded.items():
            if k in out:
                out[k] = v
        return out
    except Exception:
        return _DEFAULT_PARAMS.copy()


def _ensure_rgb_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def process_frame(
    image: Union[str, Path, np.ndarray],
    *,
    verbose: bool = False,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Process a single microscope frame (one image).

    Args:
        image: File path or numpy array (HxWx3 RGB, uint8)
        verbose: If True, print progress; otherwise suppress

    Returns:
        Tuple of:
          - mask_overlay: HxWx3 image — original with shapes masked (background visible,
            shapes blacked out). From batch_robust_contours_and_masks logic.
          - contours: list of contours (each Nx1x2, OpenCV format). From
            batch_filtered_sensitive_overlays_2x2 (ContourPipeline). Raw contours, not drawn.
    """
    # Load image
    if isinstance(image, (str, Path)):
        img = np.array(Image.open(image))
    else:
        img = np.asarray(image)
    img = _ensure_rgb_uint8(img)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_path = tmp_path / "frame.png"
        Image.fromarray(img).save(img_path)
        output_dir = tmp_path / "robust_out"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Run batch_robust_contours_and_masks to get mask overlay
        suppress = contextlib.redirect_stdout(io.StringIO()) if not verbose else contextlib.nullcontext()
        with suppress:
            ok = process_image_with_masks(img_path, output_dir)
        if not ok:
            # Fallback: return original image and empty contours if robust fails
            mask_overlay = img.copy()
        else:
            masked_path = output_dir / "frame_masked_background_only.png"
            if masked_path.exists():
                mask_overlay = np.array(Image.open(masked_path))
                if mask_overlay.ndim == 2:
                    mask_overlay = np.stack([mask_overlay, mask_overlay, mask_overlay], axis=-1)
                if mask_overlay.shape[2] == 4:
                    mask_overlay = mask_overlay[:, :, :3]
            else:
                mask_overlay = img.copy()

        # 2. Run batch_filtered_sensitive (ContourPipeline) to get contours only
        params = _load_params()
        pipeline = ContourPipeline()
        with suppress if not verbose else contextlib.nullcontext():
            result = pipeline.run(img, params, return_edges=False, return_binary=False)
        contours = result.get("contours", [])
        contours = list(contours)  # ensure list of np arrays

    return mask_overlay, contours


def main():
    """CLI: process one image and print summary."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process a single microscope frame; returns mask overlay + contours."
    )
    parser.add_argument("image", type=Path, help="Path to image file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print progress")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Error: {args.image} not found")
        return 1

    mask_overlay, contours = process_frame(args.image, verbose=args.verbose)

    print(f"Processed: {args.image.name}")
    print(f"  Mask overlay shape: {mask_overlay.shape}")
    print(f"  Contours: {len(contours)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
