#!/usr/bin/env python3
"""
Single-frame pipeline for microscope handoff.

Takes one image (file path or numpy array) and returns exactly two things:
  1. Full-frame mask overlay (from batch_robust_contours_and_masks) — original image with
     shapes masked (background visible, shapes blacked out). Uses robust_contours_settings.json.
  2. Contours (from batch_filtered_sensitive_overlays_2x2) — raw contour list from edge
     detection and contour drawing. Uses batch_filtered_settings.json for params. np.ndarray with dtype=object, shape (n_contours,), where each element is a contour array with shape (N, 1, 2).

Usage:
    from single_frame_pipeline import process_frame

    mask_overlay, contours = process_frame("path/to/image.png")
    # mask_overlay: HxWx3 RGB image (from batch_robust)
    # contours: np.ndarray (dtype=object) of contour arrays, each Nx1x2 (OpenCV format)
"""
from __future__ import annotations

import contextlib
import io
import tempfile
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from PIL import Image

from batch_filtered_sensitive_overlays_2x2 import load_filtered_overlay_params
from batch_robust_contours_and_masks import load_contour_params, process_image_with_masks
from pipeline import ContourPipeline


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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a single microscope frame (one image).

    Args:
        image: File path or numpy array (HxWx3 RGB, uint8)
        verbose: If True, print progress; otherwise suppress

    Returns:
        Tuple of:
          - mask_overlay: HxWx3 image from batch_robust_contours_and_masks (shapes blacked out).
            Uses robust_contours_settings.json (same as full pipeline masked_background_only).
          - contours: np.ndarray (dtype=object) of contour arrays (each Nx1x2) from
            batch_filtered_sensitive_overlays_2x2 (ContourPipeline). Raw contours, not drawn.
    """
    # Load image
    if isinstance(image, (str, Path)):
        img = np.array(Image.open(image))
    else:
        img = np.asarray(image)
    img = _ensure_rgb_uint8(img)

    suppress = contextlib.redirect_stdout(io.StringIO()) if not verbose else contextlib.nullcontext()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_path = tmp_path / "frame.png"
        Image.fromarray(img).save(img_path)
        output_dir = tmp_path / "robust_out"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Mask overlay from batch_robust_contours_and_masks (background masking)
        # Uses robust_contours_settings.json + no_gap_close to match robust_contours_and_masks_no_gap_close/
        robust_params = load_contour_params()
        with suppress:
            ok = process_image_with_masks(
                img_path, output_dir, params=robust_params, no_gap_close=True
            )
        if not ok:
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

        # 2. Contours from batch_filtered_sensitive_overlays_2x2 (edge detection + contour drawing)
        # Uses batch_filtered_settings.json via load_filtered_overlay_params
        params = load_filtered_overlay_params()
        pipeline = ContourPipeline()
        with suppress if not verbose else contextlib.nullcontext():
            result = pipeline.run(img, params, return_edges=False, return_binary=False)
        contours_list = result.get("contours", [])
        contours = np.array(contours_list, dtype=object) if contours_list else np.array([], dtype=object)

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
