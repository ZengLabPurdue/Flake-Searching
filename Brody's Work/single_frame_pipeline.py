#!/usr/bin/env python3
"""
Single-frame pipeline for microscope handoff.

Takes one image (file path or numpy array) and runs the full flake extraction pipeline.
Returns ONLY the minimal data needed downstream:
  - contours for each crop (in crop-local coordinates)
  - background mask for each crop
  - optional: full-frame background mask

No file I/O for results — everything is returned in memory for passing to the next stage.

Usage:
    from single_frame_pipeline import process_frame, FrameResult

    result = process_frame("path/to/image.png")
    # or
    result = process_frame(numpy_image_array)

    for crop in result.crops:
        contour = crop.contour        # Nx1x2, crop-local coords
        bg_mask = crop.background_mask  # uint8, 255=background, 0=flake
        bbox = crop.bbox              # (x, y, w, h) in original image
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
from PIL import Image

# Import from existing pipeline components
from batch_robust_contours_and_masks import filter_nested_contours
from extract_background import extract_background
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

MIN_AREA = 800
BBOX_PADDING = 20
HSV_HUE_LO, HSV_HUE_HI = 25, 45
MAJORITY_THRESHOLD = 0.5


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


def _filter_hsv_contours(contours: list, img: np.ndarray) -> list:
    """Remove contours where majority of interior pixels have HSV H in yellow range (25-45)."""
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    h, w = img.shape[:2]
    img_u8 = np.clip(img, 0, 255).astype(np.uint8) if img.dtype != np.uint8 else img
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
    H = hsv[:, :, 0]

    filtered = []
    for c in contours:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        interior_H = H[mask > 127]
        n = len(interior_H)
        if n < 30:
            filtered.append(c)
            continue
        in_range = np.sum((interior_H >= HSV_HUE_LO) & (interior_H <= HSV_HUE_HI))
        if in_range / n >= MAJORITY_THRESHOLD:
            continue
        filtered.append(c)
    return filtered


def _ensure_rgb_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


@dataclass
class CropResult:
    """Result for one flake crop."""

    contour: np.ndarray  # Nx1x2 in crop-local coordinates
    background_mask: np.ndarray  # uint8: 255=background, 0=flake
    bbox: tuple  # (x, y, w, h) in original image


@dataclass
class FrameResult:
    """Result of processing a single microscope frame."""

    crops: List[CropResult] = field(default_factory=list)
    full_frame_background_mask: Optional[np.ndarray] = None  # 255=background, 0=shapes


def process_frame(
    image: Union[str, Path, np.ndarray],
    *,
    apply_hsv_filter: bool = True,
    verbose: bool = False,
) -> FrameResult:
    """
    Process a single microscope frame (one image).

    Args:
        image: File path or numpy array (HxWx3 RGB, uint8)
        apply_hsv_filter: If True, filter out contours with majority yellow interior
        verbose: If True, print progress; otherwise suppress

    Returns:
        FrameResult with:
          - crops: list of CropResult, each with contour, background_mask, bbox
          - full_frame_background_mask: full-frame mask (255=bg, 0=shapes) or None
    """
    # Load image
    if isinstance(image, (str, Path)):
        img = np.array(Image.open(image))
    else:
        img = np.asarray(image)
    img = _ensure_rgb_uint8(img)
    orig_h, orig_w = img.shape[:2]

    # 1. Run ContourPipeline to get contours (no pre-computed overlays for live frames)
    params = _load_params()
    pipeline = ContourPipeline()
    with contextlib.redirect_stdout(io.StringIO()) if not verbose else contextlib.nullcontext():
        result = pipeline.run(img, params, return_edges=False, return_binary=False)

    contours = result["contours"]
    if not contours:
        return FrameResult(crops=[], full_frame_background_mask=result.get("mask_shapes"))

    # Filter by area
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]
    if apply_hsv_filter:
        contours = _filter_hsv_contours(contours, img)

    if not contours:
        return FrameResult(
            crops=[],
            full_frame_background_mask=(255 - result["mask_shapes"]) if "mask_shapes" in result else None,
        )

    # Full-frame background mask: invert shape mask (shapes=255 -> bg=255, shapes=0 -> bg=0)
    mask_shapes = result.get("mask_shapes")
    full_frame_bg_mask = (255 - mask_shapes) if mask_shapes is not None else None

    # 2. For each contour: crop, run background extractor, get contour + mask
    crops_out: List[CropResult] = []
    pad = BBOX_PADDING

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        suppress = contextlib.redirect_stdout(io.StringIO()) if not verbose else contextlib.nullcontext()

        for idx, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            side = max(w, h) + 2 * pad
            cx, cy = x + w // 2, y + h // 2
            half = side // 2
            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(orig_w, x1 + side)
            y2 = min(orig_h, y1 + side)
            x1 = max(0, x2 - side)
            y1 = max(0, y2 - side)

            crop = img[y1:y2, x1:x2].copy()
            crop_path = tmp_path / f"crop_{idx}.png"
            Image.fromarray(crop).save(crop_path)

            # Contour in crop-local coordinates
            rel_contour = contour.copy()
            rel_contour[:, 0, 0] -= x1
            rel_contour[:, 0, 1] -= y1

            # Run background extractor on crop
            try:
                with suppress:
                    _, mask_inv, _ = extract_background(
                        crop_path,
                        output_path=tmp_path / f"bg_{idx}",
                        bin_factor=1,
                        canny_low=5,
                        canny_high=25,
                        min_area=5,
                        bridge_max_gap=30,
                        force_close_max_gap=40,
                    )
                bg_mask = mask_inv  # 255=background, 0=flake
            except Exception:
                # Fallback: use contour fill as flake mask, invert for background
                flake_mask = np.zeros((crop.shape[0], crop.shape[1]), dtype=np.uint8)
                cv2.fillPoly(flake_mask, [rel_contour], 255)
                bg_mask = 255 - flake_mask

            crops_out.append(
                CropResult(
                    contour=rel_contour,
                    background_mask=bg_mask,
                    bbox=(x, y, w, h),
                )
            )

    return FrameResult(crops=crops_out, full_frame_background_mask=full_frame_bg_mask)


def main():
    """CLI: process one image and print summary (no files written)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process a single microscope frame; output contours + masks (no files)."
    )
    parser.add_argument("image", type=Path, help="Path to image file")
    parser.add_argument("--no-filter", action="store_true", help="Disable HSV yellow filter")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print progress")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Error: {args.image} not found")
        return 1

    result = process_frame(
        args.image,
        apply_hsv_filter=not args.no_filter,
        verbose=args.verbose,
    )

    print(f"Processed: {args.image.name}")
    print(f"  Contours/crops: {len(result.crops)}")
    if result.full_frame_background_mask is not None:
        print(f"  Full-frame background mask: {result.full_frame_background_mask.shape}")

    for i, c in enumerate(result.crops):
        print(f"  Crop {i}: bbox={c.bbox}, contour pts={len(c.contour)}, mask shape={c.background_mask.shape}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
