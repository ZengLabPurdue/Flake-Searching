#!/usr/bin/env python3
"""
Batch process all images with the filtered-sensitive overlay pipeline using 2x2 binning.
Uses settings from batch_filtered_settings.json (batch_filtered_tuner_ui.py).

Input: images/x20_images/ (original, unbinned images)
Output: images/filtered_sensitive_overlays/ (compatible with flake_extraction_pipeline)
  - {stem}_binned_filtered_overlay.png
  - {stem}_binned_filtered_contours.png

Usage:
    python batch_filtered_sensitive_overlays_2x2.py [input_dir] [-o output_dir]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from contour_tuner_ui import _compute_edges_by_method
from batch_robust_contours_and_masks import (
    bin_image_by_2,
    morph_extend_to_border,
    filter_nested_contours,
)
from edge_to_contour_methods import (
    _ensure_binary,
    force_close_open_chains,
    smart_connect_endpoints_bridged,
    _find_endpoints,
)
from preprocess import clahe, clahe_then_nlmeans

BATCH_FILTERED_SETTINGS_PATH = Path(__file__).resolve().parent / "batch_filtered_settings.json"
_DEFAULT_PARAMS = {
    "blur_sigma": 0.6,
    "canny_low": 10,
    "canny_high": 50,
    "min_area": 0,
    "edge_method": "canny_h",
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
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_filtered_overlay_params() -> dict:
    """Load params from batch_filtered_settings.json if it exists."""
    if not BATCH_FILTERED_SETTINGS_PATH.exists():
        return _DEFAULT_PARAMS.copy()
    try:
        with open(BATCH_FILTERED_SETTINGS_PATH) as f:
            loaded = json.load(f)
        out = _DEFAULT_PARAMS.copy()
        for k, v in loaded.items():
            if k in out:
                out[k] = v
        return out
    except Exception:
        return _DEFAULT_PARAMS.copy()


def save_filtered_overlay_params(params: dict) -> None:
    """Save params to batch_filtered_settings.json."""
    with open(BATCH_FILTERED_SETTINGS_PATH, "w") as f:
        json.dump(params, f, indent=2)


def process_one(input_path: Path, output_dir: Path, params: dict) -> bool:
    """Process a single image and save overlay + contours to output_dir."""
    if not input_path.exists():
        print(f"  Skip (not found): {input_path.name}")
        return False

    try:
        original_img = np.array(Image.open(input_path))
        if original_img.ndim == 2:
            original_img = np.stack([original_img, original_img, original_img], axis=-1)
        if original_img.shape[2] == 4:
            original_img = original_img[:, :, :3]
        orig_h, orig_w = original_img.shape[:2]

        # 2x2 binning
        work = bin_image_by_2(original_img)
        bin_h, bin_w = work.shape[:2]
        if work.dtype != np.uint8:
            work = np.clip(work, 0, 255).astype(np.uint8)
        scale_x = orig_w / bin_w
        scale_y = orig_h / bin_h

        # Preprocess
        preproc = params.get("preprocessing", "full")
        if preproc == "none":
            preprocessed = work
        elif preproc == "clahe":
            preprocessed = clahe(work)
        else:
            preprocessed = clahe_then_nlmeans(work)

        # Edge detection (uses tuner settings, supports all edge methods)
        edges = _compute_edges_by_method(work, preprocessed, params)

        # Gap closing (uses tuner settings)
        binary = _ensure_binary(edges).copy()
        diag = int(np.hypot(bin_h, bin_w))
        bridge_f = params.get("bridge_gap_factor", 0.15)
        force_f = params.get("force_close_factor", 0.3)
        adaptive_bridge = max(50, int(diag * bridge_f))
        adaptive_force = max(100, int(diag * force_f))
        line_thick = params.get("line_thickness", 4)
        close_div = params.get("close_kernel_divisor", 100)
        ksize = max(3, int(min(bin_h, bin_w) / close_div))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = smart_connect_endpoints_bridged(
            binary, max_gap=adaptive_bridge, line_thickness=line_thick,
            direction_weight=2.0, extend_pixels=2
        )
        binary = force_close_open_chains(
            binary, line_thickness=line_thick, max_iterations=200, max_gap=adaptive_force
        )
        remaining = _find_endpoints(binary)
        if remaining:
            h, w = binary.shape
            rad = min(adaptive_force * 2, max(h, w))
            for y1, x1 in remaining:
                best_d, best_p = float("inf"), None
                for py in range(max(0, y1 - rad), min(h, y1 + int(rad) + 1)):
                    for px in range(max(0, x1 - rad), min(w, x1 + int(rad) + 1)):
                        if binary[py, px] < 127 or (py == y1 and px == x1):
                            continue
                        d = np.hypot(py - y1, px - x1)
                        if 0.5 < d < best_d:
                            best_d, best_p = d, (py, px)
                if best_p:
                    cv2.line(binary, (x1, y1), (best_p[1], best_p[0]), 255, line_thick)

        morph_margin = params.get("morph_margin", 15)
        corner_rad = params.get("corner_radius", 2)
        binary = morph_extend_to_border(binary, bin_h, bin_w, margin=morph_margin, corner_radius=corner_rad)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        all_c = []
        for c in contours:
            if len(c) > 0:
                first, last = c[0][0], c[-1][0]
                if not np.array_equal(first, last):
                    c = np.vstack([c, c[0:1]])
                all_c.append(c)
        all_c = filter_nested_contours(all_c)

        # Scale to original size and filter by area (min_area_contour is at original scale)
        min_area_contour = params.get("min_area_contour", 200)
        scaled_contours = []
        for c in all_c:
            s = c.copy().astype(np.float32)
            s[:, 0, 0] *= scale_y
            s[:, 0, 1] *= scale_x
            s = s.astype(np.int32)
            scaled_contours.append(s)
        filtered_contours = [c for c in scaled_contours if cv2.contourArea(c) >= min_area_contour]

        # Overlay on original
        overlay = original_img.copy()
        lt = max(2, int(scale_x))
        if filtered_contours:
            cv2.drawContours(overlay, filtered_contours, -1, (0, 255, 0), lt)

        # Contours-only canvas (flake_extraction reads this)
        contours_canvas = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        if filtered_contours:
            cv2.drawContours(contours_canvas, filtered_contours, -1, (255, 255, 255), lt)

        stem = input_path.stem
        overlay_path = output_dir / f"{stem}_binned_filtered_overlay.png"
        contours_path = output_dir / f"{stem}_binned_filtered_contours.png"
        Image.fromarray(overlay).save(overlay_path)
        Image.fromarray(contours_canvas).save(contours_path)

        print(f"  ✓ {input_path.name} -> {len(filtered_contours)} contours")
        return True

    except Exception as e:
        print(f"  ✗ {input_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch filtered-sensitive overlay pipeline (uses contour tuner settings)."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        nargs="?",
        default=Path("images/x20_images"),
        help="Input directory (default: images/x20_images)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("images/filtered_sensitive_overlays"),
        help="Output directory (default: images/filtered_sensitive_overlays)",
    )
    args = parser.parse_args()

    params = load_filtered_overlay_params()
    if BATCH_FILTERED_SETTINGS_PATH.exists():
        print("Using settings from batch_filtered_settings.json")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output)
    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}")
        return 1

    paths = sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        print(f"No images in {input_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(paths)} images from {input_dir}")
    print(f"Output: {output_dir}/")
    print()

    ok = 0
    for p in paths:
        if process_one(p, output_dir, params):
            ok += 1

    print(f"\nDone. {ok}/{len(paths)} images processed -> {output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
