#!/usr/bin/env python3
"""
Run the full flake extraction pipeline from raw images to cropped flakes.

1. batch_filtered_sensitive_overlays_2x2   -> contour overlays (uses tuner settings)
2. batch_robust_contours_and_masks         -> robust masks for background
3. flake_extraction_pipeline               -> extract individual flakes, crop, color

Usage:
    python run_full_flake_extraction.py [input_dir] [-o output_dir]
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run full flake extraction: raw images -> cropped flakes")
    project_root = Path(__file__).resolve().parent
    parser.add_argument("input_dir", type=Path, nargs="?", default=project_root / "images" / "x20_images")
    parser.add_argument("-o", "--output", type=Path, default=project_root / "flake_extraction_outputs")
    args = parser.parse_args()
    input_dir = Path(args.input_dir) if args.input_dir.is_absolute() else project_root / args.input_dir
    output_dir = Path(args.output) if args.output.is_absolute() else project_root / args.output

    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}")
        return 1

    print("=" * 70)
    print("FULL FLAKE EXTRACTION PIPELINE")
    print("=" * 70)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print()

    # Step 1: Filtered sensitive overlays (contours for flake extraction)
    print("\n" + "=" * 70)
    print("STEP 1: Batch filtered sensitive overlays (contour detection)")
    print("=" * 70)
    overlay_dir = project_root / "images" / "filtered_sensitive_overlays"
    r1 = subprocess.run(
        [
            sys.executable,
            str(project_root / "batch_filtered_sensitive_overlays_2x2.py"),
            str(input_dir),
            "-o", str(overlay_dir),
        ],
        cwd=str(project_root),
    )
    if r1.returncode != 0:
        print("Step 1 failed.")
        return 1

    # Step 2: Robust contours and masks (for background extraction in crops)
    print("\n" + "=" * 70)
    print("STEP 2: Batch robust contours and masks")
    print("=" * 70)
    robust_dir = project_root / "robust_contours_and_masks_no_gap_close"
    r2 = subprocess.run(
        [
            sys.executable,
            str(project_root / "batch_robust_contours_and_masks.py"),
            str(input_dir),
            "-o", str(robust_dir),
        ],
        cwd=str(project_root),
    )
    if r2.returncode != 0:
        print("Step 2 failed.")
        return 1

    # Step 3: Flake extraction (crops, background eliminator, color clustering)
    print("\n" + "=" * 70)
    print("STEP 3: Flake extraction pipeline")
    print("=" * 70)
    r3 = subprocess.run(
        [
            sys.executable,
            str(project_root / "flake_extraction_pipeline.py"),
            str(input_dir),
            "-o", str(output_dir),
        ],
        cwd=str(project_root),
    )
    if r3.returncode != 0:
        print("Step 3 failed.")
        return 1

    print("\n" + "=" * 70)
    print("DONE! Flake extraction outputs ->", output_dir)
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
