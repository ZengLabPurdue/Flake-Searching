#!/usr/bin/env python3
"""
Run the full 20x flake extraction pipeline.
Run from 20x_extraction: python run_flake_extraction.py [input_dir] [-o output_dir]
"""
import argparse
import subprocess
import sys
from pathlib import Path

# 20x_extraction folder is the project root (all scripts and images live here)
PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Run full flake extraction: raw images -> cropped flakes")
    parser.add_argument("input_dir", type=Path, nargs="?", default=PROJECT_ROOT / "images" / "x20_images")
    parser.add_argument("-o", "--output", type=Path, default=PROJECT_ROOT / "flake_extraction_outputs")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output)

    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}")
        return 1

    print("=" * 70)
    print("FULL FLAKE EXTRACTION PIPELINE")
    print("=" * 70)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print()

    overlay_dir = PROJECT_ROOT / "images" / "filtered_sensitive_overlays"
    robust_dir = PROJECT_ROOT / "robust_contours_and_masks_no_gap_close"

    # Step 1
    print("\n" + "=" * 70)
    print("STEP 1: Batch filtered sensitive overlays")
    print("=" * 70)
    r1 = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "batch_filtered_sensitive_overlays_2x2.py"), str(input_dir), "-o", str(overlay_dir)],
        cwd=str(PROJECT_ROOT),
    )
    if r1.returncode != 0:
        print("Step 1 failed.")
        return 1

    # Step 2
    print("\n" + "=" * 70)
    print("STEP 2: Batch robust contours and masks")
    print("=" * 70)
    r2 = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "batch_robust_contours_and_masks.py"), str(input_dir), "-o", str(robust_dir)],
        cwd=str(PROJECT_ROOT),
    )
    if r2.returncode != 0:
        print("Step 2 failed.")
        return 1

    # Step 3
    print("\n" + "=" * 70)
    print("STEP 3: Flake extraction pipeline")
    print("=" * 70)
    r3 = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "flake_extraction_pipeline.py"), str(input_dir), "-o", str(output_dir)],
        cwd=str(PROJECT_ROOT),
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
