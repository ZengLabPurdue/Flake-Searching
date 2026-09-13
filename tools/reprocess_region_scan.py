"""Rebuild a full scan's region results from saved PNGs, preserving old outputs."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

import cv2

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

from config import PIXEL_SIZE
from imaging.image_metadata import read_png_metadata, save_png
from scanning.contour_finder import find_flakes
from scanning.flake_detection import Flake_Detector
from scanning.scan_profile import ScanProfile


def reprocess(scan):
    scan = Path(scan).resolve()
    if scan.parent != (APP / "Scans").resolve() or not scan.name.startswith("Full Scan "):
        raise ValueError("Select a Full Scan directly inside app/Scans.")
    manifest_path = scan / "flakes_found.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("detection_model") != "Region Detection":
        raise ValueError("This tool requires an existing Region Detection scan.")
    profile = ScanProfile()
    profile.load_profile(manifest["profile_path"])
    sources = sorted(
        p for magnification in ("10x", "20x")
        for p in (scan / "All Images" / magnification).glob("*/Raw/*.png")
    )
    if not sources:
        raise ValueError("No saved 10x/20x raw images found.")
    source_keys = {p.relative_to(scan).as_posix() for p in sources}
    if any(d["source_image"] not in source_keys for d in manifest["detections"]):
        raise ValueError("Some existing detections have no corresponding source PNG.")

    run = APP / "Scan Result Backups" / scan.name / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    staged = run / "regenerated"
    original = run / "original"
    staged.mkdir(parents=True)
    original.mkdir()
    shutil.copy2(profile.path / "profile.json", run / "profile_used.json")
    # Record conversion requires no model initialization or microscope hardware.
    detector = Flake_Detector.__new__(Flake_Detector)
    records = []
    counts = []
    for index, source in enumerate(sources, 1):
        image = cv2.imread(str(source))
        if image is None:
            raise ValueError(f"Could not read {source}")
        metadata = read_png_metadata(source)
        magnification = metadata["magnification"]
        pixel_size = PIXEL_SIZE[magnification][metadata["resolution"]]
        annotated, _, details = find_flakes(
            image, return_details=True,
            profile_configuration=profile.matching_configuration(),
            pixel_size_um=pixel_size, color_seed=manifest.get("region_color_seed"),
        )
        detections = details["combined_region_results"]
        image_data = {
            "path": source, "magnification": magnification,
            "pixel_size_um": pixel_size,
            "stage_x": float(metadata["stage_x_um"]),
            "stage_y": float(metadata["stage_y_um"]),
        }
        for detection in detections:
            record = detector._create_detection_record(
                detection, "Region Detection", image_data, image.shape, scan,
            )
            if record is not None:
                records.append(record)
        processed = source.parent.parent / "Processed" / source.name
        found = scan / "Flakes Found" / source.parent.parent.name / source.name
        outputs = [(processed, details["classified_image_without_legend"])]
        if detections or found.exists():
            outputs.append((found, annotated))
        for destination, rgb in outputs:
            target = staged / destination.relative_to(scan)
            target.parent.mkdir(parents=True, exist_ok=True)
            save_png(target, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), metadata)
        counts.append({"source": source.relative_to(scan).as_posix(), "count": len(detections)})
        print(f"{index}/{len(sources)} {source.name}: {len(detections)} regions", flush=True)

    old_count = len(manifest["detections"])
    manifest["detections"] = records
    (staged / "flakes_found.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    replacements = sorted(staged.rglob("*.png")) + [staged / "flakes_found.json"]
    # Finish all processing and backups before replacing any result. Raw images
    # and all microscope calibration directories are never written.
    for source in replacements:
        relative = source.relative_to(staged)
        destination = scan / relative
        if destination.exists():
            backup = original / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, backup)
    report = {"old_count": old_count, "new_count": len(records), "images": counts}
    (run / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for source in replacements:
        destination = scan / source.relative_to(staged)
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
    print(f"Completed: {old_count} -> {len(records)} regions. Backup: {original}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan", type=Path)
    reprocess(parser.parse_args().scan)
