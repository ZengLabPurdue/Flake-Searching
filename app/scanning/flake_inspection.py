"""Inspect one saved detection using the scan worker's exclusive hardware ownership."""

import copy
import json
import math
from pathlib import Path
import re
import threading
import uuid

import cv2
import numpy as np

from config import PIXEL_SIZE, RELATIVE_XY, RELATIVE_Z
from imaging import image_metadata
from scanning import contour_finder, flake_detection


def write_manifest(path, payload):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def detection_key(record):
    return (record.get("source_image"), record.get("classification"),
            json.dumps(record.get("bounding_box_px"), sort_keys=True))


def prepare_positions(scan_path):
    """Persist IDs and physical offsets for old and new saved detections.

    All XY origins use the stage position that would center the point at 10X.
    The coverage center is frozen on first use; chip origins come from the
    scanner's original chip-center folder coordinates, never flake averages.
    """
    path = Path(scan_path) / "flakes_found.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    cache = {}
    centers = []
    coverage_centers = []
    for mag in ("10X", "20X"):
        for source in (Path(scan_path) / "All Images" / mag.lower()).glob("*/Raw/*.png"):
            metadata = image_metadata.read_png_metadata(source)
            cache[source.relative_to(scan_path).as_posix()] = metadata
            if all(f"stage_{axis}_um" in metadata for axis in ("x", "y")):
                position = {axis: float(metadata[f"stage_{axis}_um"])
                            + RELATIVE_XY["10X"][axis.upper()] - RELATIVE_XY[mag][axis.upper()]
                            for axis in ("x", "y")}
                if all(math.isfinite(value) for value in position.values()):
                    coverage_centers.append(position)
    for record in manifest["detections"]:
        source = record.get("source_image")
        if not source:
            continue
        if source not in cache:
            image_path = flake_detection._scan_file(scan_path, source, "source")
            cache[source] = image_metadata.read_png_metadata(image_path)
        metadata = cache[source]
        mag = record.get("magnification") or metadata.get("magnification")
        if mag not in ("10X", "20X"):
            continue
        offset = {axis: RELATIVE_XY["10X"][axis.upper()] - RELATIVE_XY[mag][axis.upper()]
                  for axis in ("x", "y")}
        location = record.setdefault("location", {})
        stage = location.get("stage_position_um")
        if not stage:
            continue
        target = {axis: float(stage[axis]) + offset[axis] for axis in ("x", "y")}
        if not all(math.isfinite(value) for value in target.values()):
            continue
        location["target_position_10x_um"] = target
        record.setdefault("flake_id", "flake_" + uuid.uuid5(
            uuid.NAMESPACE_URL, str(Path(scan_path).resolve()) + repr(detection_key(record)),
        ).hex)
        chip_name = Path(source).parent.parent.name
        record.setdefault("chip_id", chip_name)
        match = re.search(r"\(([-+\d.eE]+),\s*([-+\d.eE]+)\)$", chip_name)
        if match:
            chip = {axis: float(value) + offset[axis]
                    for axis, value in zip(("x", "y"), match.groups())}
            location["chip_center_10x_um"] = chip
            location["chip_relative_um"] = {axis: target[axis] - chip[axis] for axis in chip}
        if "stage_z_um" in metadata:
            record["source_focus_z_10x_um"] = (float(metadata["stage_z_um"])
                                               + RELATIVE_Z["10X"] - RELATIVE_Z[mag])
        if metadata.get("vignette_substrate"):
            record["source_vignette_substrate"] = metadata["vignette_substrate"]
        centers.append(target)
    frames = manifest.setdefault("coordinate_frames", {})
    if centers:
        # Freeze a physical scan-coverage center. This is distinct from
        # the arbitrary pixel origin of a rendered/zoomed stitched map.
        extent = coverage_centers or centers
        origin = frames.setdefault("scan_center_10x_um", {
            axis: (min(p[axis] for p in extent) + max(p[axis] for p in extent)) / 2
            for axis in ("x", "y")
        })
        frames.setdefault("scan_center_definition", "center of saved scan-tile positions" if coverage_centers
                          else "center of initial detected-flake extent (legacy fallback)")
        frames["reference_objective"] = "10X"
        frames["units"] = "um"
        for record in manifest["detections"]:
            location = record.get("location", {})
            target = location.get("target_position_10x_um")
            if target:
                location["scan_relative_um"] = {axis: target[axis] - origin[axis] for axis in origin}
    write_manifest(path, manifest)
    return manifest


def centered_candidate(regions, record, shape, pixel_size):
    """Choose an unambiguous nearby region and a point inside its exact mask."""
    candidates = []
    for region in regions:
        if region.get("matched_class") != record.get("classification"):
            continue
        area = record.get("statistics", {}).get("area_um2")
        actual_area = region.get("area_um2")
        if area and actual_area and not 0.4 <= actual_area / area <= 2.5:
            continue
        box = region["bounding_box"]
        distance = math.hypot(box["x"] + box["width"] / 2 - shape[1] / 2,
                              box["y"] + box["height"] / 2 - shape[0] / 2) * pixel_size
        if distance <= 25:
            candidates.append((distance, region))
    candidates.sort(key=lambda item: item[0])
    if not candidates or (len(candidates) > 1 and candidates[1][0] - candidates[0][0] < 5):
        raise ValueError("Could not identify the selected flake unambiguously at 10X. "
                         "Return to results and check the selected flake and chip placement.")
    region = candidates[0][1]
    mask = np.asarray(region["instance_mask"], dtype=np.uint8)
    if not np.any(mask):
        raise ValueError("The selected region has an empty mask.")
    # Padding makes the distance well-defined even for a filled rectangular mask.
    distance = cv2.distanceTransform(np.pad(mask, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(np.argmax(distance), distance.shape)
    return region["bounding_box"]["x"] + int(x), region["bounding_box"]["y"] + int(y)


class FlakeInspection:
    def __init__(self, scan_path, record, notify):
        self.scan_path = Path(scan_path).resolve()
        self.record = copy.deepcopy(record)
        self.notify = notify
        self.confirm = threading.Event()
        self.refocus = threading.Event()
        self.center_selected = threading.Event()
        self.center_point = None
        self.ready = False
        self.saved = False

    def emit(self, manager, event, value):
        manager.ui_dispatch(self.notify, event, value)

    def capture(self, manager, magnification):
        manager._check_cancelled()
        manager._wait_for_stage()
        manager.frame_processor.arm_capture_after_motion(manager.stage.get_xy_position(strict=True))
        frame = manager.frame_processor.capture_frame(cancel_check=manager._check_cancelled)
        if frame is None:
            raise RuntimeError(f"The camera did not return a {magnification} image.")
        return frame

    def run(self, manager):
        manager._scan_execution_thread_id = threading.get_ident()
        motion_started = False
        previous_substrate = None
        try:
            manifest = prepare_positions(self.scan_path)
            matches = [r for r in manifest["detections"] if detection_key(r) == detection_key(self.record)]
            if len(matches) != 1:
                raise ValueError("The selected detection changed. Reopen the scan results.")
            self.record = copy.deepcopy(matches[0])
            target = self.record["location"].get("target_position_10x_um")
            if target is None:
                raise ValueError("This flake has no calibrated stage position.")
            profile_path = manifest.get("profile_path")
            if manifest.get("detection_model") != "Region Detection" or not profile_path:
                raise ValueError("Inspection requires a region-detection scan with its saved profile.")
            # Validate and freeze profile settings before starting hardware motion.
            configuration = contour_finder.region_classifier.load_profile_configuration(profile_path)
            manager._wait_for_existing_focus()
            substrate = self.record.get("source_vignette_substrate")
            if substrate:
                previous_substrate = manager.app.get_vignette_substrate()
                manager.app.set_vignette_substrate(substrate)
            self.emit(manager, "status", "Moving to the selected flake at 10X…")
            motion_started = True
            manager._change_objective(2, autofocus=False)
            manager._move_stage_xy(target["x"], target["y"])
            source_z = self.record.get("source_focus_z_10x_um")
            if source_z is not None and math.isfinite(source_z):
                manager._move_stage_z(source_z)
            manager._check_cancelled()
            manager.app.focus_controller.start_auto_focus_thread(wait=True)
            manager._check_cancelled()
            raw = self.capture(manager, "10X")
            corrected = manager.frame_processor.apply_vignette_filter(raw)
            pixel_size = PIXEL_SIZE["10X"][manager.app.get_resolution()]
            _, _, details = contour_finder.find_flakes(
                corrected, return_details=True, profile_configuration=configuration,
                pixel_size_um=pixel_size, draw_legend=False,
            )
            try:
                x, y = centered_candidate(details["combined_region_results"], self.record, raw.shape, pixel_size)
                self.record["centering_method"] = "profile_region"
            except ValueError:
                # Freeze this frame so a click uses exactly the pixels that
                # produced the centering coordinates, not a later live frame.
                self.emit(manager, "center", corrected)
                while not self.center_selected.wait(0.05):
                    manager._check_cancelled()
                manager._check_cancelled()
                x, y = self.center_point
                if not (0 <= x < raw.shape[1] and 0 <= y < raw.shape[0]):
                    raise ValueError("The centering point is outside the 10X frame.")
                self.record["centering_method"] = "user_selected"
            manager._check_cancelled()
            sx, sy = manager.stage.get_xy_position(strict=True)
            manager._move_stage_xy(sx - (x - raw.shape[1] / 2) * pixel_size,
                                   sy - (y - raw.shape[0] / 2) * pixel_size)
            self.record["location"]["refined_position_10x_um"] = dict(zip(
                ("x", "y"), manager.stage.get_xy_position(strict=True)))
            refined = self.record["location"]["refined_position_10x_um"]
            for label, origin in (
                ("chip", self.record["location"].get("chip_center_10x_um")),
                ("scan", manifest.get("coordinate_frames", {}).get("scan_center_10x_um")),
            ):
                if origin:
                    self.record["location"][f"refined_{label}_relative_um"] = {
                        axis: refined[axis] - origin[axis] for axis in ("x", "y")
                    }
            self.emit(manager, "status", "Capturing the 20X context image…")
            manager._change_objective(3)
            image20 = manager.frame_processor.apply_vignette_filter(self.capture(manager, "20X"))
            self.record["navigation_position_20x_um"] = dict(zip(
                ("x", "y", "z"), manager.stage.get_position(strict=True)))
            # Each inspection gets its own local captures; retries never overwrite a previous capture.
            safe_id = flake_detection._database_flake_id(self.record["flake_id"])
            self.capture_name = f"{safe_id}_{uuid.uuid4().hex[:10]}.png"
            path20 = manager.frame_processor.save_image(
                image20, save_dir=self.scan_path / "Selected Flakes" / "20x",
                filename=self.capture_name, vignette_applied=True,
                metadata={"flake_id": self.record["flake_id"]},
            )
            if path20 is None:
                raise RuntimeError("The 20X image could not be saved.")
            self.record["image_20x"] = path20.relative_to(self.scan_path).as_posix()
            self.emit(manager, "status", "Switching to 100X and focusing…")
            manager._change_objective(5)
            self.capture(manager, "100X")  # Require a fresh, settled frame before enabling confirmation.
            self.ready = True
            self.emit(manager, "ready", "Inspect the live 100X image, then confirm saving.")
            while not manager._stop_event.wait(0.05):
                if self.refocus.is_set():
                    self.refocus.clear()
                    self.emit(manager, "busy", "Focusing at 100X…")
                    manager.app.focus_controller.start_auto_focus_thread(wait=True)
                    self.capture(manager, "100X")
                    self.emit(manager, "ready", "Inspect the live 100X image, then confirm saving.")
                if self.confirm.is_set() and not self.saved:
                    self.confirm.clear()
                    self.emit(manager, "busy", "Saving 20X and 100X images to the database…")
                    try:
                        self.save(manager)
                    except Exception as exc:
                        manager._check_cancelled()
                        if self.saved:
                            self.emit(manager, "saved", f"Database export completed, but local status could not be updated: {exc}")
                        else:
                            self.emit(manager, "ready", f"Save failed: {exc}. You can retry.")
                    else:
                        self.saved = True
                        self.emit(manager, "saved", "Flake saved to the database. Next returns to results.")
        except Exception as exc:
            if not manager._stop_event.is_set():
                self.emit(manager, "error", str(exc))
        finally:
            self.ready = False
            try:
                # Complete cleanup before releasing hardware ownership. The stop
                # flag must not prevent returning from the high-power objective.
                if motion_started:
                    manager.stage.wait_until_not_busy()
                    manager.turret_controller.change_objective(2, autofocus=False)
                    manager.stage.wait_until_not_busy()
            finally:
                try:
                    if previous_substrate is not None:
                        manager.app.set_vignette_substrate(previous_substrate)
                finally:
                    manager._scan_execution_thread_id = None
                    self.emit(manager, "finished", None)

    def save(self, manager):
        if manager.app.get_magnification() != "100X":
            raise RuntimeError("The microscope is no longer at 100X.")
        frame = self.capture(manager, "100X")
        path100 = manager.frame_processor.save_image(
            frame, save_dir=self.scan_path / "Selected Flakes" / "100x",
            filename=self.capture_name, vignette_applied=False,
            metadata={"flake_id": self.record["flake_id"]},
        )
        if path100 is None:
            raise RuntimeError("The 100X image could not be saved.")
        self.record["image_100x"] = path100.relative_to(self.scan_path).as_posix()
        self.record["capture_position_100x_um"] = dict(zip(
            ("x", "y", "z"), manager.stage.get_position(strict=True)))
        manager._check_cancelled()
        path = self.scan_path / "flakes_found.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        matches = [r for r in manifest["detections"] if r.get("flake_id") == self.record["flake_id"]]
        if len(matches) != 1:
            raise ValueError("The selected detection changed before saving.")
        matches[0].update(self.record)
        write_manifest(path, manifest)
        exported = flake_detection.save_database(self.scan_path, self.record)
        self.saved = True  # A later local metadata failure must not duplicate the export.
        self.record["database_export"] = {"flake_id": exported["flake_id"], "json": str(exported["json"])}
        matches[0].update(self.record)
        write_manifest(path, manifest)
