import json
import math
import re
import secrets
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from imaging import image_metadata
from scanning.flake_identifier import Flake_Identifier
from scanning.region_dataset import (
    NEW_FLAKE_FILTER_DIR,
    save_region_training_images,
)


FLAKE_CLASSIFICATIONS = {
    0: "Bad Flake",
    1: "Good Flake",
    2: "Not a Flake",
    3: "Unclear Flake",
}

DATABASE_TO_READ_DIR = Path(
    r"C:\Users\Zengl\Box\Zenglab_fabrication\DATABASE\to_read"
)
DATABASE_EXPORT_SCHEMA = "flake-search.database-import"
DATABASE_EXPORT_VERSION = 1


def _database_flake_id(value=None):
    if value is None:
        value = (
            datetime.now().strftime("flake_%Y%m%d_%H%M%S_%f_")
            + secrets.token_hex(3)
        )
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("._")
    if not value:
        raise ValueError("The database flake ID is empty.")
    return value


def _scan_file(scan_path, manifest_value, label):
    if not isinstance(manifest_value, (str, Path)) or not str(manifest_value).strip():
        raise ValueError(f"The selected result has no {label} image.")
    scan_root = Path(scan_path).resolve()
    candidate = Path(manifest_value)
    if not candidate.is_absolute():
        candidate = scan_root / candidate
    try:
        candidate = candidate.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(f"The {label} image was not found: {candidate}") from exc
    if candidate != scan_root and scan_root not in candidate.parents:
        raise ValueError(f"The {label} image is outside the selected scan folder.")
    if not candidate.is_file():
        raise FileNotFoundError(f"The {label} image was not found: {candidate}")
    return candidate


def encode_region_mask(mask):
    """Encode a bounding-box-local binary mask as compact row-major runs."""
    array = np.asarray(mask)
    if array.ndim != 2 or not array.size:
        return None
    foreground = np.ascontiguousarray(array > 0).reshape(-1)
    padded = np.concatenate((
        np.zeros(1, dtype=np.uint8),
        foreground,
        np.zeros(1, dtype=np.uint8),
    ))
    transitions = np.flatnonzero(np.diff(padded.astype(np.int8)))
    runs = [
        value
        for start, end in transitions.reshape(-1, 2)
        for value in (int(start), int(end - start))
    ]
    return {
        "encoding": "foreground-runs-row-major",
        "width": int(array.shape[1]),
        "height": int(array.shape[0]),
        "runs": runs,
    }


def save_database(
    scan_path,
    detection_record,
    destination=DATABASE_TO_READ_DIR,
    flake_id=None,
):
    """Copy one selected 100x result and its JSON into the DB intake folder.

    Image files are finalized before the JSON, preventing an intake watcher
    from observing a record whose image upload is still incomplete. Source
    scan files are copied and remain untouched.
    """
    if not isinstance(detection_record, dict):
        raise ValueError("A selected detection record is required.")

    scan_root = Path(scan_path).resolve()
    image_20x = _scan_file(scan_root, detection_record.get("image_20x"), "20X")
    image_100x = _scan_file(scan_root, detection_record.get("image_100x"), "100X")
    export_id = _database_flake_id(flake_id)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    final_images = {
        "20X": destination / f"{export_id}_20X{image_20x.suffix.lower()}",
        "100X": destination / f"{export_id}_100X{image_100x.suffix.lower()}",
    }
    json_path = destination / f"{export_id}.json"
    final_paths = (*final_images.values(), json_path)
    if any(path.exists() for path in final_paths):
        raise FileExistsError(f"A database export already exists for {export_id}.")

    manifest = {}
    manifest_path = scan_root / "flakes_found.json"
    try:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            manifest = loaded
    except (OSError, UnicodeError, json.JSONDecodeError):
        pass

    metadata = {
        "material": manifest.get("material"),
        "substrate": manifest.get("substrate_thickness"),
        "classification": detection_record.get("classification"),
        "classification_group": detection_record.get("classification_group"),
        "statistics": detection_record.get("statistics", {}),
        "location": detection_record.get("location", {}),
        "scan_name": scan_root.name,
        "scan_schema": manifest.get("schema"),
        "scan_version": manifest.get("version"),
        "detection": detection_record,
    }
    payload = {
        "schema": DATABASE_EXPORT_SCHEMA,
        "version": DATABASE_EXPORT_VERSION,
        "flake_id": export_id,
        "date_found": datetime.now().strftime("%m-%d-%Y"),
        "images": {
            magnification: str(path.resolve())
            for magnification, path in final_images.items()
        },
        "metadata": metadata,
    }

    token = secrets.token_hex(6)
    temporary_images = {
        key: destination / f".{path.name}.{token}.part"
        for key, path in final_images.items()
    }
    temporary_json = destination / f".{json_path.name}.{token}.part"
    created = []
    try:
        shutil.copy2(image_20x, temporary_images["20X"])
        shutil.copy2(image_100x, temporary_images["100X"])
        temporary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        for magnification in ("20X", "100X"):
            temporary_images[magnification].replace(final_images[magnification])
            created.append(final_images[magnification])
        temporary_json.replace(json_path)
        created.append(json_path)
    except Exception:
        for path in (*temporary_images.values(), temporary_json, *created):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise

    return {
        "flake_id": export_id,
        "images": dict(final_images),
        "json": json_path,
    }


class Flake_Detector:

    def __init__(self):
        
        self.flake_identifier = Flake_Identifier()

    def flake_detection(
        self,
        image_queue,
        frame_processor,
        stop_requested=None,
        detection_model="Flake Detection",
        profile_path=None,
        scan_path=None,
        progress_callback=None,
        draw_class_legend=True,
    ):
        if detection_model not in ("Flake Detection", "Region Detection"):
            raise ValueError(f"Unknown detection model: {detection_model}")
        if detection_model == "Region Detection" and profile_path is None:
            raise ValueError("A scan profile is required for region detection.")

        scan_path = Path(scan_path) if scan_path is not None else None
        detection_records = []
        color_seed = secrets.randbits(32) if detection_model == "Region Detection" else None
        processed_count = 0

        try:
            while True:
                if stop_requested is not None and stop_requested():
                    break
                image_data = image_queue.get()
                try:
                    if image_data is None:
                        break
                    if isinstance(image_data, dict):
                        img_path = Path(image_data["path"])
                    else:
                        img_path = Path(image_data)
                        image_data = {"path": img_path}

                    print(img_path)
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    vignette_applied = image_metadata.is_vignette_corrected(img_path)
                    source_metadata = image_metadata.read_png_metadata(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    if detection_model == "Region Detection":
                        scanned_img, detections, save, processed_img = (
                            self.flake_identifier.identify_flakes_region_model(
                                img,
                                profile_path,
                                color_seed=color_seed,
                                return_segmented_map=True,
                                pixel_size_um=image_data.get("pixel_size_um"),
                                draw_legend=draw_class_legend,
                            )
                        )
                    else:
                        scanned_img, detections, save = (
                            self.flake_identifier.identify_flakes_flake_model(img)
                        )
                        # The classifier preview contains bounding boxes and labels.
                        # Keep the stitched Processed map clean; annotations belong
                        # only in the per-image Flakes Found view.
                        processed_img = img.copy()

                    for detection in detections:
                        record = self._create_detection_record(
                            detection,
                            detection_model,
                            image_data,
                            img.shape,
                            scan_path,
                        )
                        if record is not None:
                            detection_records.append(record)

                    if detection_model == "Region Detection" and detections:
                        chip_folder = img_path.parent.parent
                        scan_root = scan_path or chip_folder.parent.parent.parent
                        save_region_training_images(
                            img,
                            detections,
                            NEW_FLAKE_FILTER_DIR
                            / scan_root.name
                            / chip_folder.name,
                            img_path.name,
                        )

                    out_path = img_path.parent.parent / "Processed" / img_path.name
                    # Processed tiles are stitched into a map, so keep them
                    # free of per-tile legends. Flakes Found images are viewed
                    # individually and retain the optional legend below.
                    frame_processor.save_image(
                        cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR),
                        save_dir=out_path.parent,
                        filename=out_path.name,
                        vignette_applied=vignette_applied,
                        metadata=source_metadata,
                    )

                    if save:
                        chip_folder = img_path.parent.parent
                        scan_root = scan_path or chip_folder.parent.parent.parent
                        flakes_dir = scan_root / "Flakes Found" / chip_folder.name
                        flakes_dir.mkdir(parents=True, exist_ok=True)
                        frame_processor.save_image(
                            cv2.cvtColor(scanned_img, cv2.COLOR_RGB2BGR),
                            save_dir=flakes_dir,
                            filename=img_path.name,
                            vignette_applied=vignette_applied,
                            metadata=source_metadata,
                        )
                finally:
                    image_queue.task_done()
                    if image_data is not None:
                        processed_count += 1
                        if progress_callback is not None:
                            progress_callback(
                                processed_count,
                                image_queue.qsize(),
                            )
        finally:
            if scan_path is not None:
                self._save_detection_records(
                    scan_path,
                    detection_model,
                    profile_path,
                    detection_records,
                    color_seed,
                )

    @staticmethod
    def _bounded_box(bounding_box, image_shape):
        if isinstance(bounding_box, dict):
            values = (
                bounding_box.get("x"),
                bounding_box.get("y"),
                bounding_box.get("width"),
                bounding_box.get("height"),
            )
        else:
            values = bounding_box

        if values is None or len(values) != 4:
            return None
        try:
            x, y, width, height = (float(value) for value in values)
        except (TypeError, ValueError):
            return None

        image_height, image_width = image_shape[:2]
        x1 = max(0.0, min(float(image_width), x))
        y1 = max(0.0, min(float(image_height), y))
        x2 = max(x1, min(float(image_width), x + max(0.0, width)))
        y2 = max(y1, min(float(image_height), y + max(0.0, height)))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2 - x1, y2 - y1

    def _create_detection_record(
        self,
        detection,
        detection_model,
        image_data,
        image_shape,
        scan_path,
    ):
        if detection_model == "Region Detection":
            classification = detection.get("matched_class")
            classification_group = detection.get("matched_group")
            bounding_box = detection.get("bounding_box")
            class_id = None
            accepted = bool(detection.get("accepted", True))
            statistic_source = detection
        else:
            if isinstance(detection, dict):
                class_id = detection.get("class_id")
                bounding_box = detection.get("bounding_box")
                statistic_source = detection.get("statistics", detection)
            else:
                try:
                    class_id, bounding_box = detection[:2]
                except (TypeError, ValueError):
                    return None
                statistic_source = {}
            if not isinstance(statistic_source, dict):
                statistic_source = {}
            try:
                class_id = int(class_id)
            except (TypeError, ValueError):
                return None
            classification = FLAKE_CLASSIFICATIONS.get(
                class_id,
                f"Class {class_id}",
            )
            classification_group = None
            accepted = class_id == 1

        bounded_box = self._bounded_box(bounding_box, image_shape)
        if classification is None or bounded_box is None:
            return None

        x, y, width, height = bounded_box
        center_x = x + width / 2
        center_y = y + height / 2
        img_path = Path(image_data["path"])
        try:
            source_image = img_path.relative_to(scan_path).as_posix()
        except (TypeError, ValueError):
            source_image = str(img_path)

        record = {
            "classification": classification,
            "classification_group": classification_group,
            "accepted": accepted,
            "source_image": source_image,
            "magnification": image_data.get("magnification"),
            "bounding_box_px": {
                "x": round(x, 3),
                "y": round(y, 3),
                "width": round(width, 3),
                "height": round(height, 3),
            },
            "location": {
                "bounding_box_center_px": {
                    "x": round(center_x, 3),
                    "y": round(center_y, 3),
                },
            },
        }
        if class_id is not None:
            record["class_id"] = class_id
        elif isinstance(detection, dict):
            color = detection.get("matched_color_rgb")
            try:
                color = [int(channel) for channel in color]
            except (TypeError, ValueError):
                color = None
            if color is not None and len(color) == 3 and all(
                0 <= channel <= 255 for channel in color
            ):
                record["display_color_rgb"] = color
            encoded_mask = encode_region_mask(detection.get("instance_mask"))
            if encoded_mask is not None:
                record["region_mask"] = encoded_mask

        statistic_keys = (
            "pixel_count",
            "size_um",
            "area_um2",
            "width_um",
            "height_um",
            "perimeter_px",
            "perimeter_um",
            "perimeter_area_ratio_per_um",
        )
        statistics = {}
        for key in statistic_keys:
            value = statistic_source.get(key)
            if value is None or isinstance(value, bool):
                statistics[key] = None
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                statistics[key] = None
                continue
            statistics[key] = number if math.isfinite(number) else None

        if class_id is not None:
            try:
                calibrated_pixel_size = float(image_data.get("pixel_size_um"))
            except (TypeError, ValueError):
                calibrated_pixel_size = None
            if (
                calibrated_pixel_size is None
                or not math.isfinite(calibrated_pixel_size)
                or calibrated_pixel_size <= 0
            ):
                calibrated_pixel_size = None
            if calibrated_pixel_size is not None:
                statistics["width_um"] = width * calibrated_pixel_size
                statistics["height_um"] = height * calibrated_pixel_size
                statistics["size_um"] = max(width, height) * calibrated_pixel_size
                if statistics["pixel_count"] is not None:
                    statistics["area_um2"] = (
                        statistics["pixel_count"] * calibrated_pixel_size ** 2
                    )
                if statistics["perimeter_px"] is not None:
                    statistics["perimeter_um"] = (
                        statistics["perimeter_px"] * calibrated_pixel_size
                    )
                if statistics["area_um2"] and statistics["perimeter_um"] is not None:
                    statistics["perimeter_area_ratio_per_um"] = (
                        statistics["perimeter_um"] / statistics["area_um2"]
                    )

        pixel_count = statistics["pixel_count"]
        statistics["pixel_count"] = (
            int(round(pixel_count)) if pixel_count is not None else None
        )
        record["statistics"] = statistics

        stage_x = image_data.get("stage_x")
        stage_y = image_data.get("stage_y")
        pixel_size_um = image_data.get("pixel_size_um")
        if stage_x is not None and stage_y is not None and pixel_size_um is not None:
            image_height, image_width = image_shape[:2]
            target_x = float(stage_x) - (center_x - image_width / 2) * float(pixel_size_um)
            target_y = float(stage_y) - (center_y - image_height / 2) * float(pixel_size_um)
            record["location"]["stage_position_um"] = {
                "x": round(target_x, 6),
                "y": round(target_y, 6),
            }
            record["capture_stage_position_um"] = {
                "x": round(float(stage_x), 6),
                "y": round(float(stage_y), 6),
            }

        return record

    @staticmethod
    def _save_detection_records(
        scan_path,
        detection_model,
        profile_path,
        detection_records,
        color_seed=None,
    ):
        scan_path.mkdir(parents=True, exist_ok=True)
        data = {
            "schema": "flake-search.scan-detections",
            "version": 2,
            "detection_model": detection_model,
            "profile_path": str(profile_path) if profile_path is not None else None,
            "region_color_seed": color_seed,
            "detections": detection_records,
        }
        (scan_path / "flakes_found.json").write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )
