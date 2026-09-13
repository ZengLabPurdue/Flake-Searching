from datetime import datetime, timezone
import json
from pathlib import Path
import re
import unicodedata
from uuid import uuid4

import cv2
import numpy as np

from config import HOME_DIR
from imaging import image_metadata
from .contour_finder import region_contrast_rgb
from .contour_extractor import get_region_from_point


PROFILE_SCHEMA = "flake-search.scan-search-profile"
PROFILE_VERSION = 7
SUPPORTED_PROFILE_VERSIONS = {1, 2, 3, 4, 5, 6, PROFILE_VERSION}
PROFILE_FILENAME = "profile.json"
DEFAULT_PROFILES_DIR = HOME_DIR / "Profiles"
CLASSIFICATION_SAMPLE = "sample"
CLASSIFICATION_RANGE = "range"
CLASSIFICATION_TYPES = {CLASSIFICATION_SAMPLE, CLASSIFICATION_RANGE}
FILTER_TYPE_SIZE = "size"
FILTER_TYPE_BOUNDING_BOX = "bounding_box"
FILTER_TYPE_PERIMETER_AREA_RATIO = "perimeter_area_ratio"
FILTER_TYPE_NEURAL_NETWORK = "neural_network"
FILTER_TYPES = {
    FILTER_TYPE_SIZE,
    FILTER_TYPE_BOUNDING_BOX,
    FILTER_TYPE_PERIMETER_AREA_RATIO,
    FILTER_TYPE_NEURAL_NETWORK,
}
FILTER_TYPE_LABELS = {
    FILTER_TYPE_SIZE: "Size",
    FILTER_TYPE_BOUNDING_BOX: "Width and height",
    FILTER_TYPE_PERIMETER_AREA_RATIO: "Perimeter / area",
    FILTER_TYPE_NEURAL_NETWORK: "Neural network (coming soon)",
}
FILTER_BOUND_KEYS = {
    FILTER_TYPE_SIZE: ("minimum_um", "maximum_um"),
    FILTER_TYPE_BOUNDING_BOX: (
        "minimum_width_um",
        "maximum_width_um",
        "minimum_height_um",
        "maximum_height_um",
    ),
    FILTER_TYPE_PERIMETER_AREA_RATIO: (
        "minimum_per_um",
        "maximum_per_um",
    ),
    FILTER_TYPE_NEURAL_NETWORK: (),
}
ScanProfileError = ValueError


_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _profile_folder_name(name):
    """Return a path-safe profile folder name without changing its display."""
    name = str(name).strip()
    if not name:
        raise ScanProfileError("Enter a profile name before saving.")
    if name in {".", ".."} or re.search(r'[<>:"/\\|?*\x00-\x1f]', name):
        raise ScanProfileError(
            "Profile names cannot contain path separators or "
            'the characters < > : " | ? *.'
        )
    if name.endswith((".", " ")):
        raise ScanProfileError("Profile names cannot end with a dot or space.")
    if name.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise ScanProfileError(f"'{name}' is a reserved Windows folder name.")
    return name


def _profile_slug(name):
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-") or "profile"
    return f"profile_{slug}" if slug.upper() in _WINDOWS_RESERVED_NAMES else slug


def _read_image(path):
    try:
        data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except (OSError, cv2.error):
        return None


def _write_image(path, image):
    image_metadata.save_png(
        path,
        image,
        metadata={
            "vignette_applied": True,
            "source": "scan_profile",
        },
    )


def build_region_overlay(image_bgr, region_mask, seed_point):
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ScanProfileError("The source image must be a three-channel BGR image.")
    if region_mask is None or region_mask.shape != image_bgr.shape[:2]:
        raise ScanProfileError("The region mask does not match the source image.")

    selected = region_mask > 0
    if not np.any(selected):
        raise ScanProfileError("The selected region is empty.")

    preview = image_bgr.copy()
    green = np.full_like(preview, (0, 255, 0))
    blended = cv2.addWeighted(preview, 0.68, green, 0.32, 0)
    preview[selected] = blended[selected]
    contours, _ = cv2.findContours(
        selected.astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(preview, contours, -1, (255, 255, 255), 2)
    cv2.circle(preview, seed_point, 5, (0, 0, 255), -1)
    return preview


class ScanProfile:
    def __init__(self, profiles_dir=DEFAULT_PROFILES_DIR):
        self.profiles_dir = Path(profiles_dir)
        self.clear()

    def clear(self):
        self.name = ""
        self.path = None
        self.created_at = ""
        self.updated_at = ""
        self.classes = []
        self.filters = []

    def profile_directory(self, name):
        return self.profiles_dir / _profile_folder_name(name)

    def find_class(self, name):
        name = name.strip().casefold()
        for index, profile_class in enumerate(self.classes):
            if profile_class["name"].casefold() == name:
                return index
        return None

    def get_class(self, index):
        return self.classes[index]

    def set_class(
        self,
        name,
        source_path,
        image_bgr,
        region_mask,
        seed_point,
        threshold,
        group="",
        identify=True,
        connectivity=8,
        index=None,
        derived_from=None,
    ):
        class_id = None
        if index is not None:
            existing_class = self.classes[index]
            class_id = existing_class["id"]
            if derived_from is None:
                derived_from = existing_class.get("derived_from")
        profile_class = self._make_class(
            name,
            source_path,
            image_bgr,
            region_mask,
            seed_point,
            threshold,
            connectivity,
            class_id=class_id,
            group=group,
            identify=identify,
            derived_from=derived_from,
        )
        duplicate = self.find_class(profile_class["name"])
        if duplicate is not None and duplicate != index:
            raise ScanProfileError(f"A class named '{profile_class['name']}' already exists.")
        if index is None:
            self.classes.append(profile_class)
        else:
            self.classes[index] = profile_class
        return profile_class

    def set_range_class(
        self,
        name,
        minimum_contrast_rgb,
        maximum_contrast_rgb,
        group="",
        identify=True,
        index=None,
        derived_from=None,
    ):
        class_id = None
        if index is not None:
            existing_class = self.classes[index]
            class_id = existing_class["id"]
            if derived_from is None:
                derived_from = existing_class.get("derived_from")
        profile_class = self._make_range_class(
            name,
            minimum_contrast_rgb,
            maximum_contrast_rgb,
            class_id=class_id,
            group=group,
            identify=identify,
            derived_from=derived_from,
        )
        duplicate = self.find_class(profile_class["name"])
        if duplicate is not None and duplicate != index:
            raise ScanProfileError(f"A class named '{profile_class['name']}' already exists.")
        if index is None:
            self.classes.append(profile_class)
        else:
            self.classes[index] = profile_class
        return profile_class

    def copy_class(self, index, name=None, *, source_profile_name=None):
        """Copy a class into this profile with a new identity and unique label."""
        source = self.get_class(index)
        copied_name = self._unique_class_name(name or f"{source['name']} copy")
        origin = {
            "profile": source_profile_name or self.name or None,
            "class_id": source["id"],
        }
        copied = self._copy_class_record(source, copied_name, origin)
        self.classes.append(copied)
        return copied

    def get_filter(self, filter_id):
        for profile_filter in self.filters:
            if profile_filter["id"] == filter_id:
                return profile_filter
        raise ScanProfileError(f"Filter was not found: {filter_id}")

    def set_filter(self, filter_id, name, filter_type, enabled, **bounds):
        """Add or update one named profile-wide geometry filter."""
        existing_index = None
        if filter_id is not None:
            existing_index = next(
                (
                    index
                    for index, profile_filter in enumerate(self.filters)
                    if profile_filter["id"] == filter_id
                ),
                None,
            )
            if existing_index is None:
                raise ScanProfileError(f"Filter was not found: {filter_id}")
        profile_filter = self._make_filter(
            name,
            filter_type,
            enabled,
            filter_id=filter_id,
            **bounds,
        )
        duplicate_name = next(
            (
                index
                for index, item in enumerate(self.filters)
                if item["name"].casefold() == profile_filter["name"].casefold()
            ),
            None,
        )
        if duplicate_name is not None and duplicate_name != existing_index:
            raise ScanProfileError(
                f"A filter named '{profile_filter['name']}' already exists."
            )
        if existing_index is None:
            self.filters.append(profile_filter)
        else:
            self.filters[existing_index] = profile_filter
        return profile_filter

    def copy_filter(self, filter_id, name=None):
        source = self.get_filter(filter_id)
        copied_name = self._unique_filter_name(name or f"{source['name']} copy")
        copied = self._make_filter(
            copied_name,
            source["type"],
            source["enabled"],
            **{
                key: source.get(key)
                for key in FILTER_BOUND_KEYS[source["type"]]
            },
        )
        self.filters.append(copied)
        return copied

    def remove_filter(self, filter_id):
        for index, profile_filter in enumerate(self.filters):
            if profile_filter["id"] == filter_id:
                return self.filters.pop(index)
        raise ScanProfileError(f"Filter was not found: {filter_id}")

    def move_filter(self, old_index, new_index):
        profile_filter = self.filters.pop(old_index)
        self.filters.insert(new_index, profile_filter)
        return profile_filter

    def extend_from_profile(self, source_profile):
        """Append independent copies of another profile's classes and filters."""
        added = []
        for source in source_profile.classes:
            copied_name = self._unique_class_name(source["name"])
            origin = {
                "profile": source_profile.name or None,
                "class_id": source["id"],
            }
            copied = self._copy_class_record(source, copied_name, origin)
            self.classes.append(copied)
            added.append(copied)
        for source in source_profile.filters:
            copied = self._make_filter(
                self._unique_filter_name(source["name"]),
                source["type"],
                source["enabled"],
                **{
                    key: source.get(key)
                    for key in FILTER_BOUND_KEYS[source["type"]]
                },
            )
            self.filters.append(copied)
        return added

    def matching_configuration(self, contrast_threshold=None):
        """Build classifier input directly from the editable in-memory profile."""
        classes = []
        for class_index, profile_class in enumerate(self.classes):
            classification_type = profile_class["classification_type"]
            configured_class = {
                "name": profile_class["name"],
                "classification_type": classification_type,
                "class_index": class_index,
                "group": profile_class["group"],
                "identify": profile_class["identify"],
            }
            if classification_type == CLASSIFICATION_RANGE:
                configured_class.update({
                    "minimum_contrast_rgb": np.asarray(
                        profile_class["minimum_contrast_rgb"],
                        dtype=np.float64,
                    ),
                    "maximum_contrast_rgb": np.asarray(
                        profile_class["maximum_contrast_rgb"],
                        dtype=np.float64,
                    ),
                })
            else:
                tolerance = (
                    profile_class["threshold"]
                    if contrast_threshold is None
                    else contrast_threshold
                )
                configured_class.update({
                    "contrast_rgb": np.asarray(
                        profile_class["contrast_rgb"],
                        dtype=np.float64,
                    ),
                    "tolerance": float(tolerance),
                })
            classes.append(configured_class)
        return {"classes": classes, "filters": [dict(item) for item in self.filters]}

    @property
    def groups(self):
        return sorted(
            {item["group"] for item in self.classes if item["group"]},
            key=str.casefold,
        )

    def remove_class(self, index):
        return self.classes.pop(index)

    def move_class(self, old_index, new_index):
        profile_class = self.classes.pop(old_index)
        self.classes.insert(new_index, profile_class)
        return profile_class

    def save_profile(self, name=None, overwrite=False):
        name = (self.name if name is None else name).strip()
        if not name:
            raise ScanProfileError("Enter a profile name before saving.")
        if not self.classes:
            raise ScanProfileError("Add at least one confirmed class before saving.")

        names = [profile_class["name"].casefold() for profile_class in self.classes]
        if len(names) != len(set(names)):
            raise ScanProfileError("Class names must be unique.")
        for profile_class in self.classes:
            self._validate_class(profile_class)
        for profile_filter in self.filters:
            self._validate_filter(profile_filter)
        filter_ids = [item["id"] for item in self.filters]
        if len(filter_ids) != len(set(filter_ids)):
            raise ScanProfileError("Filter IDs must be unique.")
        filter_names = [item["name"].casefold() for item in self.filters]
        if len(filter_names) != len(set(filter_names)):
            raise ScanProfileError("Filter names must be unique.")

        destination = self.profile_directory(name)
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)

        destination.mkdir(parents=True, exist_ok=True)
        images_dir = destination / "images"
        saved_classes = []

        for profile_class in self.classes:
            class_id = profile_class["id"]
            saved_class = {
                "id": class_id,
                "name": profile_class["name"],
                "classification_type": profile_class["classification_type"],
                "group": profile_class["group"] or None,
                "identify": profile_class["identify"],
            }
            if profile_class["classification_type"] == CLASSIFICATION_RANGE:
                minimum_rgb = profile_class["minimum_contrast_rgb"]
                maximum_rgb = profile_class["maximum_contrast_rgb"]
                saved_class["contrast_range_rgb"] = {
                    "minimum": {
                        "red": minimum_rgb[0],
                        "green": minimum_rgb[1],
                        "blue": minimum_rgb[2],
                    },
                    "maximum": {
                        "red": maximum_rgb[0],
                        "green": maximum_rgb[1],
                        "blue": maximum_rgb[2],
                    },
                }
            else:
                images_dir.mkdir(parents=True, exist_ok=True)
                filename = (
                    f"{_profile_slug(profile_class['name'])}_{class_id[:8]}_source.png"
                )
                source_path = images_dir / filename
                _write_image(source_path, profile_class["image_bgr"])
                red, green, blue = profile_class["contrast_rgb"]
                saved_class.update({
                    "source_image": f"images/{filename}",
                    "seed_point": {
                        "x": profile_class["seed_point"][0],
                        "y": profile_class["seed_point"][1],
                    },
                    "flood_fill": {
                        "threshold": profile_class["threshold"],
                        "connectivity": profile_class["connectivity"],
                    },
                    "contrast_rgb": {"red": red, "green": green, "blue": blue},
                })
            saved_classes.append(saved_class)
            if profile_class["derived_from"] is not None:
                saved_classes[-1]["derived_from"] = profile_class["derived_from"]

        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "schema": PROFILE_SCHEMA,
            "version": PROFILE_VERSION,
            "name": name,
            "created_at": now,
            "updated_at": now,
            "filters": [dict(item) for item in self.filters],
            "classes": saved_classes,
        }
        (destination / PROFILE_FILENAME).write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        return self.load_profile(destination)

    def load_profile(self, path):
        path = Path(path)
        profile_path = path / PROFILE_FILENAME if path.is_dir() else path
        if not profile_path.is_file():
            raise ScanProfileError(f"Profile JSON was not found: {profile_path}")

        try:
            payload = json.loads(profile_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ScanProfileError(f"Could not read profile JSON: {exc}") from exc

        if not isinstance(payload, dict) or payload.get("schema") != PROFILE_SCHEMA:
            raise ScanProfileError("This file is not a scan search profile.")
        profile_version = payload.get("version")
        if (
            type(profile_version) is not int
            or profile_version not in SUPPORTED_PROFILE_VERSIONS
        ):
            raise ScanProfileError(f"Unsupported profile version: {payload.get('version')!r}.")

        name = payload.get("name")
        saved_classes = payload.get("classes")
        if not isinstance(name, str) or not name.strip():
            raise ScanProfileError("The profile name is missing.")
        if not isinstance(saved_classes, list):
            raise ScanProfileError("The profile class list must be an array.")
        if not saved_classes:
            raise ScanProfileError("The profile must contain at least one class.")

        profile_dir = profile_path.parent.resolve()
        loaded_filters = []
        if profile_version >= PROFILE_VERSION:
            saved_filters = payload.get("filters", [])
            if not isinstance(saved_filters, list):
                raise ScanProfileError("The profile filter list must be an array.")
            for index, saved_filter in enumerate(saved_filters, start=1):
                if not isinstance(saved_filter, dict):
                    raise ScanProfileError(f"Filter {index} must be a JSON object.")
                filter_type = saved_filter.get("type")
                if filter_type not in FILTER_TYPES:
                    raise ScanProfileError(f"Filter {index} has an invalid type.")
                self._validate_filter(saved_filter)
                loaded_filters.append(self._make_filter(
                    saved_filter.get("name"),
                    filter_type,
                    saved_filter.get("enabled"),
                    filter_id=saved_filter.get("id"),
                    **{
                        key: saved_filter.get(key)
                        for key in FILTER_BOUND_KEYS[filter_type]
                    },
                ))
        else:
            # Versions before 7 had profile/class size requirements and, in
            # version 4, an unrelated generic ``filters`` field. Only the
            # profile-wide size requirement has a faithful v7 equivalent.
            legacy_minimum, legacy_maximum = self._read_legacy_size(
                payload.get("size_requirement")
            )
            if legacy_minimum is not None or legacy_maximum is not None:
                loaded_filters.append(self._make_filter(
                    "Legacy size",
                    FILTER_TYPE_SIZE,
                    True,
                    minimum_um=legacy_minimum,
                    maximum_um=legacy_maximum,
                ))
        loaded_classes = []

        for index, saved_class in enumerate(saved_classes, start=1):
            if not isinstance(saved_class, dict):
                raise ScanProfileError(f"Class {index} must be a JSON object.")

            class_name = saved_class.get("name")
            if not isinstance(class_name, str) or not class_name.strip():
                raise ScanProfileError(f"Class {index} has no name.")
            identify = saved_class.get("identify", True)
            classification_type = saved_class.get(
                "classification_type",
                CLASSIFICATION_SAMPLE,
            )
            if classification_type not in CLASSIFICATION_TYPES:
                raise ScanProfileError(
                    f"Class {class_name} has an invalid classification type."
                )
            if classification_type == CLASSIFICATION_RANGE:
                if profile_version < 6:
                    raise ScanProfileError(
                        f"Class {class_name} uses ranges in an older profile version."
                    )
                minimum_rgb, maximum_rgb = self._read_contrast_range(
                    saved_class.get("contrast_range_rgb"),
                    class_name,
                )
                loaded_classes.append(self._make_range_class(
                    class_name,
                    minimum_rgb,
                    maximum_rgb,
                    class_id=saved_class.get("id"),
                    group=saved_class.get("group", ""),
                    identify=identify,
                    derived_from=saved_class.get("derived_from"),
                ))
                continue

            seed = saved_class.get("seed_point")
            flood_fill = saved_class.get("flood_fill")
            if not isinstance(seed, dict) or not isinstance(flood_fill, dict):
                raise ScanProfileError(f"Class {index} is missing region metadata.")
            try:
                seed_point = (int(seed["x"]), int(seed["y"]))
                threshold = int(flood_fill["threshold"])
                connectivity = int(flood_fill["connectivity"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ScanProfileError(f"Class {index} has invalid region metadata.") from exc

            source_path = self._asset_path(
                profile_dir,
                saved_class.get("source_image"),
                index,
            )
            if not image_metadata.is_vignette_corrected(source_path):
                raise ScanProfileError(
                    f"Class {class_name} does not use a vignette-corrected PNG image."
                )
            image = _read_image(source_path)
            if image is None:
                raise ScanProfileError(f"Could not read source image for class {class_name}.")
            region_mask = self._region_mask(
                image,
                seed_point,
                threshold,
                connectivity,
                class_name,
            )
            contrast = saved_class.get("contrast_rgb")
            if isinstance(contrast, dict):
                contrast = (
                    contrast.get("red"),
                    contrast.get("green"),
                    contrast.get("blue"),
                )
            elif profile_version >= 2:
                raise ScanProfileError(f"Class {class_name} has invalid RGB contrast.")
            else:
                # Version 1 stored the absolute region color. Recalculate its
                # contrast from the source image and the current flake mask.
                contrast = None

            loaded_classes.append(self._make_class(
                class_name,
                source_path,
                image,
                region_mask,
                seed_point,
                threshold,
                connectivity,
                class_id=saved_class.get("id"),
                contrast=contrast,
                group=saved_class.get("group", ""),
                identify=identify,
                derived_from=saved_class.get("derived_from"),
            ))

        class_names = [profile_class["name"].casefold() for profile_class in loaded_classes]
        if len(class_names) != len(set(class_names)):
            raise ScanProfileError("Class names must be unique.")
        filter_ids = [item["id"] for item in loaded_filters]
        filter_names = [item["name"].casefold() for item in loaded_filters]
        if len(filter_ids) != len(set(filter_ids)):
            raise ScanProfileError("Filter IDs must be unique.")
        if len(filter_names) != len(set(filter_names)):
            raise ScanProfileError("Filter names must be unique.")

        self.name = name.strip()
        self.path = profile_dir
        self.created_at = str(payload.get("created_at", ""))
        self.updated_at = str(payload.get("updated_at", ""))
        self.classes = loaded_classes
        self.filters = loaded_filters
        return self

    def _make_class(
        self,
        name,
        source_path,
        image,
        region_mask,
        seed_point,
        threshold,
        connectivity,
        class_id=None,
        contrast=None,
        *,
        group="",
        identify=True,
        derived_from=None,
    ):
        if class_id is not None and (
            not isinstance(class_id, str) or not class_id.strip()
        ):
            raise ScanProfileError("Class ID must be non-empty text.")
        group = self._validate_group(group)
        identify = self._validate_flag(identify, "Identify")
        derived_from = self._validate_derived_from(derived_from)
        profile_class = {
            "id": class_id or uuid4().hex,
            "name": name.strip(),
            "classification_type": CLASSIFICATION_SAMPLE,
            "source_path": Path(source_path),
            "image_bgr": np.ascontiguousarray(image),
            "region_mask": np.ascontiguousarray(region_mask),
            "seed_point": (int(seed_point[0]), int(seed_point[1])),
            "threshold": int(threshold),
            "connectivity": int(connectivity),
            "contrast_rgb": contrast,
            "minimum_contrast_rgb": None,
            "maximum_contrast_rgb": None,
            "group": group,
            "identify": identify,
            "derived_from": derived_from,
        }
        self._validate_class(profile_class)
        if contrast is None:
            try:
                profile_class["contrast_rgb"] = region_contrast_rgb(image, region_mask)
            except ValueError as exc:
                raise ScanProfileError(
                    f"Could not determine the background for class {name}."
                ) from exc
        elif (
            not isinstance(contrast, (list, tuple))
            or len(contrast) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not -255 <= value <= 255
                for value in contrast
            )
        ):
            raise ScanProfileError(f"Class {name} has invalid RGB contrast.")
        else:
            profile_class["contrast_rgb"] = tuple(
                int(round(value)) for value in contrast
            )
        return profile_class

    def _make_range_class(
        self,
        name,
        minimum_contrast_rgb,
        maximum_contrast_rgb,
        class_id=None,
        *,
        group="",
        identify=True,
        derived_from=None,
    ):
        if class_id is not None and (
            not isinstance(class_id, str) or not class_id.strip()
        ):
            raise ScanProfileError("Class ID must be non-empty text.")
        minimum_rgb = self._validate_contrast(
            minimum_contrast_rgb,
            f"Class {name} minimum",
        )
        maximum_rgb = self._validate_contrast(
            maximum_contrast_rgb,
            f"Class {name} maximum",
        )
        if any(low > high for low, high in zip(minimum_rgb, maximum_rgb)):
            raise ScanProfileError(
                f"Class {name} contrast minimum cannot exceed its maximum."
            )
        profile_class = {
            "id": class_id or uuid4().hex,
            "name": name.strip(),
            "classification_type": CLASSIFICATION_RANGE,
            "source_path": None,
            "image_bgr": None,
            "region_mask": None,
            "seed_point": None,
            "threshold": None,
            "connectivity": None,
            "contrast_rgb": None,
            "minimum_contrast_rgb": minimum_rgb,
            "maximum_contrast_rgb": maximum_rgb,
            "group": self._validate_group(group),
            "identify": self._validate_flag(identify, "Identify"),
            "derived_from": self._validate_derived_from(derived_from),
        }
        self._validate_class(profile_class)
        return profile_class

    def _copy_class_record(self, source, copied_name, origin):
        if source["classification_type"] == CLASSIFICATION_RANGE:
            return self._make_range_class(
                copied_name,
                source["minimum_contrast_rgb"],
                source["maximum_contrast_rgb"],
                group=source["group"],
                identify=source["identify"],
                derived_from=origin,
            )
        return self._make_class(
            copied_name,
            source["source_path"],
            source["image_bgr"].copy(),
            source["region_mask"].copy(),
            source["seed_point"],
            source["threshold"],
            source["connectivity"],
            contrast=source["contrast_rgb"],
            group=source["group"],
            identify=source["identify"],
            derived_from=origin,
        )

    def _make_filter(
        self,
        name,
        filter_type,
        enabled,
        filter_id=None,
        **bounds,
    ):
        if not isinstance(name, str) or not name.strip():
            raise ScanProfileError("Every filter must have a name.")
        if filter_type not in FILTER_TYPES:
            raise ScanProfileError(f"Unknown filter type: {filter_type!r}.")
        enabled = self._validate_flag(enabled, "Filter enabled")
        if filter_id is not None and (
            not isinstance(filter_id, str) or not filter_id.strip()
        ):
            raise ScanProfileError("Filter ID must be non-empty text.")
        expected_keys = FILTER_BOUND_KEYS[filter_type]
        unknown_keys = set(bounds) - set(expected_keys)
        if unknown_keys:
            names = ", ".join(sorted(unknown_keys))
            raise ScanProfileError(
                f"Filter {name!r} has unsupported setting(s): {names}."
            )
        result = {
            "id": filter_id or uuid4().hex,
            "name": name.strip(),
            "type": filter_type,
            "enabled": enabled,
        }
        for key in expected_keys:
            value = bounds.get(key)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)
                or value < 0
            ):
                raise ScanProfileError(
                    f"Filter {name!r} has an invalid {key} value."
                )
            result[key] = None if value is None else float(value)
        for minimum_key, maximum_key in self._filter_bound_pairs(filter_type):
            minimum = result[minimum_key]
            maximum = result[maximum_key]
            if minimum is not None and maximum is not None and minimum > maximum:
                raise ScanProfileError(
                    f"Filter {name!r} minimum cannot exceed its maximum."
                )
        if enabled and expected_keys and all(result[key] is None for key in expected_keys):
            raise ScanProfileError(
                f"Enabled filter {name!r} needs at least one limit."
            )
        return result

    def _validate_filter(self, profile_filter):
        if not isinstance(profile_filter, dict):
            raise ScanProfileError("Every filter must be an object.")
        filter_id = profile_filter.get("id")
        if not isinstance(filter_id, str) or not filter_id.strip():
            raise ScanProfileError("Every saved filter must have a non-empty ID.")
        filter_type = profile_filter.get("type")
        if filter_type not in FILTER_TYPES:
            raise ScanProfileError(f"Unknown filter type: {filter_type!r}.")
        allowed_keys = {
            "id",
            "name",
            "type",
            "enabled",
            *FILTER_BOUND_KEYS[filter_type],
        }
        unknown_keys = set(profile_filter) - allowed_keys
        if unknown_keys:
            names = ", ".join(sorted(unknown_keys))
            raise ScanProfileError(f"Filter has unsupported setting(s): {names}.")
        self._make_filter(
            profile_filter.get("name"),
            filter_type,
            profile_filter.get("enabled"),
            filter_id=filter_id,
            **{
                key: profile_filter.get(key)
                for key in FILTER_BOUND_KEYS[filter_type]
            },
        )

    @staticmethod
    def _filter_bound_pairs(filter_type):
        if filter_type == FILTER_TYPE_SIZE:
            return (("minimum_um", "maximum_um"),)
        if filter_type == FILTER_TYPE_BOUNDING_BOX:
            return (
                ("minimum_width_um", "maximum_width_um"),
                ("minimum_height_um", "maximum_height_um"),
            )
        if filter_type == FILTER_TYPE_PERIMETER_AREA_RATIO:
            return (("minimum_per_um", "maximum_per_um"),)
        return ()

    def _validate_class(self, profile_class):
        name = profile_class["name"]
        if not name:
            raise ScanProfileError("Every class must have a name.")
        classification_type = profile_class.get(
            "classification_type",
            CLASSIFICATION_SAMPLE,
        )
        if classification_type not in CLASSIFICATION_TYPES:
            raise ScanProfileError(f"Class {name!r} has an invalid classification type.")
        self._validate_group(profile_class.get("group", ""))
        self._validate_flag(profile_class.get("identify"), "Identify")
        self._validate_derived_from(profile_class.get("derived_from"))
        if classification_type == CLASSIFICATION_RANGE:
            minimum_rgb = self._validate_contrast(
                profile_class.get("minimum_contrast_rgb"),
                f"Class {name!r} minimum",
            )
            maximum_rgb = self._validate_contrast(
                profile_class.get("maximum_contrast_rgb"),
                f"Class {name!r} maximum",
            )
            if any(low > high for low, high in zip(minimum_rgb, maximum_rgb)):
                raise ScanProfileError(
                    f"Class {name!r} contrast minimum cannot exceed its maximum."
                )
            return

        image = profile_class["image_bgr"]
        mask = profile_class["region_mask"]
        if image.ndim != 3 or image.shape[2] != 3:
            raise ScanProfileError(f"Class {name!r} has an invalid source image.")
        if mask.shape != image.shape[:2] or not np.any(mask > 0):
            raise ScanProfileError(f"Class {name!r} has an invalid region mask.")
        if profile_class["connectivity"] not in (4, 8):
            raise ScanProfileError("Flood-fill connectivity must be 4 or 8.")
        if not 0 <= profile_class["threshold"] <= 255:
            raise ScanProfileError("Flood-fill threshold must be between 0 and 255.")

        seed_x, seed_y = profile_class["seed_point"]
        height, width = image.shape[:2]
        if not (0 <= seed_x < width and 0 <= seed_y < height):
            raise ScanProfileError(f"Class {name!r} has an invalid seed point.")
        if mask[seed_y, seed_x] == 0:
            raise ScanProfileError(f"Class {name!r} does not contain its seed point.")

    @staticmethod
    def _validate_contrast(contrast, label):
        if (
            not isinstance(contrast, (list, tuple, np.ndarray))
            or len(contrast) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)
                or not -255 <= value <= 255
                for value in contrast
            )
        ):
            raise ScanProfileError(f"{label} has invalid RGB contrast.")
        return tuple(int(round(float(value))) for value in contrast)

    def _read_contrast_range(self, value, class_name):
        if not isinstance(value, dict):
            raise ScanProfileError(
                f"Class {class_name} has an invalid RGB contrast range."
            )
        result = []
        for bound_name in ("minimum", "maximum"):
            bound = value.get(bound_name)
            if not isinstance(bound, dict):
                raise ScanProfileError(
                    f"Class {class_name} has an invalid RGB contrast range."
                )
            result.append(self._validate_contrast(
                (bound.get("red"), bound.get("green"), bound.get("blue")),
                f"Class {class_name} {bound_name}",
            ))
        minimum_rgb, maximum_rgb = result
        if any(low > high for low, high in zip(minimum_rgb, maximum_rgb)):
            raise ScanProfileError(
                f"Class {class_name} contrast minimum cannot exceed its maximum."
            )
        return minimum_rgb, maximum_rgb

    def _region_mask(self, image, seed_point, threshold, connectivity, item_name):
        try:
            _, _, region_mask, contour = get_region_from_point(
                image,
                seed_point,
                threshold,
                connectivity,
            )
        except (cv2.error, ValueError) as exc:
            raise ScanProfileError(
                f"Could not reconstruct the reference region for {item_name}."
            ) from exc
        if contour is None or not np.any(region_mask > 0):
            raise ScanProfileError(
                f"Could not reconstruct the reference region for {item_name}."
            )
        return region_mask

    def _read_legacy_size(self, size):
        if size is None:
            return None, None
        if not isinstance(size, dict):
            raise ScanProfileError("The profile size requirement must be an object or null.")
        result = []
        for key in ("minimum_size_um", "maximum_size_um"):
            value = size.get(key)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value < 0
            ):
                raise ScanProfileError(f"The profile has an invalid {key} value.")
            result.append(None if value is None else float(value))
        minimum, maximum = result
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ScanProfileError("Legacy minimum size cannot exceed maximum size.")
        return minimum, maximum

    def _unique_class_name(self, preferred_name):
        preferred_name = str(preferred_name).strip() or "Class"
        if self.find_class(preferred_name) is None:
            return preferred_name
        suffix = 2
        while self.find_class(f"{preferred_name} {suffix}") is not None:
            suffix += 1
        return f"{preferred_name} {suffix}"

    def _unique_filter_name(self, preferred_name):
        preferred_name = str(preferred_name).strip() or "Filter"
        existing = {item["name"].casefold() for item in self.filters}
        if preferred_name.casefold() not in existing:
            return preferred_name
        suffix = 2
        while f"{preferred_name} {suffix}".casefold() in existing:
            suffix += 1
        return f"{preferred_name} {suffix}"

    @staticmethod
    def _validate_group(group):
        if group is None:
            return ""
        if not isinstance(group, str):
            raise ScanProfileError("Class group must be text.")
        return group.strip()

    @staticmethod
    def _validate_flag(value, label):
        if not isinstance(value, bool):
            raise ScanProfileError(f"{label} must be selected or cleared.")
        return value

    @staticmethod
    def _validate_derived_from(value):
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ScanProfileError("Class copy metadata must be an object or null.")
        profile = value.get("profile")
        class_id = value.get("class_id")
        if profile is not None and not isinstance(profile, str):
            raise ScanProfileError("Copied class profile must be text or null.")
        if class_id is not None and not isinstance(class_id, str):
            raise ScanProfileError("Copied class ID must be text or null.")
        result = {"profile": profile}
        if class_id is not None:
            result["class_id"] = class_id
        return result

    def _asset_path(self, profile_dir, relative_path, item_label):
        label = (
            f"Class {item_label}"
            if isinstance(item_label, int)
            else str(item_label).capitalize()
        )
        if not isinstance(relative_path, str) or not relative_path:
            raise ScanProfileError(f"{label} has a missing asset path.")
        relative_path = Path(relative_path)
        if relative_path.is_absolute():
            raise ScanProfileError(f"{label} asset paths must be relative.")
        path = (profile_dir / relative_path).resolve()
        if profile_dir not in path.parents:
            raise ScanProfileError(f"{label} asset path leaves the profile folder.")
        if not path.is_file():
            raise ScanProfileError(f"{label} asset was not found: {relative_path}")
        return path
