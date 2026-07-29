from datetime import datetime, timezone
import json
from pathlib import Path
import re
import unicodedata
from uuid import uuid4

import cv2
import numpy as np

from config import HOME_DIR
from Imaging import image_metadata
from .contour_finder import region_contrast_rgb
from .contour_extractor import get_region_from_point


PROFILE_SCHEMA = "flake-search.scan-search-profile"
PROFILE_VERSION = 5
SUPPORTED_PROFILE_VERSIONS = {1, 2, 3, 4, PROFILE_VERSION}
PROFILE_FILENAME = "profile.json"
DEFAULT_PROFILES_DIR = HOME_DIR / "Profiles"
ScanProfileError = ValueError

def _profile_slug(name):
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-") or "profile"
    reserved = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    return f"profile_{slug}" if slug.upper() in reserved else slug


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
        self.minimum_size_um = None
        self.maximum_size_um = None
        self.classes = []

    def profile_directory(self, name):
        return self.profiles_dir / _profile_slug(name)

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
        minimum_size_um=None,
        maximum_size_um=None,
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
            minimum_size_um,
            maximum_size_um,
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

    def set_size_requirement(self, minimum=None, maximum=None):
        minimum, maximum = self.validate_size_requirement(minimum, maximum)
        self.minimum_size_um = minimum
        self.maximum_size_um = maximum
        return minimum, maximum

    def copy_class(self, index, name=None, *, source_profile_name=None):
        """Copy a class into this profile with a new identity and unique label."""
        source = self.get_class(index)
        copied_name = self._unique_class_name(name or f"{source['name']} copy")
        origin = {
            "profile": source_profile_name or self.name or None,
            "class_id": source["id"],
        }
        copied = self._make_class(
            copied_name,
            source["source_path"],
            source["image_bgr"].copy(),
            source["region_mask"].copy(),
            source["seed_point"],
            source["threshold"],
            source["connectivity"],
            source["minimum_size_um"],
            source["maximum_size_um"],
            contrast=source["contrast_rgb"],
            group=source["group"],
            identify=source["identify"],
            derived_from=origin,
        )
        self.classes.append(copied)
        return copied

    def extend_from_profile(self, source_profile):
        """Append independent copies of every class in another profile."""
        added = []
        for source in source_profile.classes:
            copied_name = self._unique_class_name(source["name"])
            origin = {
                "profile": source_profile.name or None,
                "class_id": source["id"],
            }
            copied = self._make_class(
                copied_name,
                source["source_path"],
                source["image_bgr"].copy(),
                source["region_mask"].copy(),
                source["seed_point"],
                source["threshold"],
                source["connectivity"],
                source["minimum_size_um"],
                source["maximum_size_um"],
                contrast=source["contrast_rgb"],
                group=source["group"],
                identify=source["identify"],
                derived_from=origin,
            )
            self.classes.append(copied)
            added.append(copied)
        return added

    def matching_configuration(self, contrast_threshold=None):
        """Build classifier input directly from the editable in-memory profile."""
        classes = []
        for class_index, profile_class in enumerate(self.classes):
            tolerance = (
                profile_class["threshold"]
                if contrast_threshold is None
                else contrast_threshold
            )
            classes.append({
                "name": profile_class["name"],
                "contrast_rgb": np.asarray(profile_class["contrast_rgb"], dtype=np.float64),
                "tolerance": float(tolerance),
                "class_index": class_index,
                "group": profile_class["group"],
                "identify": profile_class["identify"],
                "minimum_size_um": profile_class["minimum_size_um"],
                "maximum_size_um": profile_class["maximum_size_um"],
            })
        return {
            "classes": classes,
            "minimum_size_um": self.minimum_size_um,
            "maximum_size_um": self.maximum_size_um,
        }

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

        profile_minimum, profile_maximum = self.validate_size_requirement(
            self.minimum_size_um,
            self.maximum_size_um,
        )

        destination = self.profile_directory(name)
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)

        images_dir = destination / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        saved_classes = []

        for profile_class in self.classes:
            class_id = profile_class["id"]
            filename = f"{_profile_slug(profile_class['name'])}_{class_id[:8]}_source.png"
            source_path = images_dir / filename
            _write_image(source_path, profile_class["image_bgr"])
            minimum, maximum = self.validate_size_requirement(
                profile_class["minimum_size_um"],
                profile_class["maximum_size_um"],
            )
            red, green, blue = profile_class["contrast_rgb"]
            saved_classes.append({
                "id": class_id,
                "name": profile_class["name"],
                "source_image": f"images/{filename}",
                "seed_point": {
                    "x": profile_class["seed_point"][0],
                    "y": profile_class["seed_point"][1],
                },
                "flood_fill": {
                    "threshold": profile_class["threshold"],
                    "connectivity": profile_class["connectivity"],
                },
                "size_requirement": (
                    {"minimum_size_um": minimum, "maximum_size_um": maximum}
                    if minimum is not None or maximum is not None
                    else None
                ),
                "contrast_rgb": {"red": red, "green": green, "blue": blue},
                "group": profile_class["group"] or None,
                "identify": profile_class["identify"],
            })
            if profile_class["derived_from"] is not None:
                saved_classes[-1]["derived_from"] = profile_class["derived_from"]

        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "schema": PROFILE_SCHEMA,
            "version": PROFILE_VERSION,
            "name": name,
            "created_at": now,
            "updated_at": now,
            "size_requirement": (
                {
                    "minimum_size_um": profile_minimum,
                    "maximum_size_um": profile_maximum,
                }
                if profile_minimum is not None or profile_maximum is not None
                else None
            ),
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
        if profile_version not in SUPPORTED_PROFILE_VERSIONS:
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
        profile_size = self._read_size(payload.get("size_requirement"))
        loaded_classes = []

        for index, saved_class in enumerate(saved_classes, start=1):
            if not isinstance(saved_class, dict):
                raise ScanProfileError(f"Class {index} must be a JSON object.")

            class_name = saved_class.get("name")
            seed = saved_class.get("seed_point")
            flood_fill = saved_class.get("flood_fill")
            if not isinstance(class_name, str) or not class_name.strip():
                raise ScanProfileError(f"Class {index} has no name.")
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

            size = (
                self._read_size(saved_class.get("size_requirement"))
                if "size_requirement" in saved_class
                else (profile_size if profile_version < 3 else (None, None))
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

            identify = saved_class.get("identify", True)
            loaded_classes.append(self._make_class(
                class_name,
                source_path,
                image,
                region_mask,
                seed_point,
                threshold,
                connectivity,
                size[0],
                size[1],
                saved_class.get("id"),
                contrast,
                group=saved_class.get("group", ""),
                identify=identify,
                derived_from=saved_class.get("derived_from"),
            ))

        class_names = [profile_class["name"].casefold() for profile_class in loaded_classes]
        if len(class_names) != len(set(class_names)):
            raise ScanProfileError("Class names must be unique.")

        self.name = name.strip()
        self.path = profile_dir
        self.created_at = str(payload.get("created_at", ""))
        self.updated_at = str(payload.get("updated_at", ""))
        self.minimum_size_um, self.maximum_size_um = profile_size
        self.classes = loaded_classes
        return self

    def validate_size_requirement(self, minimum, maximum):
        for label, value in (("Minimum size", minimum), ("Maximum size", maximum)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise ScanProfileError(f"{label} must be a number of micrometers.")
            if value is not None and value <= 0:
                raise ScanProfileError(f"{label} must be greater than zero.")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ScanProfileError("Minimum size cannot be greater than maximum size.")
        return (
            None if minimum is None else float(minimum),
            None if maximum is None else float(maximum),
        )

    def _make_class(
        self,
        name,
        source_path,
        image,
        region_mask,
        seed_point,
        threshold,
        connectivity,
        minimum,
        maximum,
        class_id=None,
        contrast=None,
        *,
        group="",
        identify=True,
        derived_from=None,
    ):
        minimum, maximum = self.validate_size_requirement(minimum, maximum)
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
            "source_path": Path(source_path),
            "image_bgr": np.ascontiguousarray(image),
            "region_mask": np.ascontiguousarray(region_mask),
            "seed_point": (int(seed_point[0]), int(seed_point[1])),
            "threshold": int(threshold),
            "connectivity": int(connectivity),
            "minimum_size_um": minimum,
            "maximum_size_um": maximum,
            "contrast_rgb": contrast,
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

    def _validate_class(self, profile_class):
        name = profile_class["name"]
        image = profile_class["image_bgr"]
        mask = profile_class["region_mask"]
        if not name:
            raise ScanProfileError("Every class must have a name.")
        self._validate_group(profile_class.get("group", ""))
        self._validate_flag(profile_class.get("identify"), "Identify")
        self._validate_derived_from(profile_class.get("derived_from"))
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

    def _read_size(self, size):
        if size is None:
            return None, None
        if not isinstance(size, dict):
            raise ScanProfileError("The profile size requirement must be an object or null.")
        return self.validate_size_requirement(
            size.get("minimum_size_um"),
            size.get("maximum_size_um"),
        )

    def _unique_class_name(self, preferred_name):
        preferred_name = str(preferred_name).strip() or "Class"
        if self.find_class(preferred_name) is None:
            return preferred_name
        suffix = 2
        while self.find_class(f"{preferred_name} {suffix}") is not None:
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
