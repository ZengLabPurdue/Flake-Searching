"""Flood-fill and color-match regions produced by contour detection."""

import colorsys
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_CONTRAST_MATCH_THRESHOLD = 3
DEFAULT_REGION_FLOOD_FILL_THRESHOLD = 10
PROFILE_SCHEMA = "flake-search.scan-search-profile"
PROFILE_VERSION = 7
SUPPORTED_PROFILE_VERSIONS = frozenset(range(1, PROFILE_VERSION + 1))
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
CLASS_COLOR_SATURATION = 0.58
CLASS_COLOR_LIGHTNESS = 0.72
LEGEND_TOP_LEFT = "top_left"
LEGEND_TOP_RIGHT = "top_right"
LEGEND_BOTTOM_LEFT = "bottom_left"
LEGEND_BOTTOM_RIGHT = "bottom_right"
LEGEND_POSITIONS = {
    LEGEND_TOP_LEFT,
    LEGEND_TOP_RIGHT,
    LEGEND_BOTTOM_LEFT,
    LEGEND_BOTTOM_RIGHT,
}
DEFAULT_LEGEND_POSITION = LEGEND_TOP_LEFT


def _timed_call(stats, name, function, *args):
    if stats is None:
        return function(*args)

    started_at = perf_counter()
    try:
        return function(*args)
    finally:
        elapsed = perf_counter() - started_at
        result = stats.setdefault(
            name,
            {"calls": 0, "total_seconds": 0.0, "max_seconds": 0.0},
        )
        result["calls"] += 1
        result["total_seconds"] += elapsed
        result["max_seconds"] = max(result["max_seconds"], elapsed)


def _read_rgb_triplet(value, field_name, class_name):
    if isinstance(value, dict):
        value = (value.get("red"), value.get("green"), value.get("blue"))
    if (
        not isinstance(value, (list, tuple, np.ndarray))
        or len(value) != 3
        or any(
            isinstance(channel, bool)
            or not isinstance(channel, (int, float))
            or not np.isfinite(channel)
            for channel in value
        )
    ):
        raise ValueError(f"Profile class {class_name!r} has invalid {field_name}.")
    return np.asarray(value, dtype=np.float64)


def _read_legacy_size_requirement(value):
    if value is None:
        return None, None
    if not isinstance(value, dict):
        raise ValueError("Profile size requirement must be an object or null.")
    result = []
    for key in ("minimum_size_um", "maximum_size_um"):
        size = value.get(key)
        if size is not None and (
            isinstance(size, bool)
            or not isinstance(size, (int, float))
            or not np.isfinite(size)
            or size <= 0
        ):
            raise ValueError(f"Profile has an invalid {key} value.")
        result.append(None if size is None else float(size))
    minimum, maximum = result
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("Profile minimum size cannot exceed its maximum size.")
    return minimum, maximum


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


def normalize_profile_filters(filters):
    """Validate and return canonical copies of profile-wide filters."""
    if filters is None:
        return []
    if not isinstance(filters, list):
        raise ValueError("Profile filters must be an array.")
    normalized = []
    seen_ids = set()
    seen_names = set()
    for index, profile_filter in enumerate(filters, start=1):
        if not isinstance(profile_filter, dict):
            raise ValueError(f"Profile filter {index} must be an object.")
        filter_id = profile_filter.get("id")
        name = profile_filter.get("name")
        filter_type = profile_filter.get("type")
        enabled = profile_filter.get("enabled")
        if not isinstance(filter_id, str) or not filter_id.strip():
            raise ValueError(f"Profile filter {index} has an invalid ID.")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Profile filter {index} has an invalid name.")
        if filter_type not in FILTER_TYPES:
            raise ValueError(f"Profile filter {name!r} has an invalid type.")
        if not isinstance(enabled, bool):
            raise ValueError(f"Profile filter {name!r} has an invalid enabled flag.")
        allowed_keys = {
            "id",
            "name",
            "type",
            "enabled",
            *FILTER_BOUND_KEYS[filter_type],
        }
        unknown_keys = set(profile_filter) - allowed_keys
        if unknown_keys:
            labels = ", ".join(sorted(unknown_keys))
            raise ValueError(
                f"Profile filter {name!r} has unsupported setting(s): {labels}."
            )
        if filter_id in seen_ids:
            raise ValueError("Profile filter IDs must be unique.")
        folded_name = name.strip().casefold()
        if folded_name in seen_names:
            raise ValueError("Profile filter names must be unique.")
        seen_ids.add(filter_id)
        seen_names.add(folded_name)
        result = {
            "id": filter_id,
            "name": name.strip(),
            "type": filter_type,
            "enabled": enabled,
        }
        bound_keys = FILTER_BOUND_KEYS[filter_type]
        for key in bound_keys:
            value = profile_filter.get(key)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"Profile filter {name!r} has an invalid {key} value."
                )
            result[key] = None if value is None else float(value)
        for minimum_key, maximum_key in _filter_bound_pairs(filter_type):
            minimum = result[minimum_key]
            maximum = result[maximum_key]
            if minimum is not None and maximum is not None and minimum > maximum:
                raise ValueError(
                    f"Profile filter {name!r} minimum cannot exceed its maximum."
                )
        if enabled and bound_keys and all(result[key] is None for key in bound_keys):
            raise ValueError(
                f"Enabled profile filter {name!r} needs at least one limit."
            )
        normalized.append(result)
    return normalized


def load_profile_configuration(profile_path, contrast_threshold=None):
    """Load color classes and v7 profile-wide geometry filters."""
    profile_path = Path(profile_path)
    if profile_path.is_dir():
        profile_path = profile_path / "profile.json"
    if contrast_threshold is not None and (
        isinstance(contrast_threshold, bool)
        or not isinstance(contrast_threshold, (int, float))
        or not np.isfinite(contrast_threshold)
        or not 0 <= contrast_threshold <= 255
    ):
        raise ValueError("The contrast threshold must be between 0 and 255.")

    try:
        profile_data = json.loads(profile_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Profile was not found: {profile_path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read profile: {profile_path}") from exc

    if (
        not isinstance(profile_data, dict)
        or profile_data.get("schema") != PROFILE_SCHEMA
    ):
        raise ValueError(f"File is not a scan search profile: {profile_path}")
    profile_version = profile_data.get("version")
    if (
        type(profile_version) is not int
        or profile_version not in SUPPORTED_PROFILE_VERSIONS
    ):
        raise ValueError(f"Unsupported profile version: {profile_version!r}.")
    saved_classes = profile_data.get("classes")
    if not isinstance(saved_classes, list):
        raise ValueError(f"Profile has an invalid class list: {profile_path}")
    if profile_version == PROFILE_VERSION:
        profile_filters = normalize_profile_filters(profile_data.get("filters", []))
    else:
        # Do not interpret the unrelated generic ``filters`` payload used by
        # some version-4 profiles. Only the old top-level size range migrates.
        legacy_minimum, legacy_maximum = _read_legacy_size_requirement(
            profile_data.get("size_requirement")
        )
        profile_filters = []
        if legacy_minimum is not None or legacy_maximum is not None:
            profile_filters.append({
                "id": "legacy-profile-size",
                "name": "Legacy size",
                "type": FILTER_TYPE_SIZE,
                "enabled": True,
                "minimum_um": legacy_minimum,
                "maximum_um": legacy_maximum,
            })
    profile_classes = []
    for saved_class in saved_classes:
        if not isinstance(saved_class, dict):
            raise ValueError(f"Profile class has invalid data: {profile_path}")
        name = saved_class.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Profile class has an invalid name: {profile_path}")
        name = name.strip()
        classification_type = saved_class.get(
            "classification_type",
            CLASSIFICATION_SAMPLE,
        )
        if classification_type not in CLASSIFICATION_TYPES:
            raise ValueError(
                f"Profile class {name!r} has an invalid classification type."
            )
        color_configuration = {"classification_type": classification_type}
        if classification_type == CLASSIFICATION_RANGE:
            try:
                saved_range = saved_class["contrast_range_rgb"]
                minimum_rgb = _read_rgb_triplet(
                    saved_range["minimum"],
                    "minimum RGB contrast",
                    name,
                )
                maximum_rgb = _read_rgb_triplet(
                    saved_range["maximum"],
                    "maximum RGB contrast",
                    name,
                )
            except KeyError as exc:
                raise ValueError(
                    f"Profile class {name!r} is missing RGB contrast range data."
                ) from exc
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Profile class {name!r} has invalid RGB contrast range data."
                ) from exc
            if (
                np.any(minimum_rgb < -255)
                or np.any(maximum_rgb > 255)
                or np.any(minimum_rgb > maximum_rgb)
            ):
                raise ValueError(
                    f"Profile class {name!r} has an invalid RGB contrast range."
                )
            color_configuration.update({
                "minimum_contrast_rgb": minimum_rgb,
                "maximum_contrast_rgb": maximum_rgb,
            })
        else:
            try:
                contrast_rgb = _read_rgb_triplet(
                    saved_class["contrast_rgb"],
                    "RGB contrast",
                    name,
                )
                saved_threshold = saved_class.get("contrast_threshold")
                if saved_threshold is None:
                    saved_threshold = saved_class["flood_fill"]["threshold"]
                tolerance = (
                    float(saved_threshold)
                    if contrast_threshold is None
                    else float(contrast_threshold)
                )
            except KeyError as exc:
                raise ValueError(
                    f"Profile class {name!r} is missing RGB contrast data or a threshold."
                ) from exc
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Profile class {name!r} has invalid color data.") from exc

            if np.any(contrast_rgb < -255) or np.any(contrast_rgb > 255):
                raise ValueError(f"Profile class {name!r} has an invalid RGB contrast.")
            if not np.isfinite(tolerance) or not 0 <= tolerance <= 255:
                raise ValueError(f"Profile class {name!r} has an invalid tolerance.")
            color_configuration.update({
                "contrast_rgb": contrast_rgb,
                "tolerance": tolerance,
            })
        identify = saved_class.get("identify", True)
        if not isinstance(identify, bool):
            raise ValueError(f"Profile class {name!r} has invalid matching flags.")
        group = saved_class.get("group")
        if group is None:
            group = ""
        if not isinstance(group, str):
            raise ValueError(f"Profile class {name!r} has an invalid group.")
        configured_class = {
            "name": name,
            "class_index": len(profile_classes),
            "group": group.strip(),
            "identify": identify,
        }
        configured_class.update(color_configuration)
        profile_classes.append(configured_class)

    names = [item["name"].casefold() for item in profile_classes]
    if len(names) != len(set(names)):
        raise ValueError("Profile class names must be unique.")

    return {"classes": profile_classes, "filters": profile_filters}


def load_profile_classes(profile_path, contrast_threshold=None):
    """Load class matching data while retaining the former public API."""
    return load_profile_configuration(profile_path, contrast_threshold)["classes"]


def match_profile_class(contrast_rgb, profile_classes):
    """Return the nearest enabled sample or inclusive contrast-range class."""
    matches = []
    for profile_class in profile_classes:
        if not profile_class.get("identify", True):
            continue
        classification_type = profile_class.get(
            "classification_type",
            CLASSIFICATION_SAMPLE,
        )
        if classification_type == CLASSIFICATION_RANGE:
            minimum_rgb = profile_class["minimum_contrast_rgb"]
            maximum_rgb = profile_class["maximum_contrast_rgb"]
            if np.all(contrast_rgb >= minimum_rgb) and np.all(
                contrast_rgb <= maximum_rgb
            ):
                center = (minimum_rgb + maximum_rgb) / 2.0
                difference = np.abs(contrast_rgb - center)
                matches.append((
                    float(np.linalg.norm(difference)),
                    profile_class,
                    difference,
                ))
        else:
            difference = np.abs(contrast_rgb - profile_class["contrast_rgb"])
            if np.all(difference <= profile_class["tolerance"]):
                matches.append((
                    float(np.linalg.norm(difference)),
                    profile_class,
                    difference,
                ))
    return min(matches, key=lambda match: match[0]) if matches else None


def generate_class_colors(profile_classes, color_seed=None):
    """Generate randomly assigned soft class colors with well-spaced hues."""
    if not profile_classes:
        return {}
    random = np.random.default_rng(color_seed)
    hue_offset = float(random.random())
    hue_step = 1.0 / len(profile_classes)
    hues = np.asarray([
        (hue_offset + index * hue_step) % 1.0
        for index in range(len(profile_classes))
    ])
    random.shuffle(hues)
    colors = {}
    for index, profile_class in enumerate(profile_classes):
        hue = float(hues[index])
        rgb = colorsys.hls_to_rgb(
            hue,
            CLASS_COLOR_LIGHTNESS,
            CLASS_COLOR_SATURATION,
        )
        colors[profile_class["name"]] = tuple(
            int(round(channel * 255)) for channel in rgb
        )
    return colors


def contour_mask(image_shape, contours):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    if contours:
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    return mask


def _contour_depths(hierarchy):
    depths = np.zeros(len(hierarchy), dtype=np.int32)
    for index in range(len(hierarchy)):
        parent = hierarchy[index][3]
        while parent != -1:
            depths[index] += 1
            parent = hierarchy[parent][3]
    return depths


def _deepest_contour_at_point(point, contour_indices, contours, depths):
    containing = [
        index
        for index in contour_indices
        if cv2.pointPolygonTest(contours[index], point, False) >= 0
    ]
    return max(containing, key=lambda index: depths[index]) if containing else contour_indices[0]


def _flood_fill_region(image_rgb, seed_point, allowed_mask, threshold):
    """Run fixed-range, 8-connected RGB flood fill without changing the source."""
    if not allowed_mask[seed_point[1], seed_point[0]]:
        return np.zeros(image_rgb.shape[:2], dtype=bool)
    flood_mask = np.ones(
        (image_rgb.shape[0] + 2, image_rgb.shape[1] + 2),
        dtype=np.uint8,
    )
    flood_mask[1:-1, 1:-1] = np.where(allowed_mask, 0, 1).astype(np.uint8)
    flags = 8 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    color_tolerance = (threshold, threshold, threshold)
    cv2.floodFill(
        image_rgb,
        flood_mask,
        seed_point,
        (0, 0, 0),
        color_tolerance,
        color_tolerance,
        flags,
    )
    return flood_mask[1:-1, 1:-1] == 255


def _validate_classification_inputs(pixel_size_um, profile_filters):
    if pixel_size_um is not None and (
        isinstance(pixel_size_um, bool)
        or not isinstance(pixel_size_um, (int, float))
        or not np.isfinite(pixel_size_um)
        or pixel_size_um <= 0
    ):
        raise ValueError("Pixel size must be a positive number of micrometers.")
    filters = normalize_profile_filters(profile_filters)
    for profile_filter in filters:
        if not profile_filter["enabled"]:
            continue
        if profile_filter["type"] == FILTER_TYPE_NEURAL_NETWORK:
            raise ValueError(
                "The neural network flake filter is not implemented yet."
            )
        if pixel_size_um is None:
            raise ValueError(
                "A calibrated pixel size is required by enabled profile filters."
            )
    return filters


def _prepare_classification_workspace(
    image_rgb,
    edge_mask,
    contours,
    all_external_contours,
    external_indices,
):
    height, width = edge_mask.shape
    external_contours = [contours[index] for index in external_indices]
    external_mask = contour_mask(image_rgb.shape, external_contours)
    all_external_mask = contour_mask(image_rgb.shape, all_external_contours)
    background_pixels = image_rgb[all_external_mask == 0]
    if len(background_pixels) == 0:
        raise ValueError("The detected contours cover the entire image background.")
    background_color = np.rint(np.mean(background_pixels, axis=0))
    return {
        "height": height,
        "width": width,
        "external_mask": external_mask,
        "all_external_mask": all_external_mask,
        "background_color": background_color,
        "all_region_mask": np.zeros((height, width), dtype=np.uint8),
        "region_overlap_count_map": np.zeros((height, width), dtype=np.uint16),
        "all_internal_mask": np.zeros((height, width), dtype=np.uint8),
        "matched_region_mask": np.zeros((height, width), dtype=np.uint8),
        "matched_internal_mask": np.zeros((height, width), dtype=np.uint8),
        "class_masks": {},
        "region_results": [],
    }


def _connected_components(allowed):
    return cv2.connectedComponentsWithStats(allowed, connectivity=8)


def _describe_seed_components(
    component_count,
    labels,
    component_stats,
    centroids,
    x1,
    y1,
    contour_indices,
    contours,
    depths,
    benchmark_stats,
):
    components = []
    for component_label in range(1, component_count):
        component_size = int(component_stats[component_label, cv2.CC_STAT_AREA])
        if component_size == 0:
            continue
        seed_x, seed_y = (
            int(round(value)) for value in centroids[component_label]
        )
        if (
            not 0 <= seed_x < labels.shape[1]
            or not 0 <= seed_y < labels.shape[0]
            or labels[seed_y, seed_x] != component_label
        ):
            left = int(component_stats[component_label, cv2.CC_STAT_LEFT])
            top = int(component_stats[component_label, cv2.CC_STAT_TOP])
            component_width = int(
                component_stats[component_label, cv2.CC_STAT_WIDTH]
            )
            component_height = int(
                component_stats[component_label, cv2.CC_STAT_HEIGHT]
            )
            relative_y, relative_x = np.argwhere(
                labels[
                    top:top + component_height,
                    left:left + component_width,
                ] == component_label
            )[0]
            seed_x = left + int(relative_x)
            seed_y = top + int(relative_y)
        global_seed = (seed_x + x1, seed_y + y1)
        source_index = _timed_call(
            benchmark_stats,
            "deepest_contour_at_point",
            _deepest_contour_at_point,
            global_seed,
            contour_indices,
            contours,
            depths,
        )
        components.append({
            "component_label": component_label,
            "pixel_count": component_size,
            "global_seed": global_seed,
            "source_contour_index": source_index,
            "raw_depth": int(depths[source_index]),
            "bounding_box": {
                "x": int(x1 + component_stats[component_label, cv2.CC_STAT_LEFT]),
                "y": int(y1 + component_stats[component_label, cv2.CC_STAT_TOP]),
                "width": int(component_stats[component_label, cv2.CC_STAT_WIDTH]),
                "height": int(component_stats[component_label, cv2.CC_STAT_HEIGHT]),
            },
        })
    return components


def _extract_external_components(
    external_index,
    contours,
    edge_mask,
    contour_indices_by_external,
    depths,
    width,
    height,
    benchmark_stats,
):
    external_contour = contours[external_index]
    x, y, contour_width, contour_height = cv2.boundingRect(external_contour)
    x1, y1 = max(0, x - 1), max(0, y - 1)
    x2 = min(width, x + contour_width + 1)
    y2 = min(height, y + contour_height + 1)

    local_contour = external_contour.copy()
    local_contour[:, 0, 0] -= x1
    local_contour[:, 0, 1] -= y1
    local_external_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cv2.drawContours(local_external_mask, [local_contour], -1, 255, cv2.FILLED)
    allowed = (
        (local_external_mask > 0)
        & (edge_mask[y1:y2, x1:x2] == 0)
    ).astype(np.uint8)
    component_count, labels, component_stats, centroids = _timed_call(
        benchmark_stats,
        "connected_components",
        _connected_components,
        allowed,
    )
    components = _timed_call(
        benchmark_stats,
        "describe_seed_components",
        _describe_seed_components,
        component_count,
        labels,
        component_stats,
        centroids,
        x1,
        y1,
        contour_indices_by_external[external_index],
        contours,
        depths,
        benchmark_stats,
    )
    if not components:
        return None
    outer_depth = min(component["raw_depth"] for component in components)
    components.sort(
        key=lambda component: (
            component["raw_depth"],
            component["pixel_count"],
        ),
        reverse=True,
    )
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "local_external_mask": local_external_mask,
        "component_count": component_count,
        "component_labels": labels,
        "components": components,
        "outer_depth": outer_depth,
    }


def _prepare_flood_inputs(image_rgb, external_region, local_seed):
    return (
        np.ascontiguousarray(image_rgb[
            external_region["y1"]:external_region["y2"],
            external_region["x1"]:external_region["x2"],
        ]),
        external_region["local_external_mask"] > 0,
        local_seed,
    )


def _measure_contour_colors(
    image_rgb,
    component_labels,
    component_count,
    x1,
    y1,
    x2,
    y2,
    background_color,
):
    """Measure every connected contour interior in one exact label reduction."""
    flat_labels = component_labels.reshape(-1)
    flat_colors = np.ascontiguousarray(
        image_rgb[y1:y2, x1:x2]
    ).reshape(-1, 3)
    pixel_counts = np.bincount(flat_labels, minlength=component_count).astype(
        np.float64,
    )
    color_sums = np.column_stack([
        np.bincount(
            flat_labels,
            weights=flat_colors[:, channel],
            minlength=component_count,
        )
        for channel in range(3)
    ])
    average_colors = np.divide(
        color_sums,
        pixel_counts[:, np.newaxis],
        out=np.zeros_like(color_sums),
        where=pixel_counts[:, np.newaxis] > 0,
    )
    return {
        "average_colors": average_colors,
        "contrasts": np.rint(average_colors - background_color),
        "average_intensities": np.mean(average_colors, axis=1),
    }


def _component_color_measurement(color_measurements, component_label):
    return {
        "average_color": color_measurements["average_colors"][component_label],
        "contrast": color_measurements["contrasts"][component_label],
        "average_intensity": float(
            color_measurements["average_intensities"][component_label]
        ),
    }


def _measure_contour_geometry(
    component,
    pixel_size_um,
):
    bounding_box = component["bounding_box"]
    pixel_count = component["pixel_count"]
    size_um = (
        max(bounding_box["width"], bounding_box["height"]) * pixel_size_um
        if pixel_size_um is not None
        else None
    )
    return {
        "pixel_count": pixel_count,
        "bounding_box": bounding_box,
        "size_um": size_um,
        "width_um": (
            bounding_box["width"] * pixel_size_um
            if pixel_size_um is not None
            else None
        ),
        "height_um": (
            bounding_box["height"] * pixel_size_um
            if pixel_size_um is not None
            else None
        ),
        "area_um2": (
            pixel_count * pixel_size_um * pixel_size_um
            if pixel_size_um is not None
            else None
        ),
    }


def _measure_region_geometry(
    region_mask,
    x1,
    y1,
    pixel_size_um,
):
    pixel_count = int(np.count_nonzero(region_mask))
    if pixel_count == 0:
        return None
    region_y, region_x = np.nonzero(region_mask)
    bounding_box = {
        "x": int(x1 + region_x.min()),
        "y": int(y1 + region_y.min()),
        "width": int(region_x.max() - region_x.min() + 1),
        "height": int(region_y.max() - region_y.min() + 1),
    }
    size_um = (
        max(bounding_box["width"], bounding_box["height"]) * pixel_size_um
        if pixel_size_um is not None
        else None
    )
    return {
        "pixel_count": pixel_count,
        "bounding_box": bounding_box,
        "size_um": size_um,
        "width_um": (
            bounding_box["width"] * pixel_size_um
            if pixel_size_um is not None
            else None
        ),
        "height_um": (
            bounding_box["height"] * pixel_size_um
            if pixel_size_um is not None
            else None
        ),
        "area_um2": (
            pixel_count * pixel_size_um * pixel_size_um
            if pixel_size_um is not None
            else None
        ),
    }


def _match_region(
    color_measurement,
    geometry,
    profile_classes,
    benchmark_stats,
):
    return (
        _timed_call(
            benchmark_stats,
            "match_profile_class",
            match_profile_class,
            color_measurement["contrast"],
            profile_classes,
        )
        if profile_classes
        else None
    )


def _update_region_masks(
    workspace,
    region_mask,
    x1,
    y1,
    x2,
    y2,
    region_type,
    matched_class,
):
    workspace["all_region_mask"][y1:y2, x1:x2][region_mask] = 255
    workspace["region_overlap_count_map"][y1:y2, x1:x2][region_mask] += 1
    if region_type == "internal":
        workspace["all_internal_mask"][y1:y2, x1:x2][region_mask] = 255
    if not matched_class:
        return
    workspace["matched_region_mask"][y1:y2, x1:x2][region_mask] = 255
    if region_type == "internal":
        workspace["matched_internal_mask"][y1:y2, x1:x2][region_mask] = 255
    class_mask = workspace["class_masks"].get(matched_class["name"])
    if class_mask is None:
        class_mask = np.zeros(
            (workspace["height"], workspace["width"]),
            dtype=np.uint8,
        )
        workspace["class_masks"][matched_class["name"]] = class_mask
    class_mask[y1:y2, x1:x2][region_mask] = 255


def _make_region_result(
    component,
    region_type,
    nesting_depth,
    color_measurement,
    geometry,
    matched_class,
    match,
    flood_filled,
    region_mask,
    region_origin,
):
    average_color = color_measurement["average_color"]
    contrast = color_measurement["contrast"]
    size_um = geometry["size_um"]
    area_um2 = geometry["area_um2"]
    instance_mask = None
    if matched_class is not None and region_mask is not None:
        bounding_box = geometry["bounding_box"]
        mask_width = int(bounding_box["width"])
        mask_height = int(bounding_box["height"])
        local_x = int(bounding_box["x"]) - int(region_origin[0])
        local_y = int(bounding_box["y"]) - int(region_origin[1])
        cropped_mask = region_mask[
            local_y:local_y + mask_height,
            local_x:local_x + mask_width,
        ]
        if cropped_mask.shape == (mask_height, mask_width):
            instance_mask = np.where(cropped_mask, 255, 0).astype(np.uint8)

    return {
        "seed_point": component["global_seed"],
        "region_type": region_type,
        "source_contour_index": int(component["source_contour_index"]),
        "nesting_depth": nesting_depth,
        "seed_component_pixel_count": component["pixel_count"],
        "classification_source": "contour_interior",
        "flood_filled": flood_filled,
        "average_color_rgb": tuple(
            int(round(channel)) for channel in average_color
        ),
        "average_intensity": color_measurement["average_intensity"],
        "contrast_rgb": tuple(int(round(channel)) for channel in contrast),
        "matched_class": matched_class["name"] if matched_class else None,
        "matched_group": (
            (matched_class.get("group") or None) if matched_class else None
        ),
        "matched_classification_type": (
            matched_class.get("classification_type", CLASSIFICATION_SAMPLE)
            if matched_class
            else None
        ),
        "size_um": float(size_um) if size_um is not None else None,
        "width_um": (
            float(geometry["width_um"])
            if geometry["width_um"] is not None
            else None
        ),
        "height_um": (
            float(geometry["height_um"])
            if geometry["height_um"] is not None
            else None
        ),
        "area_um2": float(area_um2) if area_um2 is not None else None,
        "matched_color_rgb": (
            matched_class["display_color_rgb"] if matched_class else None
        ),
        "match_threshold": (
            matched_class.get("tolerance") if matched_class else None
        ),
        "match_distance": match[0] if match else None,
        "channel_difference_rgb": (
            tuple(float(value) for value in match[2])
            if match and match[2] is not None
            else None
        ),
        "pixel_count": geometry["pixel_count"],
        "bounding_box": geometry["bounding_box"],
        "instance_mask": instance_mask,
    }


def _build_class_index_maps(height, width, profile_classes, class_masks):
    class_index_map = np.zeros((height, width), dtype=np.int32)
    class_overlap_count_map = np.zeros((height, width), dtype=np.uint16)
    for profile_class in profile_classes:
        class_mask = class_masks.get(profile_class["name"])
        if class_mask is None:
            continue
        selected = class_mask > 0
        class_overlap_count_map[selected] += 1
        class_index_map[selected] = profile_class["class_index"] + 1
    return class_index_map, class_overlap_count_map


def classify_contour_regions(
    image_rgb,
    edge_mask,
    contours,
    hierarchy,
    all_external_contours,
    external_indices,
    contour_indices_by_external,
    profile_classes,
    flood_fill_threshold,
    benchmark_stats=None,
    pixel_size_um=None,
    profile_filters=None,
):
    """Classify contour interiors first, then flood-fill identified regions."""
    _timed_call(
        benchmark_stats,
        "validate_classification_inputs",
        _validate_classification_inputs,
        pixel_size_um,
        profile_filters,
    )
    workspace = _timed_call(
        benchmark_stats,
        "prepare_classification_workspace",
        _prepare_classification_workspace,
        image_rgb,
        edge_mask,
        contours,
        all_external_contours,
        external_indices,
    )
    depths = _timed_call(
        benchmark_stats,
        "contour_depths",
        _contour_depths,
        hierarchy,
    )
    for external_index in external_indices:
        external_region = _timed_call(
            benchmark_stats,
            "extract_external_components",
            _extract_external_components,
            external_index,
            contours,
            edge_mask,
            contour_indices_by_external,
            depths,
            workspace["width"],
            workspace["height"],
            benchmark_stats,
        )
        if external_region is None:
            continue
        x1 = external_region["x1"]
        y1 = external_region["y1"]
        x2 = external_region["x2"]
        y2 = external_region["y2"]
        color_measurements = _timed_call(
            benchmark_stats,
            "measure_contour_colors",
            _measure_contour_colors,
            image_rgb,
            external_region["component_labels"],
            external_region["component_count"],
            x1,
            y1,
            x2,
            y2,
            workspace["background_color"],
        )
        for component in external_region["components"]:
            global_seed = component["global_seed"]
            local_seed = (global_seed[0] - x1, global_seed[1] - y1)
            color_measurement = _component_color_measurement(
                color_measurements,
                component["component_label"],
            )
            contour_geometry = _timed_call(
                benchmark_stats,
                "measure_contour_geometry",
                _measure_contour_geometry,
                component,
                pixel_size_um,
            )
            depth_difference = (
                component["raw_depth"] - external_region["outer_depth"]
            )
            nesting_depth = max(0, (depth_difference + 1) // 2)
            region_type = "external" if nesting_depth == 0 else "internal"
            match = _timed_call(
                benchmark_stats,
                "match_region",
                _match_region,
                color_measurement,
                contour_geometry,
                profile_classes,
                benchmark_stats,
            )
            matched_class = match[1] if match else None
            region_mask = None
            result_geometry = contour_geometry
            flood_filled = False
            if matched_class is not None:
                color_region, allowed_mask, local_seed = _timed_call(
                    benchmark_stats,
                    "prepare_flood_inputs",
                    _prepare_flood_inputs,
                    image_rgb,
                    external_region,
                    local_seed,
                )
                region_mask = _timed_call(
                    benchmark_stats,
                    "flood_fill_region",
                    _flood_fill_region,
                    color_region,
                    local_seed,
                    allowed_mask,
                    flood_fill_threshold,
                )
                flooded_geometry = _timed_call(
                    benchmark_stats,
                    "measure_flooded_region_geometry",
                    _measure_region_geometry,
                    region_mask,
                    x1,
                    y1,
                    pixel_size_um,
                )
                if flooded_geometry is not None:
                    result_geometry = flooded_geometry
                    flood_filled = True
            if region_mask is not None:
                _timed_call(
                    benchmark_stats,
                    "update_region_masks",
                    _update_region_masks,
                    workspace,
                    region_mask,
                    x1,
                    y1,
                    x2,
                    y2,
                    region_type,
                    matched_class,
                )
            workspace["region_results"].append(_timed_call(
                benchmark_stats,
                "build_region_result",
                _make_region_result,
                component,
                region_type,
                nesting_depth,
                color_measurement,
                result_geometry,
                matched_class,
                match,
                flood_filled,
                region_mask,
                (x1, y1),
            ))

    class_index_map, class_overlap_count_map = _timed_call(
        benchmark_stats,
        "build_class_index_maps",
        _build_class_index_maps,
        workspace["height"],
        workspace["width"],
        profile_classes,
        workspace["class_masks"],
    )

    return {
        "external_mask": workspace["external_mask"],
        "all_external_mask": workspace["all_external_mask"],
        "all_region_mask": workspace["all_region_mask"],
        "region_overlap_count_map": workspace["region_overlap_count_map"],
        "all_internal_mask": workspace["all_internal_mask"],
        "matched_region_mask": workspace["matched_region_mask"],
        "matched_internal_mask": workspace["matched_internal_mask"],
        "class_index_map": class_index_map,
        "class_masks": workspace["class_masks"],
        "class_overlap_count_map": class_overlap_count_map,
        "background_color_rgb": tuple(
            int(channel) for channel in workspace["background_color"]
        ),
        "region_results": workspace["region_results"],
    }


def create_classified_image(image_shape, external_mask, class_index_map, profile_classes):
    """Create a black-background RGB class map with white unmatched flakes."""
    image = np.zeros(image_shape, dtype=np.uint8)
    image[external_mask > 0] = (255, 255, 255)
    for profile_class in profile_classes:
        if not profile_class.get("identify", True):
            continue
        image[class_index_map == profile_class["class_index"] + 1] = profile_class[
            "display_color_rgb"
        ]
    return image


def _component_perimeter_px(instance_mask):
    contours, _hierarchy = cv2.findContours(
        instance_mask.copy(),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE,
    )
    return float(sum(cv2.arcLength(contour, True) for contour in contours))


def _inside_inclusive_range(value, minimum, maximum):
    if value is None:
        return False
    if minimum is not None and value < minimum:
        return False
    if maximum is not None and value > maximum:
        return False
    return True


def _region_passes_filters(region, profile_filters):
    filter_results = []
    passed_all = True
    for profile_filter in profile_filters:
        enabled = profile_filter["enabled"]
        filter_type = profile_filter["type"]
        passed = None
        if enabled:
            if filter_type == FILTER_TYPE_NEURAL_NETWORK:
                raise ValueError(
                    "The neural network flake filter is not implemented yet."
                )
            if filter_type == FILTER_TYPE_SIZE:
                passed = _inside_inclusive_range(
                    region["area_um2"],
                    profile_filter["minimum_um"],
                    profile_filter["maximum_um"],
                )
            elif filter_type == FILTER_TYPE_BOUNDING_BOX:
                passed = (
                    _inside_inclusive_range(
                        region["width_um"],
                        profile_filter["minimum_width_um"],
                        profile_filter["maximum_width_um"],
                    )
                    and _inside_inclusive_range(
                        region["height_um"],
                        profile_filter["minimum_height_um"],
                        profile_filter["maximum_height_um"],
                    )
                )
            elif filter_type == FILTER_TYPE_PERIMETER_AREA_RATIO:
                passed = _inside_inclusive_range(
                    region["perimeter_area_ratio_per_um"],
                    profile_filter["minimum_per_um"],
                    profile_filter["maximum_per_um"],
                )
            passed_all = passed_all and bool(passed)
        filter_results.append({
            "id": profile_filter["id"],
            "name": profile_filter["name"],
            "type": filter_type,
            "enabled": enabled,
            "passed": passed,
        })
    return passed_all, filter_results


def filter_combined_class_regions(
    class_index_map,
    profile_classes,
    pixel_size_um=None,
    profile_filters=None,
):
    """Measure, filter, and rebuild final connected class components."""
    filters = _validate_classification_inputs(pixel_size_um, profile_filters)
    accepted_regions = []
    rejected_regions = []
    all_regions = []
    filtered_class_index_map = np.zeros_like(class_index_map)
    for profile_class in profile_classes:
        if not profile_class.get("identify", True):
            continue
        class_value = int(profile_class["class_index"]) + 1
        class_mask = np.where(
            class_index_map == class_value,
            255,
            0,
        ).astype(np.uint8)
        component_count, component_labels, stats, _centroids = (
            cv2.connectedComponentsWithStats(
                class_mask,
                connectivity=8,
            )
        )
        for component_label in range(1, component_count):
            x = int(stats[component_label, cv2.CC_STAT_LEFT])
            y = int(stats[component_label, cv2.CC_STAT_TOP])
            width = int(stats[component_label, cv2.CC_STAT_WIDTH])
            height = int(stats[component_label, cv2.CC_STAT_HEIGHT])
            pixel_count = int(stats[component_label, cv2.CC_STAT_AREA])
            instance_mask = np.where(
                component_labels[y:y + height, x:x + width] == component_label,
                255,
                0,
            ).astype(np.uint8)
            perimeter_px = _component_perimeter_px(instance_mask)
            width_um = width * pixel_size_um if pixel_size_um is not None else None
            height_um = height * pixel_size_um if pixel_size_um is not None else None
            area_um2 = (
                pixel_count * pixel_size_um * pixel_size_um
                if pixel_size_um is not None
                else None
            )
            perimeter_um = (
                perimeter_px * pixel_size_um
                if pixel_size_um is not None
                else None
            )
            region = {
                "region_type": "combined",
                "matched_class": profile_class["name"],
                "matched_group": profile_class.get("group") or None,
                "matched_classification_type": profile_class.get(
                    "classification_type",
                    CLASSIFICATION_SAMPLE,
                ),
                "matched_color_rgb": profile_class["display_color_rgb"],
                "pixel_count": pixel_count,
                "size_um": (
                    max(width_um, height_um)
                    if width_um is not None and height_um is not None
                    else None
                ),
                "area_um2": area_um2,
                "width_um": width_um,
                "height_um": height_um,
                "perimeter_px": perimeter_px,
                "perimeter_um": perimeter_um,
                "perimeter_area_ratio_per_um": (
                    perimeter_um / area_um2
                    if perimeter_um is not None and area_um2
                    else None
                ),
                "bounding_box": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                },
                "instance_mask": instance_mask,
            }
            accepted, filter_results = _region_passes_filters(region, filters)
            region["accepted"] = accepted
            region["filter_results"] = filter_results
            region["failed_filter_ids"] = [
                item["id"]
                for item in filter_results
                if item["enabled"] and not item["passed"]
            ]
            all_regions.append(region)
            if accepted:
                accepted_regions.append(region)
                local_component = (
                    component_labels[y:y + height, x:x + width]
                    == component_label
                )
                filtered_class_index_map[y:y + height, x:x + width][
                    local_component
                ] = class_value
            else:
                rejected_regions.append(region)
    return {
        "class_index_map": filtered_class_index_map,
        "accepted_regions": accepted_regions,
        "rejected_regions": rejected_regions,
        "all_regions": all_regions,
    }


def combined_class_regions(
    class_index_map,
    profile_classes,
    pixel_size_um=None,
    profile_filters=None,
):
    """Return accepted connected regions while retaining the former API."""
    return filter_combined_class_regions(
        class_index_map,
        profile_classes,
        pixel_size_um,
        profile_filters,
    )["accepted_regions"]


def _load_class_legend_font(font_size):
    """Load a crisp Git Bash-style monospace font with portable fallbacks."""
    font_candidates = (
        Path("C:/Windows/Fonts/lucon.ttf"),
        Path("C:/Windows/Fonts/consola.ttf"),
    )
    for font_path in font_candidates:
        if font_path.is_file():
            return ImageFont.truetype(str(font_path), font_size)
    for font_name in ("DejaVuSansMono.ttf", "LiberationMono-Regular.ttf"):
        try:
            return ImageFont.truetype(font_name, font_size)
        except OSError:
            continue
    return ImageFont.load_default(size=font_size)


def draw_class_legend(
    image_rgb,
    profile_classes,
    region_results=None,
    position=DEFAULT_LEGEND_POSITION,
):
    """Draw a compact linked legend in any corner of the classified image."""
    if not isinstance(position, str) or position not in LEGEND_POSITIONS:
        choices = ", ".join(sorted(LEGEND_POSITIONS))
        raise ValueError(f"Legend position must be one of: {choices}.")
    display_classes = [
        item
        for item in profile_classes
        if item.get("identify", True)
    ]
    if not display_classes:
        return image_rgb.copy()

    result = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(result)
    scale = max(1.0, min(image_rgb.shape[:2]) / 700.0)
    font_size = max(14, int(round(18 * scale)))
    font = _load_class_legend_font(font_size)
    line_height = max(16, int(round(font_size * 1.15)))
    line_width = max(1, int(round(scale / 2)))
    horizontal_margin = max(10, int(round(14 * scale)))
    vertical_margin = max(8, int(round(12 * scale)))
    line_gap = max(4, int(round(5 * scale)))
    on_right = position in (LEGEND_TOP_RIGHT, LEGEND_BOTTOM_RIGHT)
    on_bottom = position in (LEGEND_BOTTOM_LEFT, LEGEND_BOTTOM_RIGHT)

    maximum_lines = max(
        0,
        (image_rgb.shape[0] - 2 * vertical_margin) // line_height,
    )
    display_classes = display_classes[:maximum_lines]
    if not display_classes:
        return image_rgb.copy()
    block_height = len(display_classes) * line_height
    start_y = (
        image_rgb.shape[0] - vertical_margin - block_height
        if on_bottom
        else vertical_margin
    )

    regions_by_class = {}
    for region in region_results or ():
        class_name = region.get("matched_class")
        bounding_box = region.get("bounding_box")
        if class_name is None or not bounding_box:
            continue
        center = (
            int(round(bounding_box["x"] + bounding_box["width"] / 2)),
            int(round(bounding_box["y"] + bounding_box["height"] / 2)),
        )
        regions_by_class.setdefault(class_name, []).append(center)

    label_layout = []
    for index, profile_class in enumerate(display_classes):
        text_y = start_y + index * line_height
        class_name = profile_class["name"]
        color = tuple(int(channel) for channel in profile_class["display_color_rgb"])
        display_name = (
            f"{profile_class['group']} / {class_name}"
            if profile_class.get("group")
            else class_name
        ).upper()
        unpositioned_bounds = draw.textbbox((0, 0), display_name, font=font)
        text_width = unpositioned_bounds[2] - unpositioned_bounds[0]
        text_x = (
            max(horizontal_margin, image_rgb.shape[1] - horizontal_margin - text_width)
            if on_right
            else horizontal_margin
        )
        text_bounds = draw.textbbox((text_x, text_y), display_name, font=font)
        line_start = (
            text_bounds[0] - line_gap if on_right else text_bounds[2] + line_gap,
            (text_bounds[1] + text_bounds[3]) // 2,
        )
        label_layout.append(
            (class_name, display_name, color, (text_x, text_y), line_start)
        )

    for class_name, _, color, _, line_start in label_layout:
        for region_center in regions_by_class.get(class_name, ()):
            draw.line(
                (line_start, region_center),
                fill=color,
                width=line_width,
            )

    for _, display_name, color, text_position, _ in label_layout:
        draw.text(text_position, display_name, font=font, fill=color)

    return np.asarray(result).copy()


def draw_region_measurement_labels(
    image_rgb,
    region_results,
    *,
    show_area=True,
    show_perimeter_area_ratio=False,
):
    """Draw selected physical measurements beside each matched region."""
    if not show_area and not show_perimeter_area_ratio:
        return image_rgb.copy()
    result = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(result)
    scale = max(1.0, min(image_rgb.shape[:2]) / 700.0)
    font_size = max(7, int(round(8 * scale)))
    font = _load_class_legend_font(font_size)
    line_spacing = max(1, int(round(scale)))
    margin = max(3, int(round(4 * scale)))
    image_height, image_width = image_rgb.shape[:2]
    occupied = []
    placements = []

    def clamp_position(x, y, text_width, text_height):
        return (
            max(margin, min(int(round(x)), image_width - margin - text_width)),
            max(margin, min(int(round(y)), image_height - margin - text_height)),
        )

    def overlaps(rectangle):
        left, top, right, bottom = rectangle
        return any(
            not (
                right + margin <= other_left
                or other_right + margin <= left
                or bottom + margin <= other_top
                or other_bottom + margin <= top
            )
            for other_left, other_top, other_right, other_bottom in occupied
        )

    for region in region_results or ():
        area_um2 = region.get("area_um2")
        ratio_per_um = region.get("perimeter_area_ratio_per_um")
        bounding_box = region.get("bounding_box")
        color = region.get("matched_color_rgb")
        label_parts = []
        if show_area and area_um2 is not None:
            label_parts.append(f"Size {area_um2:,.1f}")
        if show_perimeter_area_ratio and ratio_per_um is not None:
            label_parts.append(f"P:A {ratio_per_um:,.4f}")
        if (
            region.get("matched_class") is None
            or not label_parts
            or not bounding_box
            or color is None
        ):
            continue

        label = "\n".join(label_parts)
        color = tuple(int(channel) for channel in color)
        bounds = draw.multiline_textbbox(
            (0, 0),
            label,
            font=font,
            spacing=line_spacing,
        )
        text_width = bounds[2] - bounds[0]
        text_height = bounds[3] - bounds[1]
        region_right = int(round(bounding_box["x"] + bounding_box["width"]))
        region_left = int(round(bounding_box["x"]))
        region_top = int(round(bounding_box["y"]))
        region_bottom = int(round(bounding_box["y"] + bounding_box["height"]))
        center_x = (region_left + region_right) / 2
        center_y = (region_top + region_bottom) / 2
        candidates = [
            (region_right + margin, region_top),
            (region_left - margin - text_width, region_top),
            (region_left, region_top - margin - text_height),
            (region_left, region_bottom + margin),
            (region_right + margin, region_bottom - text_height),
            (region_left - margin - text_width, region_bottom - text_height),
        ]
        step = max(text_height + 2 * margin, 1)
        maximum_radius = max(image_width, image_height)
        for radius in range(step, maximum_radius + step, step):
            candidates.extend([
                (center_x + radius, center_y - text_height / 2),
                (center_x - radius - text_width, center_y - text_height / 2),
                (center_x - text_width / 2, center_y - radius - text_height),
                (center_x - text_width / 2, center_y + radius),
                (center_x + radius, center_y + radius),
                (center_x - radius - text_width, center_y + radius),
                (center_x + radius, center_y - radius - text_height),
                (
                    center_x - radius - text_width,
                    center_y - radius - text_height,
                ),
            ])

        selected = None
        seen_positions = set()
        for candidate_x, candidate_y in candidates:
            text_x, text_y = clamp_position(
                candidate_x,
                candidate_y,
                text_width,
                text_height,
            )
            if (text_x, text_y) in seen_positions:
                continue
            seen_positions.add((text_x, text_y))
            rectangle = (
                text_x,
                text_y,
                text_x + text_width,
                text_y + text_height,
            )
            if not overlaps(rectangle):
                selected = (text_x, text_y, rectangle)
                break
        if selected is None:
            text_x, text_y = clamp_position(
                region_right + margin,
                region_top,
                text_width,
                text_height,
            )
            selected = (
                text_x,
                text_y,
                (text_x, text_y, text_x + text_width, text_y + text_height),
            )

        text_x, text_y, rectangle = selected
        occupied.append(rectangle)
        placements.append(
            (label, color, (text_x, text_y), rectangle, bounding_box)
        )

    for label, color, (text_x, text_y), _, _ in placements:
        draw.multiline_text(
            (text_x, text_y),
            label,
            font=font,
            fill=color,
            spacing=line_spacing,
        )

    return np.asarray(result).copy()


def draw_region_area_labels(image_rgb, region_results):
    """Draw each matched region's physical area beside its bounding box."""
    return draw_region_measurement_labels(
        image_rgb,
        region_results,
        show_area=True,
    )


def _seed_inside_contour(contour, allowed_mask):
    candidates = (contour_mask(allowed_mask.shape, [contour]) > 0) & allowed_mask
    ys, xs = np.nonzero(candidates)
    if len(xs) == 0:
        return None
    moments = cv2.moments(contour)
    if moments["m00"]:
        center = (
            int(moments["m10"] / moments["m00"]),
            int(moments["m01"] / moments["m00"]),
        )
        if (
            0 <= center[0] < allowed_mask.shape[1]
            and 0 <= center[1] < allowed_mask.shape[0]
            and candidates[center[1], center[0]]
        ):
            return center
    x, y, width, height = cv2.boundingRect(contour)
    center_x = x + width / 2
    center_y = y + height / 2
    closest = np.argmin((xs - center_x) ** 2 + (ys - center_y) ** 2)
    return int(xs[closest]), int(ys[closest])


def internal_region_mask(edge_mask, contour_groups):
    """Build the legacy geometric mask used by the production scanner."""
    height, width = edge_mask.shape
    result = np.zeros((height, width), dtype=np.uint8)
    processed = np.zeros((height, width), dtype=np.uint8)
    for external_contour, internal_contours in contour_groups:
        x, y, contour_width, contour_height = cv2.boundingRect(external_contour)
        x1, y1 = max(0, x - 1), max(0, y - 1)
        x2 = min(width, x + contour_width + 1)
        y2 = min(height, y + contour_height + 1)
        local_external = external_contour.copy()
        local_external[:, 0] -= (x1, y1)
        allowed = (
            (contour_mask((y2 - y1, x2 - x1), [local_external]) > 0)
            & (edge_mask[y1:y2, x1:x2] == 0)
            & (processed[y1:y2, x1:x2] == 0)
        )
        for internal_contour in internal_contours:
            local_contour = internal_contour.copy()
            local_contour[:, 0] -= (x1, y1)
            seed = _seed_inside_contour(local_contour, allowed)
            if seed is None:
                continue
            flood_image = np.zeros(allowed.shape, dtype=np.uint8)
            flood_mask = np.ones(
                (allowed.shape[0] + 2, allowed.shape[1] + 2),
                dtype=np.uint8,
            )
            flood_mask[1:-1, 1:-1] = np.where(allowed, 0, 1).astype(np.uint8)
            cv2.floodFill(flood_image, flood_mask, seed, 255, flags=4)
            region = flood_image == 255
            result[y1:y2, x1:x2][region] = 255
            processed[y1:y2, x1:x2][region] = 255
            allowed[region] = False
    return result


def background_color_rgb(image_bgr, flake_mask):
    """Return mean RGB background outside the supplied flake mask."""
    pixels = image_bgr[flake_mask == 0]
    if len(pixels) == 0:
        raise ValueError("The contour mask covers the entire image background.")
    blue, green, red = np.mean(pixels, axis=0)
    return int(round(red)), int(round(green)), int(round(blue))


def region_contrast_rgb(image_bgr, region_mask, flake_mask):
    """Return signed RGB region contrast relative to the supplied background."""
    if region_mask is None or region_mask.shape != image_bgr.shape[:2]:
        raise ValueError("The region mask does not match the image.")
    if not np.any(region_mask > 0):
        raise ValueError("The selected region is empty.")
    blue, green, red = np.mean(image_bgr[region_mask > 0], axis=0)
    background = background_color_rgb(image_bgr, flake_mask)
    return tuple(
        int(round(region_channel - background_channel))
        for region_channel, background_channel in zip((red, green, blue), background)
    )
