import json
from pathlib import Path
from queue import Queue
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np


APP_DIR = Path(__file__).resolve().parents[1] / "App"
sys.path.insert(0, str(APP_DIR))

from Scanning.region_classifier import (  # noqa: E402
    LEGEND_BOTTOM_LEFT,
    LEGEND_BOTTOM_RIGHT,
    LEGEND_TOP_LEFT,
    LEGEND_TOP_RIGHT,
    _flood_fill_region,
    classify_contour_regions,
    combined_class_regions,
    create_classified_image,
    draw_class_legend,
    draw_region_area_labels,
    draw_region_measurement_labels,
    filter_combined_class_regions,
    load_profile_configuration,
    match_profile_class,
)
from Scanning.contour_finder import (  # noqa: E402
    _filter_external_candidates,
    find_flakes,
)
from Scanning.contour_extractor import get_region_from_point  # noqa: E402
from Scanning.flake_detection import (  # noqa: E402
    Flake_Detector,
    encode_region_mask,
    save_database,
)
from Scanning.scan_profile import (  # noqa: E402
    CLASSIFICATION_RANGE,
    FILTER_TYPE_BOUNDING_BOX,
    FILTER_TYPE_NEURAL_NETWORK,
    FILTER_TYPE_PERIMETER_AREA_RATIO,
    FILTER_TYPE_SIZE,
    PROFILE_SCHEMA,
    PROFILE_VERSION,
    ScanProfile,
)


class ScanProfileOptionsTests(unittest.TestCase):
    def test_scan_and_preview_preserve_saved_class_tolerance(self):
        image = np.full((100, 100, 3), 180, dtype=np.uint8)
        image[30:70, 30:70] = 150
        with tempfile.TemporaryDirectory() as directory:
            profile = ScanProfile(directory)
            item = self._add_class(profile, "Flake", (-24, -24, -24))
            item["threshold"] = 10
            profile.save_profile("Tolerance")
            common = dict(return_details=True, draw_legend=False, color_seed=0)
            preview, _, preview_details = find_flakes(
                image, profile_configuration=profile.matching_configuration(), **common,
            )
            scanned, _, scan_details = find_flakes(image, profile_path=profile.path, **common)
            _, _, strict_details = find_flakes(
                image, profile_path=profile.path, contrast_threshold=3, **common,
            )
        self.assertGreater(len(preview_details["combined_region_results"]), 0)
        self.assertEqual(len(strict_details["combined_region_results"]), 0)
        self.assertEqual(
            len(scan_details["combined_region_results"]),
            len(preview_details["combined_region_results"]),
        )
        np.testing.assert_array_equal(scanned, preview)

    def _add_class(
        self,
        profile,
        name,
        contrast,
        *,
        group="",
        identify=True,
    ):
        image = np.full((8, 8, 3), 180, dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 255
        profile_class = profile._make_class(
            name,
            Path(f"{name}.png"),
            image,
            mask,
            (3, 3),
            5,
            8,
            contrast=contrast,
            group=group,
            identify=identify,
        )
        profile.classes.append(profile_class)
        return profile_class

    def test_current_version_round_trip_persists_classes_and_profile_options(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = ScanProfile(directory)
            profile.set_filter(
                None,
                "Useful sizes",
                FILTER_TYPE_SIZE,
                True,
                minimum_um=4,
                maximum_um=90,
            )
            profile.set_filter(
                None,
                "Shape model",
                FILTER_TYPE_NEURAL_NETWORK,
                False,
            )
            self._add_class(
                profile,
                "Thin blue",
                (-20, -10, 4),
                group="hBN",
                identify=False,
            )

            loaded = profile.save_profile("Options")
            payload = json.loads((loaded.path / "profile.json").read_text(encoding="utf-8"))

            self.assertEqual(payload["version"], PROFILE_VERSION)
            self.assertNotIn("size_requirement", payload)
            self.assertNotIn("size_requirement", payload["classes"][0])
            self.assertEqual(payload["filters"][0]["type"], FILTER_TYPE_SIZE)
            self.assertEqual(payload["filters"][0]["minimum_um"], 4.0)
            self.assertEqual(payload["filters"][1]["type"], FILTER_TYPE_NEURAL_NETWORK)
            self.assertEqual(loaded.groups, ["hBN"])
            self.assertFalse(loaded.classes[0]["identify"])
            self.assertEqual(len(loaded.filters), 2)

    def test_profile_folder_preserves_the_exact_valid_profile_name(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = ScanProfile(directory)
            self._add_class(profile, "Thin", (1, 2, 3))
            profile_name = "Gr_base (285nm - 4in)"

            loaded = profile.save_profile(profile_name)
            payload = json.loads(
                (loaded.path / "profile.json").read_text(encoding="utf-8")
            )

            self.assertEqual(loaded.path.name, profile_name)
            self.assertEqual(loaded.name, profile_name)
            self.assertEqual(payload["name"], profile_name)

    def test_profile_folder_rejects_unsafe_or_reserved_names(self):
        profile = ScanProfile()

        for invalid_name in ("../escape", "bad/name", "bad:name", "CON"):
            with self.subTest(name=invalid_name):
                with self.assertRaises(ValueError):
                    profile.profile_directory(invalid_name)

    def test_range_class_round_trip_needs_no_reference_image(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = ScanProfile(directory)
            created = profile.set_range_class(
                "Graphene range",
                (-40, -30, -20),
                (-10, 5, 15),
                group="Graphene",
            )

            loaded = profile.save_profile("Ranges")
            payload = json.loads(
                (loaded.path / "profile.json").read_text(encoding="utf-8")
            )
            saved_class = payload["classes"][0]

            self.assertEqual(created["classification_type"], CLASSIFICATION_RANGE)
            self.assertEqual(saved_class["classification_type"], CLASSIFICATION_RANGE)
            self.assertNotIn("source_image", saved_class)
            self.assertNotIn("flood_fill", saved_class)
            self.assertEqual(
                saved_class["contrast_range_rgb"],
                {
                    "minimum": {"red": -40, "green": -30, "blue": -20},
                    "maximum": {"red": -10, "green": 5, "blue": 15},
                },
            )
            self.assertFalse((loaded.path / "images").exists())
            self.assertEqual(
                loaded.classes[0]["minimum_contrast_rgb"],
                (-40, -30, -20),
            )
            self.assertIsNone(loaded.classes[0]["image_bgr"])

    def test_range_matching_is_inclusive_on_all_rgb_channels(self):
        profile = ScanProfile()
        profile.set_range_class(
            "Inclusive",
            (-20, 0, 10),
            (-5, 15, 30),
        )
        configured = profile.matching_configuration()["classes"]

        minimum_match = match_profile_class(
            np.asarray((-20, 0, 10), dtype=float),
            configured,
        )
        maximum_match = match_profile_class(
            np.asarray((-5, 15, 30), dtype=float),
            configured,
        )

        self.assertEqual(minimum_match[1]["name"], "Inclusive")
        self.assertEqual(maximum_match[1]["name"], "Inclusive")
        self.assertIsNone(match_profile_class(
            np.asarray((-21, 0, 10), dtype=float),
            configured,
        ))
        self.assertIsNone(match_profile_class(
            np.asarray((-10, 16, 20), dtype=float),
            configured,
        ))

    def test_range_classes_copy_and_keep_identity_when_converted(self):
        profile = ScanProfile()
        sampled = self._add_class(profile, "Convertible", (1, 2, 3))

        converted = profile.set_range_class(
            "Convertible",
            (-15, -10, -5),
            (5, 10, 15),
            index=0,
        )
        copied = profile.copy_class(0)

        self.assertEqual(converted["id"], sampled["id"])
        self.assertEqual(copied["classification_type"], CLASSIFICATION_RANGE)
        self.assertEqual(copied["minimum_contrast_rgb"], (-15, -10, -5))
        self.assertNotEqual(copied["id"], converted["id"])
        self.assertEqual(copied["derived_from"]["class_id"], converted["id"])

    def test_matching_loader_reads_range_classification(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "profile.json"
            profile_path.write_text(
                json.dumps({
                    "schema": PROFILE_SCHEMA,
                    "version": 6,
                    "classes": [{
                        "name": "Range",
                        "classification_type": "range",
                        "contrast_range_rgb": {
                            "minimum": {"red": -30, "green": -20, "blue": -10},
                            "maximum": {"red": 1, "green": 2, "blue": 3},
                        },
                        "identify": True,
                        "size_requirement": None,
                    }],
                }),
                encoding="utf-8",
            )

            configured = load_profile_configuration(
                profile_path,
                contrast_threshold=0,
            )["classes"][0]

            self.assertEqual(configured["classification_type"], CLASSIFICATION_RANGE)
            self.assertTrue(np.array_equal(
                configured["minimum_contrast_rgb"],
                (-30, -20, -10),
            ))
            self.assertNotIn("tolerance", configured)

    def test_version_two_profiles_default_to_identify(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = ScanProfile(directory)
            self._add_class(profile, "Legacy", (1, 2, 3))
            saved = profile.save_profile("Legacy")
            profile_path = saved.path / "profile.json"
            payload = json.loads(profile_path.read_text(encoding="utf-8"))
            payload["version"] = 2
            payload.pop("filters", None)
            for item in payload["classes"]:
                item.pop("group", None)
                item.pop("identify", None)
            profile_path.write_text(json.dumps(payload), encoding="utf-8")

            loaded = ScanProfile(directory).load_profile(profile_path)

            self.assertEqual(loaded.classes[0]["group"], "")
            self.assertTrue(loaded.classes[0]["identify"])

    def test_copy_and_extend_make_independent_classes_with_provenance(self):
        source = ScanProfile()
        source.name = "Source"
        original = self._add_class(source, "Class A", (5, 6, 7), group="Group A")

        copied = source.copy_class(0)
        copied["image_bgr"][0, 0] = 0

        self.assertEqual(copied["name"], "Class A copy")
        self.assertFalse(np.array_equal(copied["image_bgr"], original["image_bgr"]))
        self.assertEqual(copied["derived_from"]["class_id"], original["id"])

        destination = ScanProfile()
        self._add_class(destination, "Class A", (0, 0, 0))
        added = destination.extend_from_profile(source)

        self.assertEqual([item["name"] for item in added], ["Class A 2", "Class A copy"])
        self.assertTrue(all(item["derived_from"]["profile"] == "Source" for item in added))

    def test_saved_copy_provenance_points_to_the_stable_source_id(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = ScanProfile(directory)
            source = self._add_class(profile, "Source", (1, 2, 3))
            copied = profile.copy_class(0)

            saved = profile.save_profile("Copies")
            payload = json.loads((saved.path / "profile.json").read_text(encoding="utf-8"))

            self.assertEqual(payload["classes"][0]["id"], source["id"])
            self.assertEqual(payload["classes"][1]["id"], copied["id"])
            self.assertEqual(
                payload["classes"][1]["derived_from"]["class_id"],
                payload["classes"][0]["id"],
            )

    def test_disabled_classes_are_ignored(self):
        classes = [
            {
                "name": "Nearest identify",
                "contrast_rgb": np.asarray((10, 10, 10), dtype=float),
                "tolerance": 10,
                "identify": True,
            },
            {
                "name": "Disabled exact",
                "contrast_rgb": np.asarray((12, 12, 12), dtype=float),
                "tolerance": 10,
                "identify": False,
            },
        ]

        match = match_profile_class(np.asarray((12, 12, 12)), classes)

        self.assertEqual(match[1]["name"], "Nearest identify")

    def test_obsolete_class_size_fields_do_not_affect_color_matching(self):
        classes = [
            {
                "name": "Small",
                "contrast_rgb": np.asarray((1, 1, 1), dtype=float),
                "tolerance": 2,
                "identify": True,
                "minimum_size_um": 2,
                "maximum_size_um": 5,
            }
        ]

        self.assertEqual(
            match_profile_class(np.asarray((1, 1, 1)), classes)[1]["name"],
            "Small",
        )

    def test_matching_loader_reads_profile_size_and_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "profile.json"
            profile_path.write_text(
                json.dumps({
                    "schema": PROFILE_SCHEMA,
                    "version": 6,
                    "size_requirement": {
                        "minimum_size_um": 3,
                        "maximum_size_um": 30,
                    },
                    "classes": [{
                        "name": "Class",
                        "contrast_rgb": {"red": 1, "green": 2, "blue": 3},
                        "flood_fill": {"threshold": 4},
                        "group": "Artifacts",
                        "identify": True,
                        "size_requirement": None,
                    }],
                }),
                encoding="utf-8",
            )

            configuration = load_profile_configuration(profile_path)

            self.assertEqual(configuration["filters"][0]["type"], FILTER_TYPE_SIZE)
            self.assertEqual(configuration["filters"][0]["minimum_um"], 3.0)
            self.assertEqual(configuration["filters"][0]["maximum_um"], 30.0)
            self.assertEqual(configuration["classes"][0]["group"], "Artifacts")

    def test_size_filter_uses_final_connected_filled_area(self):
        class_index_map = np.zeros((12, 12), dtype=np.int32)
        class_index_map[2:8, 3:7] = 1
        profile_class = {
            "name": "Class",
            "class_index": 0,
            "group": "Material",
            "identify": True,
            "display_color_rgb": (255, 0, 0),
        }
        profile_filter = {
            "id": "size",
            "name": "Larger than this",
            "type": FILTER_TYPE_SIZE,
            "enabled": True,
            "minimum_um": 7,
            "maximum_um": None,
        }

        filtered = filter_combined_class_regions(
            class_index_map,
            [profile_class],
            pixel_size_um=1,
            profile_filters=[profile_filter],
        )

        self.assertEqual(len(filtered["accepted_regions"]), 1)
        self.assertEqual(filtered["accepted_regions"][0]["area_um2"], 24)
        self.assertEqual(filtered["accepted_regions"][0]["size_um"], 6)

        profile_filter["minimum_um"] = 25
        rejected = filter_combined_class_regions(
            class_index_map,
            [profile_class],
            pixel_size_um=1,
            profile_filters=[profile_filter],
        )
        self.assertEqual(rejected["accepted_regions"], [])
        self.assertEqual(len(rejected["rejected_regions"]), 1)
        self.assertFalse(np.any(rejected["class_index_map"]))

    def test_filter_crud_copy_and_extend_keep_independent_ids(self):
        source = ScanProfile()
        created = source.set_filter(
            None,
            "Size",
            FILTER_TYPE_SIZE,
            True,
            minimum_um=2,
            maximum_um=20,
        )
        updated = source.set_filter(
            created["id"],
            "Useful size",
            FILTER_TYPE_SIZE,
            True,
            minimum_um=3,
            maximum_um=30,
        )
        copied = source.copy_filter(updated["id"])

        self.assertEqual(source.get_filter(updated["id"])["minimum_um"], 3)
        self.assertNotEqual(copied["id"], updated["id"])
        self.assertEqual(copied["name"], "Useful size copy")
        source.move_filter(1, 0)
        self.assertEqual(source.filters[0]["id"], copied["id"])
        self.assertEqual(source.remove_filter(copied["id"])["id"], copied["id"])

        destination = ScanProfile()
        destination.extend_from_profile(source)
        self.assertEqual(len(destination.filters), 1)
        self.assertNotEqual(destination.filters[0]["id"], updated["id"])
        self.assertEqual(destination.filters[0]["minimum_um"], 3)

    def test_v4_generic_filters_are_not_treated_as_v7_geometry_filters(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "profile.json"
            profile_path.write_text(json.dumps({
                "schema": PROFILE_SCHEMA,
                "version": 4,
                "filters": [{
                    "id": "obsolete",
                    "name": "Wrong schema",
                    "type": FILTER_TYPE_SIZE,
                    "enabled": True,
                    "minimum_um": 999,
                }],
                "classes": [{
                    "name": "Class",
                    "contrast_rgb": {"red": 1, "green": 2, "blue": 3},
                    "flood_fill": {"threshold": 4},
                    "identify": True,
                }],
            }), encoding="utf-8")

            configuration = load_profile_configuration(profile_path)

        self.assertEqual(configuration["filters"], [])

    def test_loaders_reject_invalid_versions_and_missing_filter_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "invalid-version.json"
            for invalid_version in (PROFILE_VERSION + 1, True, float(PROFILE_VERSION), None):
                with self.subTest(version=invalid_version):
                    profile_path.write_text(json.dumps({
                        "schema": PROFILE_SCHEMA,
                        "version": invalid_version,
                        "classes": [],
                    }), encoding="utf-8")

                    with self.assertRaisesRegex(ValueError, "Unsupported profile version"):
                        load_profile_configuration(profile_path)
                    with self.assertRaisesRegex(ValueError, "Unsupported profile version"):
                        ScanProfile(directory).load_profile(profile_path)

            profile = ScanProfile(directory)
            self._add_class(profile, "Class", (1, 2, 3))
            profile.set_filter(
                None,
                "Size",
                FILTER_TYPE_SIZE,
                True,
                minimum_um=1,
                maximum_um=None,
            )
            saved = profile.save_profile("Missing ID")
            saved_path = saved.path / "profile.json"
            payload = json.loads(saved_path.read_text(encoding="utf-8"))
            payload["filters"][0].pop("id")
            saved_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "non-empty ID"):
                ScanProfile(directory).load_profile(saved_path)
            with self.assertRaisesRegex(ValueError, "invalid ID"):
                load_profile_configuration(saved_path)

    def test_all_enabled_geometry_filters_are_inclusive_and_anded(self):
        class_index_map = np.zeros((16, 18), dtype=np.int32)
        class_index_map[2:10, 3:13] = 1
        profile_class = {
            "name": "Class",
            "class_index": 0,
            "group": "Material",
            "identify": True,
            "display_color_rgb": (255, 0, 0),
        }
        unfiltered = filter_combined_class_regions(
            class_index_map,
            [profile_class],
            pixel_size_um=0.5,
        )["accepted_regions"][0]
        ratio = unfiltered["perimeter_area_ratio_per_um"]
        filters = [
            {
                "id": "size",
                "name": "Exact size",
                "type": FILTER_TYPE_SIZE,
                "enabled": True,
                "minimum_um": 20,
                "maximum_um": 20,
            },
            {
                "id": "box",
                "name": "Exact box",
                "type": FILTER_TYPE_BOUNDING_BOX,
                "enabled": True,
                "minimum_width_um": 5,
                "maximum_width_um": 5,
                "minimum_height_um": 4,
                "maximum_height_um": 4,
            },
            {
                "id": "shape",
                "name": "Exact shape",
                "type": FILTER_TYPE_PERIMETER_AREA_RATIO,
                "enabled": True,
                "minimum_per_um": ratio,
                "maximum_per_um": ratio,
            },
            {
                "id": "disabled",
                "name": "Disabled rejection",
                "type": FILTER_TYPE_SIZE,
                "enabled": False,
                "minimum_um": 100,
                "maximum_um": None,
            },
        ]

        accepted = filter_combined_class_regions(
            class_index_map,
            [profile_class],
            pixel_size_um=0.5,
            profile_filters=filters,
        )
        self.assertEqual(len(accepted["accepted_regions"]), 1)
        self.assertTrue(all(
            item["passed"] is not False
            for item in accepted["accepted_regions"][0]["filter_results"]
        ))

        filters[1]["minimum_width_um"] = 0
        filters[1]["maximum_width_um"] = 4.99
        rejected = filter_combined_class_regions(
            class_index_map,
            [profile_class],
            pixel_size_um=0.5,
            profile_filters=filters,
        )
        self.assertEqual(rejected["accepted_regions"], [])
        self.assertEqual(rejected["rejected_regions"][0]["failed_filter_ids"], ["box"])
        self.assertFalse(np.any(rejected["class_index_map"]))

    def test_bounding_box_filter_checks_width_and_height_independently(self):
        class_index_map = np.zeros((12, 12), dtype=np.int32)
        class_index_map[1:9, 2:6] = 1
        profile_class = {
            "name": "Class",
            "class_index": 0,
            "group": "",
            "identify": True,
            "display_color_rgb": (1, 2, 3),
        }
        box_filter = {
            "id": "box",
            "name": "Wide but not tall",
            "type": FILTER_TYPE_BOUNDING_BOX,
            "enabled": True,
            "minimum_width_um": 4,
            "maximum_width_um": None,
            "minimum_height_um": None,
            "maximum_height_um": 7,
        }

        rejected = filter_combined_class_regions(
            class_index_map,
            [profile_class],
            pixel_size_um=1,
            profile_filters=[box_filter],
        )
        self.assertEqual(rejected["accepted_regions"], [])
        self.assertEqual(rejected["rejected_regions"][0]["width_um"], 4)
        self.assertEqual(rejected["rejected_regions"][0]["height_um"], 8)

        box_filter["maximum_height_um"] = 8
        accepted = filter_combined_class_regions(
            class_index_map,
            [profile_class],
            pixel_size_um=1,
            profile_filters=[box_filter],
        )
        self.assertEqual(len(accepted["accepted_regions"]), 1)

    def test_perimeter_uses_component_mask_and_includes_hole_boundaries(self):
        profile_class = {
            "name": "Class",
            "class_index": 0,
            "group": "",
            "identify": True,
            "display_color_rgb": (1, 2, 3),
        }
        filled_map = np.zeros((16, 18), dtype=np.int32)
        filled_map[2:10, 3:13] = 1
        holed_map = filled_map.copy()
        holed_map[5:7, 6:9] = 0

        filled = combined_class_regions(
            filled_map,
            [profile_class],
            pixel_size_um=0.5,
        )[0]
        holed = combined_class_regions(
            holed_map,
            [profile_class],
            pixel_size_um=0.5,
        )[0]

        self.assertEqual(holed["width_um"], 5)
        self.assertEqual(holed["height_um"], 4)
        self.assertEqual(holed["size_um"], 5)
        self.assertEqual(holed["area_um2"], holed["pixel_count"] * 0.25)
        self.assertGreater(holed["perimeter_px"], filled["perimeter_px"])
        self.assertAlmostEqual(
            holed["perimeter_area_ratio_per_um"],
            holed["perimeter_um"] / holed["area_um2"],
        )

    def test_perimeter_area_filter_distinguishes_compact_and_thin_shapes(self):
        class_index_map = np.zeros((18, 24), dtype=np.int32)
        class_index_map[2:7, 2:7] = 1
        class_index_map[12, 12:21] = 1
        profile_class = {
            "name": "Class",
            "class_index": 0,
            "group": "",
            "identify": True,
            "display_color_rgb": (1, 2, 3),
        }
        regions = combined_class_regions(
            class_index_map,
            [profile_class],
            pixel_size_um=1,
        )
        ratios = sorted(
            region["perimeter_area_ratio_per_um"] for region in regions
        )
        threshold = sum(ratios) / 2
        shape_filter = {
            "id": "shape",
            "name": "Thin shapes",
            "type": FILTER_TYPE_PERIMETER_AREA_RATIO,
            "enabled": True,
            "minimum_per_um": threshold,
            "maximum_per_um": None,
        }

        filtered = filter_combined_class_regions(
            class_index_map,
            [profile_class],
            pixel_size_um=1,
            profile_filters=[shape_filter],
        )

        self.assertEqual(len(filtered["accepted_regions"]), 1)
        self.assertEqual(len(filtered["rejected_regions"]), 1)
        self.assertGreater(
            filtered["accepted_regions"][0]["perimeter_area_ratio_per_um"],
            filtered["rejected_regions"][0]["perimeter_area_ratio_per_um"],
        )

    def test_enabled_physical_filters_require_calibration_and_neural_is_explicit(self):
        class_index_map = np.zeros((5, 5), dtype=np.int32)
        class_index_map[1:4, 1:4] = 1
        profile_class = {
            "name": "Class",
            "class_index": 0,
            "group": "",
            "identify": True,
            "display_color_rgb": (1, 2, 3),
        }
        disabled_size = {
            "id": "size",
            "name": "Size",
            "type": FILTER_TYPE_SIZE,
            "enabled": False,
            "minimum_um": 1,
            "maximum_um": None,
        }
        self.assertEqual(len(filter_combined_class_regions(
            class_index_map,
            [profile_class],
            profile_filters=[disabled_size],
        )["accepted_regions"]), 1)

        enabled_size = dict(disabled_size, enabled=True)
        with self.assertRaisesRegex(ValueError, "calibrated pixel size"):
            filter_combined_class_regions(
                class_index_map,
                [profile_class],
                profile_filters=[enabled_size],
            )

        neural = {
            "id": "neural",
            "name": "Neural",
            "type": FILTER_TYPE_NEURAL_NETWORK,
            "enabled": True,
        }
        with self.assertRaisesRegex(ValueError, "not implemented"):
            filter_combined_class_regions(
                class_index_map,
                [profile_class],
                pixel_size_um=1,
                profile_filters=[neural],
            )

    def test_detection_manifest_v2_serializes_region_statistics(self):
        detector = Flake_Detector()
        detection = {
            "matched_class": "Thin",
            "matched_group": "Graphene",
            "bounding_box": {"x": 2, "y": 3, "width": 4, "height": 5},
            "pixel_count": 17,
            "size_um": 2.5,
            "area_um2": 4.25,
            "width_um": 2.0,
            "height_um": 2.5,
            "perimeter_px": 13.5,
            "perimeter_um": 6.75,
            "perimeter_area_ratio_per_um": 6.75 / 4.25,
        }
        with tempfile.TemporaryDirectory() as directory:
            scan_path = Path(directory)
            image_path = scan_path / "All Images" / "10x" / "wafer 1" / "Raw" / "img.png"
            record = detector._create_detection_record(
                detection,
                "Region Detection",
                {
                    "path": image_path,
                    "magnification": "10X",
                    "pixel_size_um": 0.5,
                },
                (20, 30, 3),
                scan_path,
            )
            bad_legacy = detector._create_detection_record(
                (0, (1, 1, 2, 2)),
                "Flake Detection",
                {"path": image_path, "magnification": "10X"},
                (20, 30, 3),
                scan_path,
            )
            good_legacy = detector._create_detection_record(
                {
                    "class_id": 1,
                    "bounding_box": (4, 4, 2, 2),
                    "pixel_count": 3,
                    "perimeter_px": 4,
                },
                "Flake Detection",
                {
                    "path": image_path,
                    "magnification": "10X",
                    "pixel_size_um": 0.5,
                },
                (20, 30, 3),
                scan_path,
            )
            detector._save_detection_records(
                scan_path,
                "Region Detection",
                None,
                [record, bad_legacy, good_legacy],
            )
            payload = json.loads(
                (scan_path / "flakes_found.json").read_text(encoding="utf-8")
            )

        self.assertEqual(payload["version"], 2)
        self.assertTrue(payload["detections"][0]["accepted"])
        self.assertFalse(payload["detections"][1]["accepted"])
        self.assertTrue(payload["detections"][2]["accepted"])
        self.assertEqual(payload["detections"][0]["statistics"]["pixel_count"], 17)
        self.assertEqual(payload["detections"][0]["statistics"]["perimeter_px"], 13.5)
        self.assertAlmostEqual(
            payload["detections"][0]["statistics"][
                "perimeter_area_ratio_per_um"
            ],
            6.75 / 4.25,
        )
        self.assertEqual(payload["detections"][2]["statistics"], {
            "pixel_count": 3,
            "size_um": 1.0,
            "area_um2": 0.75,
            "width_um": 1.0,
            "height_um": 1.0,
            "perimeter_px": 4.0,
            "perimeter_um": 2.0,
            "perimeter_area_ratio_per_um": 2.0 / 0.75,
        })

    def test_legacy_detection_saves_clean_processed_tile_and_annotated_preview(self):
        detector = Flake_Detector()
        source_bgr = np.zeros((8, 9, 3), dtype=np.uint8)
        source_bgr[:, :] = (11, 37, 83)
        annotated_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
        annotated_rgb = annotated_rgb.copy()
        annotated_rgb[1:4, 2:6] = (255, 0, 0)
        saved_images = []

        class FrameProcessor:
            @staticmethod
            def save_image(image, *, save_dir, filename, **_kwargs):
                saved_images.append(
                    (Path(save_dir) / filename, np.asarray(image).copy())
                )

        detector.flake_identifier.identify_flakes_flake_model = (
            lambda _image: (annotated_rgb, [(1, (2, 1, 4, 3))], True)
        )

        with tempfile.TemporaryDirectory() as directory:
            scan_path = Path(directory) / "scan"
            raw_dir = scan_path / "All Images" / "10x" / "wafer 1" / "Raw"
            raw_dir.mkdir(parents=True)
            source_path = raw_dir / "tile.png"
            cv2.imwrite(str(source_path), source_bgr)
            image_queue = Queue()
            image_queue.put({"path": source_path, "magnification": "10X"})
            image_queue.put(None)

            with (
                patch(
                    "Scanning.flake_detection.image_metadata.is_vignette_corrected",
                    return_value=False,
                ),
                patch(
                    "Scanning.flake_detection.image_metadata.read_png_metadata",
                    return_value={},
                ),
            ):
                detector.flake_detection(
                    image_queue,
                    FrameProcessor(),
                    detection_model="Flake Detection",
                    scan_path=scan_path,
                )

        self.assertEqual(len(saved_images), 2)
        processed_path, processed_image = saved_images[0]
        preview_path, preview_image = saved_images[1]
        self.assertEqual(processed_path.parent.name, "Processed")
        self.assertTrue(np.array_equal(processed_image, source_bgr))
        self.assertEqual(preview_path.parent.parent.name, "Flakes Found")
        self.assertTrue(
            np.array_equal(
                preview_image,
                cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR),
            )
        )

    def test_region_pipeline_classifies_with_a_contrast_range(self):
        image = np.full((12, 12, 3), 100, dtype=np.uint8)
        image[2:10, 2:10] = (120, 110, 105)
        contour = np.asarray(
            [[[2, 2]], [[9, 2]], [[9, 9]], [[2, 9]]],
            dtype=np.int32,
        )
        range_class = {
            "name": "Range",
            "classification_type": CLASSIFICATION_RANGE,
            "minimum_contrast_rgb": np.asarray((18, 8, 3), dtype=float),
            "maximum_contrast_rgb": np.asarray((22, 12, 7), dtype=float),
            "class_index": 0,
            "group": "",
            "identify": True,
            "display_color_rgb": (255, 0, 0),
        }

        result = classify_contour_regions(
            image,
            np.zeros((12, 12), dtype=np.uint8),
            [contour],
            np.asarray([[-1, -1, -1, -1]], dtype=np.int32),
            [contour],
            [0],
            {0: [0]},
            [range_class],
            2,
        )["region_results"][0]

        self.assertEqual(result["contrast_rgb"], (20, 10, 5))
        self.assertEqual(result["matched_class"], "Range")
        self.assertEqual(
            result["matched_classification_type"],
            CLASSIFICATION_RANGE,
        )
        self.assertIsNone(result["match_threshold"])

    def test_range_classification_uses_the_class_colored_region_mask(self):
        external_mask = np.zeros((7, 7), dtype=np.uint8)
        external_mask[1:6, 1:6] = 255
        class_index_map = np.zeros((7, 7), dtype=np.int32)
        class_index_map[2:5, 3] = 1
        class_index_map[4, 2:5] = 1
        range_class = {
            "classification_type": CLASSIFICATION_RANGE,
            "class_index": 0,
            "identify": True,
            "display_color_rgb": (12, 34, 56),
        }

        classified = create_classified_image(
            (7, 7, 3),
            external_mask,
            class_index_map,
            [range_class],
        )

        expected_region = class_index_map == 1
        self.assertTrue(np.all(classified[expected_region] == (12, 34, 56)))
        self.assertTrue(np.all(classified[1:6, 1:6][~expected_region[1:6, 1:6]] == 255))
        self.assertTrue(np.all(classified[external_mask == 0] == 0))

    def test_reference_and_runtime_flood_fill_are_color_connected(self):
        image_rgb = np.zeros((4, 6, 3), dtype=np.uint8)
        image_rgb[:, :3] = (150, 0, 0)
        image_rgb[:, 3:] = (0, 76, 0)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        self.assertTrue(np.all(gray == gray[0, 0]))

        runtime_mask = _flood_fill_region(
            image_rgb.copy(),
            (1, 1),
            np.ones(image_rgb.shape[:2], dtype=bool),
            5,
        )

        self.assertTrue(np.all(runtime_mask[:, :3]))
        self.assertFalse(np.any(runtime_mask[:, 3:]))

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        _, _, reference_mask, contour = get_region_from_point(
            image_bgr,
            (1, 1),
            threshold=5,
            connectivity=8,
            min_area=0,
        )

        self.assertIsNotNone(contour)
        self.assertTrue(np.all(reference_mask[:, :3] > 0))
        self.assertFalse(np.any(reference_mask[:, 3:] > 0))

    def test_contour_colors_precede_flood_fill_and_exclude_nested_interiors(self):
        image = np.full((60, 60, 3), 100, dtype=np.uint8)
        image[2:51, 2:51] = (120, 100, 100)
        image[20:41, 20:41] = (100, 130, 100)
        external = np.asarray(
            [[[2, 2]], [[50, 2]], [[50, 50]], [[2, 50]]],
            dtype=np.int32,
        )
        internal = np.asarray(
            [[[20, 20]], [[40, 20]], [[40, 40]], [[20, 40]]],
            dtype=np.int32,
        )
        edge_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(edge_mask, [internal], -1, 255, 1)
        hierarchy = np.asarray(
            [[-1, -1, 1, -1], [-1, -1, -1, 0]],
            dtype=np.int32,
        )

        def profile_class(name, contrast, index, color):
            return {
                "name": name,
                "contrast_rgb": np.asarray(contrast, dtype=float),
                "tolerance": 1,
                "class_index": index,
                "group": "",
                "identify": True,
                "display_color_rgb": color,
            }

        outer_class = profile_class("Outer", (20, 0, 0), 0, (255, 0, 0))
        inner_class = profile_class("Inner", (0, 30, 0), 1, (0, 255, 0))
        common_arguments = (
            image,
            edge_mask,
            [external, internal],
            hierarchy,
            [external],
            [0],
            {0: [0, 1]},
        )

        both_stats = {}
        both = classify_contour_regions(
            *common_arguments,
            [outer_class, inner_class],
            1,
            both_stats,
        )
        results_by_type = {
            item["region_type"]: item for item in both["region_results"]
        }

        self.assertEqual(results_by_type["external"]["contrast_rgb"], (20, 0, 0))
        self.assertEqual(results_by_type["external"]["matched_class"], "Outer")
        self.assertEqual(results_by_type["internal"]["contrast_rgb"], (0, 30, 0))
        self.assertEqual(results_by_type["internal"]["matched_class"], "Inner")
        self.assertTrue(all(item["flood_filled"] for item in both["region_results"]))
        self.assertEqual(both_stats["measure_contour_colors"]["calls"], 1)
        self.assertEqual(both_stats["flood_fill_region"]["calls"], 2)

        outer_only_stats = {}
        outer_only = classify_contour_regions(
            *common_arguments,
            [outer_class],
            1,
            outer_only_stats,
        )
        unmatched = next(
            item for item in outer_only["region_results"]
            if item["region_type"] == "internal"
        )

        self.assertIsNone(unmatched["matched_class"])
        self.assertFalse(unmatched["flood_filled"])
        self.assertEqual(outer_only_stats["flood_fill_region"]["calls"], 1)

        unrestricted = classify_contour_regions(
            *common_arguments,
            [outer_class],
            255,
        )
        unrestricted_outer = next(
            item for item in unrestricted["region_results"]
            if item["region_type"] == "external"
        )

        # The internal contour is excluded from the outer color measurement,
        # but it must not constrain the later flood-fill region.
        self.assertEqual(unrestricted_outer["contrast_rgb"], (20, 0, 0))
        self.assertGreater(
            unrestricted_outer["pixel_count"],
            unrestricted_outer["seed_component_pixel_count"],
        )
        self.assertEqual(unrestricted["matched_region_mask"][30, 30], 255)
        self.assertEqual(unrestricted["matched_region_mask"][20, 20], 255)

    def test_in_memory_profile_configuration_runs_the_preview_pipeline(self):
        image = np.full((100, 100, 3), 180, dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (79, 79), (70, 70, 70), -1)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[20:80, 20:80] = 255
        profile = ScanProfile()
        profile.classes.append(profile._make_class(
            "Preview class",
            Path("preview.png"),
            image,
            mask,
            (40, 40),
            255,
            8,
            contrast=(110, 110, 110),
        ))

        classified, contours, details = find_flakes(
            image,
            area_threshold=10,
            return_details=True,
            profile_configuration=profile.matching_configuration(),
            color_seed=0,
            legend_position=LEGEND_TOP_RIGHT,
        )

        self.assertEqual(classified.shape, image.shape)
        self.assertEqual(len(contours), 1)
        self.assertEqual(details["legend_position"], LEGEND_TOP_RIGHT)
        self.assertEqual(
            details["region_results"][0]["matched_class"],
            "Preview class",
        )

        without_legend, _, without_legend_details = find_flakes(
            image,
            area_threshold=10,
            return_details=True,
            profile_configuration=profile.matching_configuration(),
            color_seed=0,
            draw_legend=False,
        )
        self.assertIsNone(without_legend_details["legend_position"])
        self.assertTrue(np.array_equal(
            without_legend,
            without_legend_details["classified_image_without_legend"],
        ))
        self.assertFalse(np.array_equal(classified, without_legend))

        with_areas, _, area_details = find_flakes(
            image,
            area_threshold=10,
            return_details=True,
            profile_configuration=profile.matching_configuration(),
            color_seed=0,
            draw_legend=False,
            draw_region_areas=True,
            pixel_size_um=0.5,
        )
        self.assertFalse(np.array_equal(
            with_areas,
            area_details["classified_image_without_legend"],
        ))
        self.assertGreater(area_details["region_results"][0]["area_um2"], 0)
        self.assertEqual(len(area_details["combined_region_results"]), 1)
        self.assertGreater(
            area_details["combined_region_results"][0]["area_um2"],
            0,
        )

    def test_find_flakes_keeps_white_base_while_rejecting_components(self):
        image = np.full((100, 100, 3), 180, dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (79, 79), (70, 70, 70), -1)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[20:80, 20:80] = 255
        profile = ScanProfile()
        profile.classes.append(profile._make_class(
            "Preview class",
            Path("preview.png"),
            image,
            mask,
            (40, 40),
            255,
            8,
            contrast=(110, 110, 110),
        ))
        profile.set_filter(
            None,
            "Reject this region",
            FILTER_TYPE_SIZE,
            True,
            minimum_um=1000,
            maximum_um=None,
        )

        classified, _, details = find_flakes(
            image,
            area_threshold=10,
            return_details=True,
            profile_configuration=profile.matching_configuration(),
            pixel_size_um=0.5,
            color_seed=0,
            draw_legend=False,
        )

        self.assertEqual(details["combined_region_results"], [])
        self.assertEqual(len(details["rejected_combined_region_results"]), 1)
        self.assertEqual(len(details["all_combined_region_results"]), 1)
        self.assertFalse(np.any(details["class_index_map"]))
        self.assertFalse(np.any(details["matched_region_mask"]))
        self.assertEqual(details["unfiltered_external_mask"][50, 50], 255)
        self.assertEqual(details["external_mask"][50, 50], 0)
        self.assertTrue(np.array_equal(classified[50, 50], (255, 255, 255)))
        self.assertTrue(np.array_equal(
            classified,
            details["classified_image_without_legend"],
        ))

    def test_rejected_partial_class_removes_its_whole_white_candidate(self):
        external_mask = np.zeros((12, 18), dtype=np.uint8)
        external_mask[1:10, 1:8] = 255
        external_mask[2:9, 11:17] = 255
        unfiltered_class_map = np.zeros_like(external_mask, dtype=np.int32)
        unfiltered_class_map[4:6, 3:5] = 1
        unfiltered_class_map[4:7, 13:15] = 1
        filtered_class_map = np.zeros_like(unfiltered_class_map)
        filtered_class_map[4:7, 13:15] = 1

        filtered_external = _filter_external_candidates(
            external_mask,
            unfiltered_class_map,
            filtered_class_map,
        )

        self.assertFalse(np.any(filtered_external[1:10, 1:8]))
        self.assertTrue(np.all(filtered_external[2:9, 11:17] == 255))

    def test_class_legend_supports_each_image_corner(self):
        image = np.zeros((200, 240, 3), dtype=np.uint8)
        classes = [{
            "name": "A",
            "group": "",
            "identify": True,
            "display_color_rgb": (255, 80, 40),
        }]
        expectations = {
            LEGEND_TOP_LEFT: (False, False),
            LEGEND_TOP_RIGHT: (True, False),
            LEGEND_BOTTOM_LEFT: (False, True),
            LEGEND_BOTTOM_RIGHT: (True, True),
        }

        for position, (expect_right, expect_bottom) in expectations.items():
            with self.subTest(position=position):
                rendered = draw_class_legend(image, classes, position=position)
                ys, xs = np.nonzero(np.any(rendered != 0, axis=2))

                self.assertGreater(len(xs), 0)
                self.assertEqual(float(np.mean(xs)) > image.shape[1] / 2, expect_right)
                self.assertEqual(float(np.mean(ys)) > image.shape[0] / 2, expect_bottom)

        with self.assertRaises(ValueError):
            draw_class_legend(image, classes, position="center")

    def test_matched_region_area_is_measured_and_drawn_in_class_color(self):
        image = np.full((80, 120, 3), 255, dtype=np.uint8)
        regions = [{
            "matched_class": "Thin",
            "matched_color_rgb": (40, 220, 80),
            "bounding_box": {"x": 20, "y": 25, "width": 20, "height": 15},
            "area_um2": 123.4,
        }]

        rendered = draw_region_area_labels(image, regions)

        self.assertFalse(np.array_equal(rendered, image))
        self.assertTrue(np.any(np.all(rendered == (40, 220, 80), axis=2)))
        self.assertFalse(np.any(np.all(rendered == (0, 0, 0), axis=2)))

    def test_overlapping_flood_fills_become_one_combined_class_region(self):
        class_index_map = np.zeros((14, 18), dtype=np.int32)
        class_index_map[1:7, 1:7] = 1
        class_index_map[4:10, 4:11] = 1
        class_index_map[11:13, 14:17] = 1
        profile_classes = [{
            "name": "Thin",
            "group": "Graphene",
            "identify": True,
            "class_index": 0,
            "display_color_rgb": (40, 220, 80),
        }]

        combined = combined_class_regions(
            class_index_map,
            profile_classes,
            pixel_size_um=0.5,
        )

        self.assertEqual(len(combined), 2)
        expected_pixel_counts = sorted([
            int(np.count_nonzero(class_index_map[1:10, 1:11] == 1)),
            6,
        ])
        self.assertEqual(
            sorted(region["pixel_count"] for region in combined),
            expected_pixel_counts,
        )
        for region in combined:
            self.assertAlmostEqual(
                region["area_um2"],
                region["pixel_count"] * 0.25,
            )
            self.assertEqual(
                int(np.count_nonzero(region["instance_mask"])),
                region["pixel_count"],
            )

    def test_region_area_labels_move_apart_when_default_positions_overlap(self):
        image = np.zeros((120, 240, 3), dtype=np.uint8)
        first = {
            "matched_class": "Thin",
            "matched_color_rgb": (40, 220, 80),
            "bounding_box": {"x": 40, "y": 50, "width": 20, "height": 15},
            "area_um2": 123.4,
            "perimeter_area_ratio_per_um": 0.625,
        }
        one_label = draw_region_area_labels(image, [first])
        two_labels = draw_region_area_labels(image, [first, dict(first)])
        ratio_label = draw_region_measurement_labels(
            image,
            [first],
            show_area=False,
            show_perimeter_area_ratio=True,
        )

        self.assertFalse(np.array_equal(one_label, two_labels))
        self.assertFalse(np.array_equal(image, ratio_label))
        self.assertFalse(np.array_equal(one_label, ratio_label))

    def test_region_measurement_labels_are_small_stacked_and_unitless(self):
        class RecordingDraw:
            def __init__(self):
                self.labels = []

            @staticmethod
            def multiline_textbbox(_position, text, **_kwargs):
                lines = text.splitlines()
                return (0, 0, max(len(line) for line in lines) * 5, len(lines) * 8)

            def multiline_text(self, _position, text, **_kwargs):
                self.labels.append(text)

        image = np.zeros((100, 160, 3), dtype=np.uint8)
        region = {
            "matched_class": "Thin",
            "matched_color_rgb": (40, 220, 80),
            "bounding_box": {"x": 40, "y": 50, "width": 20, "height": 15},
            "area_um2": 12.5,
            "perimeter_area_ratio_per_um": 0.625,
        }
        recording_draw = RecordingDraw()

        with (
            patch(
                "Scanning.region_classifier.ImageDraw.Draw",
                return_value=recording_draw,
            ),
            patch(
                "Scanning.region_classifier._load_class_legend_font",
                return_value=object(),
            ) as load_font,
        ):
            draw_region_measurement_labels(
                image,
                [region],
                show_area=True,
                show_perimeter_area_ratio=True,
            )

        load_font.assert_called_once_with(8)
        self.assertEqual(
            recording_draw.labels,
            ["Size 12.5\nP:A 0.6250"],
        )

    def test_region_area_uses_segmented_pixel_count_and_pixel_scale(self):
        image = np.full((12, 12, 3), 100, dtype=np.uint8)
        image[2:10, 2:10] = 120
        contour = np.asarray(
            [[[2, 2]], [[9, 2]], [[9, 9]], [[2, 9]]],
            dtype=np.int32,
        )
        identify_class = {
            "name": "Class",
            "contrast_rgb": np.asarray((20, 20, 20), dtype=float),
            "tolerance": 2,
            "class_index": 0,
            "group": "",
            "identify": True,
            "display_color_rgb": (255, 0, 0),
        }
        result = classify_contour_regions(
            image,
            np.zeros((12, 12), dtype=np.uint8),
            [contour],
            np.asarray([[-1, -1, -1, -1]], dtype=np.int32),
            [contour],
            [0],
            {0: [0]},
            [identify_class],
            2,
            pixel_size_um=0.5,
        )
        region = result["region_results"][0]

        self.assertAlmostEqual(
            region["area_um2"],
            region["pixel_count"] * 0.25,
        )
        self.assertEqual(
            int(np.count_nonzero(region["instance_mask"])),
            region["pixel_count"],
        )

    def test_database_export_copies_selected_images_and_writes_json_last(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scan_path = root / "100x Scan (2026-07-31) (12-00-00)"
            image_20x = scan_path / "All Images" / "20x" / "group.png"
            image_100x = scan_path / "All Images" / "100x" / "group.png"
            image_20x.parent.mkdir(parents=True)
            image_100x.parent.mkdir(parents=True)
            image_20x.write_bytes(b"20x-image")
            image_100x.write_bytes(b"100x-image")
            (scan_path / "flakes_found.json").write_text(
                json.dumps({
                    "schema": "flake-search.100x-scan",
                    "version": 2,
                    "material": "Graphene",
                    "substrate_thickness": "285nm SiO2",
                }),
                encoding="utf-8",
            )
            record = {
                "classification": "Thin",
                "classification_group": "Graphene",
                "statistics": {"area_um2": 12.5},
                "location": {"stage_position_um": {"x": 1, "y": 2}},
                "image_20x": "All Images/20x/group.png",
                "image_100x": "All Images/100x/group.png",
            }
            destination = root / "to_read"

            exported = save_database(
                scan_path,
                record,
                destination=destination,
                flake_id="selected_0001",
            )
            payload = json.loads(exported["json"].read_text(encoding="utf-8"))

            self.assertEqual(exported["images"]["20X"].read_bytes(), b"20x-image")
            self.assertEqual(exported["images"]["100X"].read_bytes(), b"100x-image")
            self.assertTrue(image_20x.is_file())
            self.assertTrue(image_100x.is_file())
            self.assertEqual(payload["schema"], "flake-search.database-import")
            self.assertEqual(payload["flake_id"], "selected_0001")
            self.assertEqual(payload["metadata"]["material"], "Graphene")
            self.assertEqual(payload["metadata"]["classification"], "Thin")
            self.assertEqual(Path(payload["images"]["20X"]).parent, destination)
            self.assertFalse(any(destination.glob("*.part")))

    def test_database_export_rejects_images_outside_the_scan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scan_path = root / "scan"
            scan_path.mkdir()
            outside = root / "outside.png"
            outside.write_bytes(b"outside")
            record = {
                "image_20x": str(outside),
                "image_100x": str(outside),
            }
            with self.assertRaisesRegex(ValueError, "outside"):
                save_database(scan_path, record, destination=root / "to_read")

    def test_region_mask_encoding_preserves_exact_shape(self):
        mask = np.zeros((4, 5), dtype=np.uint8)
        mask[0, 1:4] = 255
        mask[2:4, 2] = 255

        encoded = encode_region_mask(mask)
        restored = np.zeros(mask.size, dtype=np.uint8)
        for offset in range(0, len(encoded["runs"]), 2):
            start, length = encoded["runs"][offset:offset + 2]
            restored[start:start + length] = 255

        self.assertTrue(np.array_equal(restored.reshape(mask.shape), mask))


if __name__ == "__main__":
    unittest.main()
