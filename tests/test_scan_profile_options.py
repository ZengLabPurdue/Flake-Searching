import json
from pathlib import Path
import sys
import tempfile
import unittest

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
    draw_class_legend,
    draw_region_area_labels,
    load_profile_configuration,
    match_profile_class,
)
from Scanning.contour_finder import find_flakes  # noqa: E402
from Scanning.contour_extractor import get_region_from_point  # noqa: E402
from Scanning.scan_profile import (  # noqa: E402
    CLASSIFICATION_RANGE,
    PROFILE_VERSION,
    ScanProfile,
)


class ScanProfileOptionsTests(unittest.TestCase):
    def _add_class(
        self,
        profile,
        name,
        contrast,
        *,
        group="",
        identify=True,
        minimum=None,
        maximum=None,
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
            minimum,
            maximum,
            contrast=contrast,
            group=group,
            identify=identify,
        )
        profile.classes.append(profile_class)
        return profile_class

    def test_current_version_round_trip_persists_classes_and_profile_options(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = ScanProfile(directory)
            profile.set_size_requirement(4, 90)
            self._add_class(
                profile,
                "Thin blue",
                (-20, -10, 4),
                group="hBN",
                identify=False,
                minimum=8,
                maximum=40,
            )

            loaded = profile.save_profile("Options")
            payload = json.loads((loaded.path / "profile.json").read_text(encoding="utf-8"))

            self.assertEqual(payload["version"], PROFILE_VERSION)
            self.assertEqual(
                payload["size_requirement"],
                {"minimum_size_um": 4.0, "maximum_size_um": 90.0},
            )
            self.assertEqual(loaded.groups, ["hBN"])
            self.assertFalse(loaded.classes[0]["identify"])
            self.assertEqual(loaded.classes[0]["minimum_size_um"], 8.0)

    def test_range_class_round_trip_needs_no_reference_image(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = ScanProfile(directory)
            created = profile.set_range_class(
                "Graphene range",
                (-40, -30, -20),
                (-10, 5, 15),
                minimum_size_um=2,
                maximum_size_um=80,
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
            payload.pop("size_requirement", None)
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
                "minimum_size_um": None,
                "maximum_size_um": None,
            },
            {
                "name": "Disabled exact",
                "contrast_rgb": np.asarray((12, 12, 12), dtype=float),
                "tolerance": 10,
                "identify": False,
                "minimum_size_um": None,
                "maximum_size_um": None,
            },
        ]

        match = match_profile_class(np.asarray((12, 12, 12)), classes)

        self.assertEqual(match[1]["name"], "Nearest identify")

    def test_class_size_limits_are_applied_during_matching(self):
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

        self.assertIsNone(match_profile_class(np.asarray((1, 1, 1)), classes, size_um=6))
        self.assertEqual(
            match_profile_class(np.asarray((1, 1, 1)), classes, size_um=4)[1]["name"],
            "Small",
        )

    def test_matching_loader_reads_profile_size_and_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "profile.json"
            profile_path.write_text(
                json.dumps({
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

            self.assertEqual(configuration["minimum_size_um"], 3.0)
            self.assertEqual(configuration["maximum_size_um"], 30.0)
            self.assertEqual(configuration["classes"][0]["group"], "Artifacts")

    def test_region_classification_applies_profile_size(self):
        image = np.full((12, 12, 3), 100, dtype=np.uint8)
        image[2:10, 2:10] = 120
        contour = np.asarray([[[2, 2]], [[9, 2]], [[9, 9]], [[2, 9]]], dtype=np.int32)
        hierarchy = np.asarray([[-1, -1, -1, -1]], dtype=np.int32)
        identify_class = {
            "name": "Class",
            "contrast_rgb": np.asarray((20, 20, 20), dtype=float),
            "tolerance": 2,
            "class_index": 0,
            "group": "Material",
            "identify": True,
            "minimum_size_um": None,
            "maximum_size_um": None,
            "display_color_rgb": (255, 0, 0),
        }
        too_small = classify_contour_regions(
            image,
            np.zeros((12, 12), dtype=np.uint8),
            [contour],
            hierarchy,
            [contour],
            [0],
            {0: [0]},
            [identify_class],
            2,
            pixel_size_um=2,
            minimum_size_um=20,
        )

        self.assertFalse(too_small["region_results"][0]["inside_profile_size"])
        self.assertIsNone(too_small["region_results"][0]["matched_class"])

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
            "minimum_size_um": None,
            "maximum_size_um": None,
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
                "minimum_size_um": None,
                "maximum_size_um": None,
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
            None,
            None,
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
        }
        one_label = draw_region_area_labels(image, [first])
        two_labels = draw_region_area_labels(image, [first, dict(first)])

        self.assertFalse(np.array_equal(one_label, two_labels))

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
            "minimum_size_um": None,
            "maximum_size_um": None,
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


if __name__ == "__main__":
    unittest.main()
