from collections import OrderedDict
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image


APP_DIR = Path(__file__).resolve().parents[1] / "App"
sys.path.insert(0, str(APP_DIR))

from UI.display_acceleration import (  # noqa: E402
    CPU_BACKEND,
    OPENCL_BACKEND,
    DisplayResizer,
    opencl_available,
    requested_backend,
)
from UI.sparse_tile_viewer import SparseTileViewer  # noqa: E402
from UI.panels.view_scans_panel import (  # noqa: E402
    ViewScansPanel,
    detection_result_key,
    format_region_statistics,
    is_scan_folder_name,
)


class _FakeCanvas:
    @staticmethod
    def winfo_width():
        return 800

    @staticmethod
    def winfo_height():
        return 600


class _FakeVariable:
    def __init__(self, value=False):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class DisplayAccelerationTests(unittest.TestCase):
    def test_camera_auto_uses_cpu_and_returns_requested_size(self):
        image = np.zeros((120, 200, 3), dtype=np.uint8)
        with patch.dict(
            os.environ,
            {"FLAKE_SEARCH_CAMERA_RENDERER": "auto"},
        ):
            resizer = DisplayResizer()

        resized, elapsed = resizer.resize(
            image,
            (50, 30),
            cv2.INTER_AREA,
        )

        self.assertEqual(resizer.backend, CPU_BACKEND)
        self.assertEqual(resized.shape, (30, 50, 3))
        self.assertGreaterEqual(elapsed, 0.0)

    def test_invalid_renderer_setting_uses_auto(self):
        with patch.dict(os.environ, {"TEST_RENDERER": "not-a-backend"}):
            self.assertEqual(requested_backend("TEST_RENDERER"), "auto")

    def test_interactive_map_buffer_is_smaller_than_quality_buffer(self):
        quality_size = SparseTileViewer._render_buffer_dimensions(1600, 900)
        interactive_size = SparseTileViewer._render_buffer_dimensions(
            1600,
            900,
            ratio=SparseTileViewer.INTERACTIVE_RENDER_BUFFER_RATIO,
        )

        self.assertEqual(quality_size, (3200, 1800))
        self.assertEqual(interactive_size, (1920, 1080))

    def test_stale_tile_request_does_not_open_the_source_image(self):
        viewer = self._viewer_without_tk()
        viewer._generation = 2
        missing_record = {"path": Path("this-file-does-not-exist.png")}

        sampled = viewer._sampled_tile(
            missing_record,
            factor=1,
            nearest=False,
            generation=1,
        )

        self.assertIsNone(sampled)

    def test_in_memory_region_image_uses_the_normal_pyramid_cache(self):
        viewer = self._viewer_without_tk()
        pixels = np.zeros((8, 10, 3), dtype=np.uint8)
        pixels[:, :, 1] = 180
        record = {
            "path": Path("memory-region.png"),
            "source_image": Image.fromarray(pixels),
            "cache_key": "memory-region-test",
        }

        sampled = viewer._sampled_tile(
            record,
            factor=2,
            nearest=True,
            generation=1,
        )

        self.assertEqual(sampled.size, (5, 4))
        self.assertTrue(np.all(np.asarray(sampled)[:, :, 1] == 180))
        self.assertEqual(len(viewer._sampled_cache), 1)

    def test_map_view_state_preserves_relative_position_and_zoom(self):
        source = self._viewer_without_tk()
        source.records = [object()]
        source.map_width = 1000.0
        source.map_height = 1000.0
        source.center_x = 400.0
        source.center_y = 600.0
        source.minimum_scale = 0.2
        source.scale = 0.4
        state = source.capture_view_state()

        target = self._viewer_without_tk()
        target.records = [object()]
        target.canvas = _FakeCanvas()
        target.map_width = 2000.0
        target.map_height = 2000.0
        target.maximum_scale = 1.0
        scheduled = []
        target._schedule_render = lambda **kwargs: scheduled.append(kwargs)

        target._apply_view_state(state)

        self.assertAlmostEqual(target.center_x, 800.0)
        self.assertAlmostEqual(target.center_y, 1200.0)
        self.assertAlmostEqual(
            target.scale / target.minimum_scale,
            2.0,
        )
        self.assertTrue(scheduled)

    def test_filtered_map_toggle_maps_2x_and_10x_layers(self):
        for current_view, checked, expected_view in (
            ("Raw 2x", True, "Filtered 2x"),
            ("Filtered 2x", False, "Raw 2x"),
            ("Raw 10x", True, "Processed 10x"),
            ("Processed 10x", False, "Raw 10x"),
        ):
            with self.subTest(current_view=current_view):
                panel = ViewScansPanel.__new__(ViewScansPanel)
                panel.selected_view = current_view
                pair = panel._map_layer_pair(current_view)
                panel.available_result_views = set(pair)
                panel.filtered_map_var = _FakeVariable(checked)
                expected_state = {
                    "center_x_ratio": 0.4,
                    "center_y_ratio": 0.6,
                    "zoom_ratio": 3.0,
                }
                panel.tile_viewer = type(
                    "_Viewer",
                    (),
                    {
                        "capture_view_state": (
                            lambda _self: expected_state
                        )
                    },
                )()
                calls = []
                panel.set_view_folder = (
                    lambda view, **kwargs: calls.append((view, kwargs))
                )

                panel._toggle_filtered_map()

                self.assertEqual(calls[0][0], expected_view)
                self.assertEqual(
                    calls[0][1]["view_state"],
                    expected_state,
                )
                self.assertTrue(calls[0][1]["preserve_context"])

    def test_result_raw_path_finds_matching_10x_or_20x_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            scan_path = Path(directory)
            result_path = scan_path / "Flakes Found" / "wafer 1" / "img_20x_3.png"
            raw_path = (
                scan_path
                / "All Images"
                / "20x"
                / "wafer 1"
                / "Raw"
                / result_path.name
            )
            raw_path.parent.mkdir(parents=True)
            raw_path.touch()

            panel = ViewScansPanel.__new__(ViewScansPanel)
            panel.view_scan_path = scan_path
            panel.magnification_roots = {}

            self.assertEqual(panel._raw_result_path(result_path), raw_path)

    def test_scan_discovery_exposes_individual_10x_images(self):
        with tempfile.TemporaryDirectory() as directory:
            scan_path = Path(directory)
            raw_path = (
                scan_path
                / "All Images"
                / "10x"
                / "wafer 1"
                / "Raw"
                / "img_10x_0.png"
            )
            processed_path = (
                scan_path
                / "All Images"
                / "10x"
                / "wafer 1"
                / "Processed"
                / raw_path.name
            )
            raw_path.parent.mkdir(parents=True)
            processed_path.parent.mkdir(parents=True)
            raw_path.touch()
            processed_path.touch()

            panel = ViewScansPanel.__new__(ViewScansPanel)
            panel.view_scan_path = scan_path

            roots, available = panel._discover_result_layers()

            self.assertEqual(roots["10x"], scan_path / "All Images" / "10x")
            self.assertIn("10x Images", available)

    def test_individual_capture_images_pair_raw_and_processed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            chip_folder = Path(directory) / "wafer 1"
            raw_folder = chip_folder / "Raw"
            processed_folder = chip_folder / "Processed"
            raw_folder.mkdir(parents=True)
            processed_folder.mkdir(parents=True)
            raw_zero = raw_folder / "img_10x_0.png"
            raw_one = raw_folder / "img_10x_1.png"
            processed_zero = processed_folder / raw_zero.name
            processed_two = processed_folder / "img_10x_2.png"
            for path in (
                raw_zero,
                raw_one,
                processed_zero,
                processed_two,
            ):
                path.touch()

            panel = ViewScansPanel.__new__(ViewScansPanel)
            panel.capture_chip_folder = chip_folder

            image_files = panel._capture_image_files(chip_folder)

            self.assertEqual(
                [path.name for path in image_files],
                ["img_10x_0.png", "img_10x_1.png", "img_10x_2.png"],
            )
            self.assertEqual(image_files[0], processed_zero)
            self.assertEqual(
                panel._capture_result_path(image_files[0], "Raw"),
                raw_zero,
            )
            self.assertEqual(
                panel._capture_result_path(image_files[0], "Processed"),
                processed_zero,
            )

    def test_raw_toggle_refreshes_individual_capture_image(self):
        panel = ViewScansPanel.__new__(ViewScansPanel)
        panel.selected_view = "10x Images"
        panel.capture_images_mode = True
        panel.raw_result_var = _FakeVariable(True)
        refreshed = []
        panel.display_current_image = lambda **kwargs: refreshed.append(kwargs)

        panel._toggle_raw_result()

        self.assertEqual(refreshed, [{"preserve_region": True}])
        self.assertTrue(panel.raw_result_var.get())

    def test_flakes_found_raw_toggle_preserves_selected_region(self):
        panel = ViewScansPanel.__new__(ViewScansPanel)
        panel.selected_view = "Flakes Found"
        panel.capture_images_mode = False
        panel.raw_result_var = _FakeVariable(True)
        refreshed = []
        panel.display_current_image = lambda **kwargs: refreshed.append(kwargs)

        panel._toggle_raw_result()

        self.assertEqual(refreshed, [{"preserve_region": True}])

    def test_detection_result_key_normalizes_source_path_slashes(self):
        expected = ("wafer 7", "img_20x_3.png")

        self.assertEqual(
            detection_result_key({
                "source_image": (
                    "All Images/20x/wafer 7/Raw/img_20x_3.png"
                ),
            }),
            expected,
        )
        self.assertEqual(
            detection_result_key({
                "source_image": (
                    r"All Images\20x\wafer 7\Raw\img_20x_3.png"
                ),
            }),
            expected,
        )
        self.assertEqual(
            detection_result_key({
                "result_image": "Flakes Found/Wafer 7/IMG_20X_3.PNG",
            }),
            expected,
        )

    def test_scan_folder_name_accepts_full_and_100x_scans_only(self):
        self.assertTrue(
            is_scan_folder_name("Full Scan (2026-07-31) (14-05-09)")
        )
        self.assertTrue(
            is_scan_folder_name("100x Scan (2026-07-31) (14-05-09)")
        )
        for folder_name in (
            "Scan (2026-07-31) (14-05-09)",
            "100X Scan (2026-07-31) (14-05-09)",
            "100x Scan (2026-07-31)",
            "Full Scan (2026-7-31) (14-05-09)",
            "Full Scan (2026-07-31) (14-05-09) backup",
        ):
            with self.subTest(folder_name=folder_name):
                self.assertFalse(is_scan_folder_name(folder_name))

    def test_detection_manifest_indexes_duplicate_filenames_by_chip(self):
        with tempfile.TemporaryDirectory() as directory:
            scan_path = Path(directory)
            records = [
                {
                    "classification": "A",
                    "source_image": (
                        "All Images/10x/wafer 1/Raw/img_10x_0.png"
                    ),
                },
                {
                    "classification": "B",
                    "source_image": (
                        r"All Images\10x\wafer 2\Raw\img_10x_0.png"
                    ),
                },
                {"classification": "missing source"},
            ]
            (scan_path / "flakes_found.json").write_text(
                json.dumps({"version": 2, "detections": records}),
                encoding="utf-8",
            )

            indexed = ViewScansPanel._load_region_records(scan_path)

            self.assertEqual(
                indexed[("wafer 1", "img_10x_0.png")][0]["classification"],
                "A",
            )
            self.assertEqual(
                indexed[("wafer 2", "img_10x_0.png")][0]["classification"],
                "B",
            )
            self.assertEqual(len(indexed), 2)

    def test_detection_manifest_indexes_only_accepted_good_regions(self):
        source = "All Images/10x/wafer 1/Raw/img_10x_0.png"
        with tempfile.TemporaryDirectory() as directory:
            scan_path = Path(directory)
            (scan_path / "flakes_found.json").write_text(
                json.dumps({
                    "detection_model": "Flake Detection",
                    "detections": [
                        {"source_image": source, "class_id": 0},
                        {"source_image": source, "class_id": 1},
                        {"source_image": source, "class_id": 2},
                        {
                            "source_image": source,
                            "class_id": 1,
                            "accepted": False,
                        },
                        {
                            "source_image": source,
                            "classification": "Legacy good record",
                        },
                    ],
                }),
                encoding="utf-8",
            )

            indexed = ViewScansPanel._load_region_records(scan_path)

        records = indexed[("wafer 1", "img_10x_0.png")]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["class_id"], 1)
        self.assertEqual(records[1]["classification"], "Legacy good record")

    def test_detection_manifest_skips_explicitly_rejected_profile_regions(self):
        source = "All Images/20x/wafer 2/Raw/img_20x_4.png"
        with tempfile.TemporaryDirectory() as directory:
            scan_path = Path(directory)
            (scan_path / "flakes_found.json").write_text(
                json.dumps({
                    "detection_model": "Region Detection",
                    "detections": [
                        {
                            "source_image": source,
                            "classification": "Accepted",
                            "accepted": True,
                        },
                        {
                            "source_image": source,
                            "classification": "Rejected",
                            "accepted": False,
                        },
                    ],
                }),
                encoding="utf-8",
            )

            indexed = ViewScansPanel._load_region_records(scan_path)

        records = indexed[("wafer 2", "img_20x_4.png")]
        self.assertEqual(
            [record["classification"] for record in records],
            ["Accepted"],
        )

    def test_missing_or_malformed_detection_manifest_is_nonfatal(self):
        with tempfile.TemporaryDirectory() as directory:
            scan_path = Path(directory)
            self.assertEqual(
                ViewScansPanel._load_region_records(scan_path),
                {},
            )

            (scan_path / "flakes_found.json").write_text(
                "not json",
                encoding="utf-8",
            )
            self.assertEqual(
                ViewScansPanel._load_region_records(scan_path),
                {},
            )

            (scan_path / "flakes_found.json").write_text(
                json.dumps({"detections": "not a list"}),
                encoding="utf-8",
            )
            self.assertEqual(
                ViewScansPanel._load_region_records(scan_path),
                {},
            )

    def test_region_statistics_show_physical_and_pixel_measurements(self):
        text = format_region_statistics({
            "classification": "Thin flake",
            "classification_group": "Graphene",
            "bounding_box_px": {
                "x": 12,
                "y": 24,
                "width": 40,
                "height": 20,
            },
            "statistics": {
                "pixel_count": 512,
                "size_um": 10.0,
                "area_um2": 32.125,
                "width_um": 10.0,
                "height_um": 5.0,
                "perimeter_px": 120.5,
                "perimeter_um": 30.125,
                "perimeter_area_ratio_per_um": 0.9375,
            },
        })

        self.assertIn("Class: Thin flake", text)
        self.assertIn("Group: Graphene", text)
        self.assertIn("Area: 32.12 µm²", text)
        self.assertIn("Size: 10 µm", text)
        self.assertIn("Width: 10 µm", text)
        self.assertIn("Height: 5 µm", text)
        self.assertIn("Perimeter: 30.12 µm", text)
        self.assertIn("Perimeter / area: 0.9375 µm⁻¹", text)
        self.assertIn("Pixels: 512 px", text)
        self.assertIn("Box (px): x 12, y 24, 40 × 20", text)

    def test_legacy_region_statistics_use_bbox_and_unavailable_markers(self):
        text = format_region_statistics({
            "classification": "Legacy",
            "bounding_box_px": {"width": 31, "height": 17},
        })

        self.assertIn("Area: —", text)
        self.assertIn("Size: 31 px", text)
        self.assertIn("Width: 31 px", text)
        self.assertIn("Height: 17 px", text)
        self.assertIn("Box (px): x —, y —, 31 × 17", text)

    def test_uncalibrated_statistics_fall_back_to_pixel_units(self):
        text = format_region_statistics({
            "bounding_box_px": {
                "x": 2,
                "y": 3,
                "width": 10,
                "height": 8,
            },
            "statistics": {
                "pixel_count": 80,
                "perimeter_px": 40,
            },
        })

        self.assertIn("Area: 80 px²", text)
        self.assertIn("Size: 10 px", text)
        self.assertIn("Perimeter: 40 px", text)
        self.assertIn("Perimeter / area: 0.5 px⁻¹", text)

    def test_region_navigation_wraps_without_changing_image(self):
        panel = ViewScansPanel.__new__(ViewScansPanel)
        panel.current_regions = [{"id": 1}, {"id": 2}, {"id": 3}]
        panel.view_region_index = 0
        panel.app = type("_App", (), {"get_view": lambda _self: "Scan Results"})()
        focus_calls = []
        panel.root = type("_Root", (), {"focus_set": lambda _self: None})()
        panel._show_current_region = (
            lambda fit_image=False: focus_calls.append(fit_image)
        )

        panel.previous_region()
        self.assertEqual(panel.view_region_index, 2)
        panel.next_region()
        self.assertEqual(panel.view_region_index, 0)
        panel.next_region()
        self.assertEqual(panel.view_region_index, 1)
        self.assertEqual(focus_calls, [True, True, True])

    def test_region_selection_resets_for_image_and_survives_raw_toggle(self):
        first_path = Path("Flakes Found") / "wafer 1" / "img_10x_0.png"
        second_path = Path("Flakes Found") / "wafer 2" / "img_10x_0.png"
        first_key = ("wafer 1", "img_10x_0.png")
        second_key = ("wafer 2", "img_10x_0.png")
        panel = ViewScansPanel.__new__(ViewScansPanel)
        panel.region_records_by_image = {
            first_key: [{"id": 1}, {"id": 2}],
            second_key: [{"id": 3}],
        }
        panel.current_region_key = first_key
        panel.current_regions = panel.region_records_by_image[first_key]
        panel.view_region_index = 1
        panel._show_current_region = lambda fit_image=False: None

        panel._set_current_regions(first_path, preserve_region=True)
        self.assertEqual(panel.view_region_index, 1)

        panel._set_current_regions(first_path, preserve_region=False)
        self.assertEqual(panel.view_region_index, 0)

        panel.view_region_index = 1
        panel._set_current_regions(second_path, preserve_region=True)
        self.assertEqual(panel.view_region_index, 0)
        self.assertEqual(panel.current_regions, [{"id": 3}])

    def test_region_navigation_keeps_full_image_and_colors_only_focus_shape(self):
        loaded = []
        viewer = type(
            "_Viewer",
            (),
            {
                "set_highlight": lambda _self, _bounds: None,
                "load_rgb_image": (
                    lambda _self, image, **options: loaded.append(
                        (image.copy(), options)
                    )
                ),
            },
        )()
        panel = ViewScansPanel.__new__(ViewScansPanel)
        panel.tile_viewer = viewer
        panel.current_result_path = Path("result.png")
        panel.raw_result_var = _FakeVariable(False)
        panel._processed_result_path = lambda _path: Path("processed.png")
        processed = np.full((12, 16, 3), 255, dtype=np.uint8)
        processed[4:7, 5:9] = (12, 34, 56)
        panel._load_processed_region_image = lambda _path: processed.copy()
        panel.current_regions = [{
            "bounding_box_px": {
                "x": 5,
                "y": 4,
                "width": 4,
                "height": 3,
            },
            "display_color_rgb": [12, 34, 56],
            "region_mask": {
                "encoding": "foreground-runs-row-major",
                "width": 4,
                "height": 3,
                "runs": [0, 2, 5, 2, 10, 2],
            },
        }]
        panel.view_region_index = 0
        panel.display_everything_var = _FakeVariable(False)

        panel._update_region_highlight(fit_image=True)

        focused = loaded[-1][0]
        expected_mask = np.zeros((3, 4), dtype=bool)
        expected_mask.reshape(-1)[[0, 1, 5, 6, 10, 11]] = True
        self.assertTrue(np.all(focused[4:7, 5:9][expected_mask] == (12, 34, 56)))
        self.assertTrue(np.all(focused[4:7, 5:9][~expected_mask] == 255))
        self.assertTrue(np.all(focused[:4] == 255))
        self.assertIsNone(loaded[-1][1]["view_state"])

        panel.display_everything_var.set(True)
        panel._update_region_highlight(fit_image=True)
        self.assertTrue(np.array_equal(loaded[-1][0], processed))

    def test_selected_100x_region_is_sent_to_database_export(self):
        panel = ViewScansPanel.__new__(ViewScansPanel)
        panel.current_regions = [{
            "classification": "Thin",
            "image_20x": "All Images/20x/group.png",
            "image_100x": "All Images/100x/group.png",
        }]
        panel.view_region_index = 0
        panel.view_scan_path = Path("100x Scan (2026-07-31) (12-00-00)")
        panel.root = object()
        exported_json = Path("to_read/selected.json")

        with (
            patch(
                "UI.panels.view_scans_panel.flake_detection.save_database",
                return_value={"json": exported_json},
            ) as save_database,
            patch("UI.panels.view_scans_panel.messagebox.showinfo") as showinfo,
        ):
            panel.save_selected_region_to_database()

        save_database.assert_called_once_with(
            panel.view_scan_path,
            panel.current_regions[0],
        )
        self.assertIn("to_read", showinfo.call_args.args[1])

    def test_sparse_viewer_highlight_has_a_visible_minimum_size(self):
        rectangles = []
        labels = []

        class Canvas:
            @staticmethod
            def create_rectangle(*coordinates, **options):
                rectangles.append((coordinates, options))

            @staticmethod
            def create_text(*coordinates, **options):
                labels.append((coordinates, options))

        viewer = SparseTileViewer.__new__(SparseTileViewer)
        viewer.canvas = Canvas()
        viewer.scale = 0.1
        viewer.highlight_bounds = (100.0, 200.0, 2.0, 3.0)

        viewer._draw_highlight(view_left=0.0, view_top=0.0)

        self.assertEqual(len(rectangles), 2)
        x1, y1, x2, y2 = rectangles[-1][0]
        self.assertGreaterEqual(x2 - x1, 18)
        self.assertGreaterEqual(y2 - y1, 18)
        self.assertEqual(labels[0][1]["text"], "Current region")

    def test_map_opencl_failure_falls_back_to_cpu(self):
        viewer = self._viewer_without_tk()
        viewer.render_backend = OPENCL_BACKEND
        viewer._select_render_records = lambda _snapshot: ((object(),), False)

        def fail_opencl(*_args):
            raise RuntimeError("simulated OpenCL failure")

        viewer._compose_viewport_opencl = fail_opencl
        viewer._compose_viewport_cpu = lambda *_args: (
            Image.new("RGB", (4, 3)),
            [],
            {1},
        )
        snapshot = {
            "generation": 1,
            "width": 4,
            "height": 3,
            "title": "test",
            "scale": 1.0,
            "view_left": 0.0,
            "view_top": 0.0,
            "view_right": 4.0,
            "view_bottom": 3.0,
        }

        result = viewer._compose_viewport(snapshot)

        self.assertEqual(result["render_backend"], CPU_BACKEND)
        self.assertEqual(viewer.render_backend, CPU_BACKEND)
        self.assertIn("simulated OpenCL failure", viewer.render_fallback_reason)

    @unittest.skipUnless(opencl_available(), "OpenCL is unavailable")
    def test_opencl_map_composition_matches_nearest_cpu_composition(self):
        with tempfile.TemporaryDirectory() as directory:
            tile_path = Path(directory) / "tile.png"
            y_values, x_values = np.mgrid[:80, :128]
            pixels = np.dstack(
                (
                    x_values.astype(np.uint8),
                    y_values.astype(np.uint8),
                    ((x_values + y_values) % 256).astype(np.uint8),
                )
            )
            Image.fromarray(pixels).save(tile_path)
            record = {
                "path": tile_path,
                "x": 0.0,
                "y": 0.0,
                "width": 128.0,
                "height": 80.0,
                "pixel_width": 128,
                "pixel_height": 80,
                "map_zoom": 1.0,
                "map_width": 128.0,
                "map_height": 80.0,
            }
            viewer = self._viewer_without_tk()
            snapshot = {
                "generation": 1,
                "width": 96,
                "height": 60,
                "viewport_width": 96,
                "viewport_height": 60,
                "view_left": 16.0,
                "view_top": 10.0,
                "view_right": 112.0,
                "view_bottom": 70.0,
                "scale": 1.0,
                "all_records": (record,),
                "buckets": {(0, 0): [0]},
                "overview_record": None,
                "sparse_tile_limit": 64,
                "title": "test",
                "nearest": True,
            }

            viewer.render_backend = CPU_BACKEND
            cpu_result = viewer._compose_viewport(snapshot)
            viewer.render_backend = OPENCL_BACKEND
            gpu_result = viewer._compose_viewport(snapshot)

            self.assertIsNotNone(cpu_result)
            self.assertIsNotNone(gpu_result)
            self.assertEqual(gpu_result["render_backend"], OPENCL_BACKEND)
            np.testing.assert_array_equal(
                np.asarray(cpu_result["viewport"]),
                np.asarray(gpu_result["viewport"]),
            )
            self.assertGreater(viewer._gpu_cache_bytes, 0)

    @staticmethod
    def _viewer_without_tk():
        viewer = SparseTileViewer.__new__(SparseTileViewer)
        viewer._generation = 1
        viewer._shutdown = False
        viewer._sampled_cache = OrderedDict()
        viewer._cache_bytes = 0
        viewer._gpu_cache = OrderedDict()
        viewer._gpu_cache_bytes = 0
        viewer._cache_lock = threading.Lock()
        viewer.render_fallback_reason = None
        viewer.render_device = cv2.ocl.Device_getDefault().name()
        return viewer


if __name__ == "__main__":
    unittest.main()
