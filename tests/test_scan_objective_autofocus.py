import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPOSITORY_ROOT / "App"
sys.path.insert(0, str(APP_DIR))

from Hardware.turret_controller import TurretController  # noqa: E402
from Scanning import scan_manager as scan_manager_module  # noqa: E402
from Scanning.scan_manager import ScanManager  # noqa: E402


class TurretAutofocusOptionTests(unittest.TestCase):
    @staticmethod
    def _controller():
        autofocus_magnifications = []

        class FrameProcessor:
            def __init__(self):
                self.default_exposure_calls = 0

            def set_default_exposure(self):
                self.default_exposure_calls += 1

            @staticmethod
            def set_auto_exposure(_active):
                return None

        class App:
            def __init__(self):
                self.magnification = "2X"
                self.frame_processor = FrameProcessor()

            @staticmethod
            def disable_buttons():
                return None

            @staticmethod
            def enable_buttons():
                return None

            def set_magnification(self, magnification):
                self.magnification = magnification

        class Stage:
            @staticmethod
            def get_position():
                return 100.0, 200.0, 300.0

            @staticmethod
            def move_to_xy(_x, _y, **_kwargs):
                return True

            @staticmethod
            def move_to_z(_z, **_kwargs):
                return True

        class Turret:
            def __init__(self):
                self.position = 1

            def check_position(self):
                return self.position

            def turn_to_position(self, position, **_kwargs):
                self.position = position

        app = App()
        controller = TurretController.__new__(TurretController)
        controller.app = app
        controller.stage = Stage()
        controller.turret = Turret()
        controller.start_auto_focus_thread = lambda: (
            autofocus_magnifications.append(app.magnification)
        )
        return controller, app, autofocus_magnifications

    def test_change_objective_autofocus_defaults_on_for_manual_callers(self):
        controller, app, autofocus_magnifications = self._controller()

        controller.change_objective(3)

        self.assertEqual(app.magnification, "20X")
        self.assertEqual(app.frame_processor.default_exposure_calls, 1)
        self.assertEqual(autofocus_magnifications, ["20X"])

    def test_change_objective_can_skip_only_the_autofocus_step(self):
        controller, app, autofocus_magnifications = self._controller()

        controller.change_objective(3, autofocus=False)

        self.assertEqual(controller.get_position(), 3)
        self.assertEqual(app.magnification, "20X")
        self.assertEqual(app.frame_processor.default_exposure_calls, 1)
        self.assertEqual(autofocus_magnifications, [])


class ScanObjectiveAutofocusTests(unittest.TestCase):
    def test_change_objective_threads_autofocus_option_to_turret(self):
        calls = []

        class Stage:
            @staticmethod
            def wait_until_not_busy(**_kwargs):
                return None

        class Turret:
            @staticmethod
            def change_objective(position, **options):
                calls.append((position, options))

        manager = ScanManager.__new__(ScanManager)
        manager.stage = Stage()
        manager.turret_controller = Turret()
        manager._stop_event = threading.Event()

        manager._change_objective(3, autofocus=False)
        manager._change_objective(5)

        self.assertEqual(
            [(position, options["autofocus"]) for position, options in calls],
            [(3, False), (5, True)],
        )
        self.assertTrue(all(callable(options["cancel_check"]) for _, options in calls))

    @staticmethod
    def _plane_scan_manager(objective_calls):
        class App:
            @staticmethod
            def set_live_mapping(_active):
                return None

            @staticmethod
            def open_panel(_panel):
                return None

            @staticmethod
            def set_view(_view):
                return None

            @staticmethod
            def set_true_map(_image):
                return None

        manager = ScanManager.__new__(ScanManager)
        manager.app = App()
        manager._stop_event = threading.Event()
        manager.update_scan_status = lambda **_status: None
        manager._change_objective = (
            lambda position, autofocus=True: objective_calls.append(
                (position, autofocus)
            )
        )
        manager._move_stage_xy = lambda *_args, **_kwargs: None
        manager._calibrate_scan_focus_plane = (
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("stop after calibration starts")
            )
        )
        return manager

    def test_plane_scans_skip_redundant_objective_change_autofocus(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            scan_path = Path(temp_dir)

            ten_x_calls = []
            ten_x_manager = self._plane_scan_manager(ten_x_calls)
            with self.assertRaisesRegex(
                RuntimeError,
                "stop after calibration starts",
            ):
                ten_x_manager.run_10x_scan(
                    [[0.0, 0.0, 1, 1]],
                    scan_path=scan_path,
                )

            twenty_x_calls = []
            twenty_x_manager = self._plane_scan_manager(twenty_x_calls)
            with self.assertRaisesRegex(
                RuntimeError,
                "stop after calibration starts",
            ):
                twenty_x_manager.run_20x_scan(
                    [[0.0, 0.0, 1, 1]],
                    scan_path=scan_path,
                )

        self.assertEqual(ten_x_calls, [(2, False)])
        self.assertEqual(twenty_x_calls, [(2, False), (3, False)])

    def test_grouped_100x_scan_focuses_captures_but_not_navigation(self):
        objective_calls = []
        saved_images = []
        source_image = np.zeros((100, 100, 3), dtype=np.uint8)

        class App:
            @staticmethod
            def set_live_mapping(_active):
                return None

            @staticmethod
            def set_view(_view):
                return None

            @staticmethod
            def open_panel(_panel):
                return None

        class Stage:
            @staticmethod
            def get_position():
                return 1000.0, 2000.0, 3000.0

        class FrameProcessor:
            @staticmethod
            def capture_frame(**_kwargs):
                return source_image.copy()

            @staticmethod
            def apply_vignette_filter(image):
                return image

            @staticmethod
            def save_image(*, image, save_dir, filename, **_kwargs):
                saved_images.append({
                    "image": image.copy(),
                    "save_dir": Path(save_dir),
                    "filename": filename,
                })
                return Path(save_dir) / filename

        class Identifier:
            @staticmethod
            def identify_flakes_flake_model(image_rgb):
                detections = [
                    {
                        "class_id": 1,
                        "bounding_box": (10, 10, 4, 4),
                        "pixel_count": 12,
                        "perimeter_px": 10,
                    },
                    {
                        "class_id": 1,
                        "bounding_box": (80, 80, 4, 4),
                        "pixel_count": 11,
                        "perimeter_px": 9,
                    },
                ]
                return np.full_like(image_rgb, 255), detections, None

        class Detector:
            flake_identifier = Identifier()
            _bounded_box = staticmethod(
                scan_manager_module.flake_detection.Flake_Detector._bounded_box
            )
            _create_detection_record = (
                scan_manager_module.flake_detection.Flake_Detector._create_detection_record
            )

        manager = ScanManager.__new__(ScanManager)
        manager.app = App()
        manager.stage = Stage()
        manager.frame_processor = FrameProcessor()
        manager.resolution = "HIGH"
        manager._stop_event = threading.Event()
        manager.update_scan_status = lambda **_status: None
        manager._move_stage_xy = lambda *_args, **_kwargs: None
        manager._process_ui_events = lambda: None
        manager._change_objective = (
            lambda position, autofocus=True: objective_calls.append(
                (position, autofocus)
            )
        )

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(
                scan_manager_module,
                "HOME_DIR",
                Path(temp_dir),
            ),
            patch.object(
                scan_manager_module.flake_detection,
                "Flake_Detector",
                return_value=Detector(),
            ),
        ):
            manager.run_100x_scan(
                detection_model="Flake Detection",
                group_threshold_um=0.01,
            )

            scan_paths = list((Path(temp_dir) / "Scans").iterdir())
            self.assertEqual(len(scan_paths), 1)
            scan_path = scan_paths[0]
            manifest = json.loads(
                (scan_path / "flakes_found.json").read_text(encoding="utf-8")
            )
            saved_by_directory = {
                item["save_dir"].relative_to(scan_path).as_posix(): item
                for item in saved_images
                if item["filename"] == "source_10x.png"
            }

        self.assertEqual(
            objective_calls,
            [
                (2, True),
                (3, False),
                (5, True),
                (3, False),
                (5, True),
                (3, False),
            ],
        )
        self.assertEqual(manifest["schema"], "flake-search.100x-scan")
        self.assertEqual(manifest["version"], 2)
        self.assertEqual(manifest["detections"], manifest["flakes"])
        self.assertEqual(
            manifest["source_image"],
            "All Images/10x/source/Raw/source_10x.png",
        )
        self.assertEqual(manifest["flake_count"], 2)
        self.assertTrue(all(item["accepted"] for item in manifest["detections"]))
        self.assertEqual(
            [item["statistics"]["pixel_count"] for item in manifest["detections"]],
            [12, 11],
        )
        self.assertTrue(all(
            item["statistics"]["area_um2"] is not None
            and item["statistics"]["perimeter_um"] is not None
            and item["statistics"]["perimeter_area_ratio_per_um"] is not None
            for item in manifest["detections"]
        ))
        self.assertTrue(all(
            item["result_image"] == "Flakes Found/source/source_10x.png"
            for item in manifest["detections"]
        ))
        self.assertEqual(
            set(saved_by_directory),
            {
                "All Images/10x/source/Raw",
                "All Images/10x/source/Processed",
                "Flakes Found/source",
            },
        )
        self.assertFalse(np.any(
            saved_by_directory["All Images/10x/source/Raw"]["image"]
        ))
        self.assertFalse(np.any(
            saved_by_directory["All Images/10x/source/Processed"]["image"]
        ))
        self.assertTrue(np.all(
            saved_by_directory["Flakes Found/source"]["image"] == 255
        ))
        capture_saves = [
            item for item in saved_images
            if item["save_dir"].name == "100x"
        ]
        self.assertEqual(len(capture_saves), 2)
        self.assertTrue(all(
            item["save_dir"].relative_to(scan_path).as_posix()
            == "All Images/100x"
            for item in capture_saves
        ))

    def test_100x_region_manifest_preserves_statistics_and_result_paths(self):
        objective_calls = []
        saved_images = []
        source_image = np.zeros((100, 100, 3), dtype=np.uint8)
        processed_rgb = np.full_like(source_image, 40)
        annotated_rgb = np.full_like(source_image, 220)
        detection = {
            "matched_class": "Thin",
            "matched_group": "Graphene",
            "accepted": True,
            "bounding_box": {
                "x": 10,
                "y": 12,
                "width": 20,
                "height": 10,
            },
            "pixel_count": 150,
            "size_um": 10.0,
            "area_um2": 37.5,
            "width_um": 10.0,
            "height_um": 5.0,
            "perimeter_px": 55.5,
            "perimeter_um": 27.75,
            "perimeter_area_ratio_per_um": 0.74,
        }

        class App:
            @staticmethod
            def set_live_mapping(_active):
                return None

            @staticmethod
            def set_view(_view):
                return None

            @staticmethod
            def open_panel(_panel):
                return None

        class Stage:
            @staticmethod
            def get_position():
                return 1000.0, 2000.0, 3000.0

        class FrameProcessor:
            @staticmethod
            def capture_frame(**_kwargs):
                return source_image.copy()

            @staticmethod
            def apply_vignette_filter(image):
                return image

            @staticmethod
            def save_image(*, image, save_dir, filename, **_kwargs):
                saved_images.append({
                    "image": image.copy(),
                    "save_dir": Path(save_dir),
                    "filename": filename,
                })
                return Path(save_dir) / filename

        class Identifier:
            @staticmethod
            def identify_flakes_region_model(
                image_rgb,
                _profile_path,
                **_kwargs,
            ):
                return (
                    annotated_rgb.copy(),
                    [detection],
                    True,
                    processed_rgb.copy(),
                )

        class Detector:
            flake_identifier = Identifier()
            _bounded_box = staticmethod(
                scan_manager_module.flake_detection.Flake_Detector._bounded_box
            )
            _create_detection_record = (
                scan_manager_module.flake_detection.Flake_Detector._create_detection_record
            )

        manager = ScanManager.__new__(ScanManager)
        manager.app = App()
        manager.stage = Stage()
        manager.frame_processor = FrameProcessor()
        manager.resolution = "HIGH"
        manager._stop_event = threading.Event()
        manager.update_scan_status = lambda **_status: None
        manager._move_stage_xy = lambda *_args, **_kwargs: None
        manager._process_ui_events = lambda: None
        manager._change_objective = (
            lambda position, autofocus=True: objective_calls.append(
                (position, autofocus)
            )
        )

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(scan_manager_module, "HOME_DIR", Path(temp_dir)),
            patch.object(
                scan_manager_module.flake_detection,
                "Flake_Detector",
                return_value=Detector(),
            ),
            patch.object(
                scan_manager_module.region_dataset,
                "save_region_training_images",
            ),
        ):
            manager.run_100x_scan(
                detection_model="Region Detection",
                profile_path=Path(temp_dir) / "profile.json",
                group_threshold_um=0.01,
            )

            scan_paths = list((Path(temp_dir) / "Scans").iterdir())
            self.assertEqual(len(scan_paths), 1)
            scan_path = scan_paths[0]
            manifest = json.loads(
                (scan_path / "flakes_found.json").read_text(encoding="utf-8")
            )
            saved_by_directory = {
                item["save_dir"].relative_to(scan_path).as_posix(): item
                for item in saved_images
                if item["filename"] == "source_10x.png"
            }

        self.assertEqual(
            objective_calls,
            [(2, True), (3, False), (5, True), (3, False)],
        )
        self.assertEqual(manifest["schema"], "flake-search.100x-scan")
        self.assertEqual(manifest["version"], 2)
        self.assertEqual(manifest["detections"], manifest["flakes"])
        self.assertEqual(manifest["flake_count"], 1)
        self.assertEqual(manifest["capture_count"], 1)

        record = manifest["detections"][0]
        self.assertEqual(record["classification"], "Thin")
        self.assertEqual(record["classification_group"], "Graphene")
        self.assertIs(record["accepted"], True)
        self.assertEqual(
            record["source_image"],
            "All Images/10x/source/Raw/source_10x.png",
        )
        self.assertEqual(
            record["result_image"],
            "Flakes Found/source/source_10x.png",
        )
        self.assertEqual(
            record["bounding_box_px"],
            {"x": 10.0, "y": 12.0, "width": 20.0, "height": 10.0},
        )
        self.assertEqual(record["statistics"], {
            "pixel_count": 150,
            "size_um": 10.0,
            "area_um2": 37.5,
            "width_um": 10.0,
            "height_um": 5.0,
            "perimeter_px": 55.5,
            "perimeter_um": 27.75,
            "perimeter_area_ratio_per_um": 0.74,
        })
        self.assertEqual(
            record["location"]["stage_position_um"],
            record["target_position_10x_um"],
        )
        self.assertEqual(
            record["capture_stage_position_um"],
            {"x": 1000.0, "y": 2000.0},
        )
        self.assertEqual(record["capture_group_id"], 1)
        self.assertEqual(record["image_20x"], "All Images/20x/flake_group_0001.png")
        self.assertEqual(record["image_100x"], "All Images/100x/flake_group_0001.png")

        self.assertEqual(
            set(saved_by_directory),
            {
                "All Images/10x/source/Raw",
                "All Images/10x/source/Processed",
                "Flakes Found/source",
            },
        )
        self.assertTrue(np.all(
            saved_by_directory["All Images/10x/source/Processed"]["image"] == 40
        ))
        self.assertTrue(np.all(
            saved_by_directory["Flakes Found/source"]["image"] == 220
        ))
        capture_saves = [
            item for item in saved_images
            if item["filename"] == "flake_group_0001.png"
        ]
        self.assertEqual(len(capture_saves), 2)
        self.assertEqual(
            {
                item["save_dir"].relative_to(scan_path).as_posix()
                for item in capture_saves
            },
            {"All Images/20x", "All Images/100x"},
        )


if __name__ == "__main__":
    unittest.main()
