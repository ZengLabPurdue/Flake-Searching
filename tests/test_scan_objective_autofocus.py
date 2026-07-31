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
            def save_image(*, save_dir, filename, **_kwargs):
                return Path(save_dir) / filename

        class Identifier:
            @staticmethod
            def identify_flakes_flake_model(image_rgb):
                detections = [
                    (1, (10, 10, 4, 4)),
                    (1, (80, 80, 4, 4)),
                ]
                return image_rgb, detections, None

        class Detector:
            flake_identifier = Identifier()

            @staticmethod
            def _bounded_box(bounding_box, _image_shape):
                return bounding_box

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


if __name__ == "__main__":
    unittest.main()
