from pathlib import Path
import sys
import threading
import unittest


APP_DIR = Path(__file__).resolve().parents[1] / "App"
sys.path.insert(0, str(APP_DIR))

from Scanning import coordinate_generator  # noqa: E402
from Scanning.focus_plane import FocusPlane  # noqa: E402
from Scanning.scan_manager import ScanManager  # noqa: E402


class _FakeStage:
    def __init__(self):
        self.xy = (0.0, 0.0)
        self.z = 0.0
        self.last_confirmed_xy = None
        self.xy_moves = []
        self.z_moves = []
        self.motion_sequence = 0

    @staticmethod
    def wait_until_not_busy(**_kwargs):
        return 0.0

    def move_to_xy(self, x, y, wait=True, **_kwargs):
        self.xy = (float(x), float(y))
        self.last_confirmed_xy = (
            int(round(float(x))),
            int(round(float(y))),
        )
        self.xy_moves.append((float(x), float(y), wait))
        self.motion_sequence += 1
        return True

    def move_to_z(self, z, wait=True, **_kwargs):
        self.z = round(float(z), 1)
        self.z_moves.append((float(z), wait))
        self.motion_sequence += 1
        return True

    def get_z_position(self, strict=False):
        return self.z


class _FakeFocusController:
    def __init__(self, stage):
        self.stage = stage
        self.stop_focus_event = threading.Event()
        self.calls = []

    def auto_focus(self, **profile):
        x, y = self.stage.xy
        best_z = 35.0 + 0.012 * x - 0.007 * y
        self.stage.z = best_z
        self.calls.append((x, y, profile))
        return best_z


class _FakeApp:
    def __init__(self, focus_controller):
        self.focus_controller = focus_controller


class _FakeFrameProcessor:
    def __init__(self, stage):
        self.stage = stage
        self.capture_barriers = []

    def arm_capture_after_motion(self, expected_xy):
        self.capture_barriers.append(
            (expected_xy, self.stage.motion_sequence)
        )


class ScanAutofocusPlaneTests(unittest.TestCase):
    @staticmethod
    def _manager():
        stage = _FakeStage()
        focus_controller = _FakeFocusController(stage)
        frame_processor = _FakeFrameProcessor(stage)
        manager = ScanManager.__new__(ScanManager)
        manager.app = _FakeApp(focus_controller)
        manager.stage = stage
        manager.frame_processor = frame_processor
        manager.resolution = "HIGH"
        manager.update_scan_status = lambda **_status: None
        manager._stop_event = threading.Event()
        return manager, stage, focus_controller, frame_processor

    def test_three_samples_reconstruct_an_angled_focus_plane(self):
        def expected_z(x, y):
            return 72.0 + 0.015 * x - 0.009 * y

        samples = [
            (102000.0, 48000.0, expected_z(102000.0, 48000.0)),
            (102000.0, 42000.0, expected_z(102000.0, 42000.0)),
            (95000.0, 42000.0, expected_z(95000.0, 42000.0)),
        ]

        plane = FocusPlane.from_samples(samples)

        for x, y in ((95000.0, 48000.0), (98500.0, 45000.0)):
            self.assertAlmostEqual(plane.z_at(x, y), expected_z(x, y))

    def test_corner_selection_insets_large_grids_and_handles_small_grids(self):
        inset_offsets, _ = coordinate_generator.generate_rect_coords(5, 6)
        rect_offsets, _ = coordinate_generator.generate_rect_coords(3, 4)
        row_offsets, _ = coordinate_generator.generate_rect_coords(5, 1)
        one_offset, _ = coordinate_generator.generate_rect_coords(1, 1)

        self.assertEqual(
            ScanManager._focus_corner_offsets(inset_offsets),
            [(1, 1), (1, -2), (-1, -2)],
        )
        # A three-location axis cannot be inset on both sides without
        # collapsing two calibration points, but its four-location Y axis
        # can still move inward.
        self.assertEqual(
            ScanManager._focus_corner_offsets(rect_offsets),
            [(1, 0), (1, -1), (-1, -1)],
        )
        self.assertEqual(
            ScanManager._focus_corner_offsets(row_offsets),
            [(1, 0), (-1, 0)],
        )
        self.assertEqual(
            ScanManager._focus_corner_offsets(one_offset),
            [(0, 0)],
        )

    def test_10x_and_20x_calibration_focus_three_inner_points_and_predict_tiles(self):
        scan_offsets, _ = coordinate_generator.generate_rect_coords(5, 6)
        expected_focus_offsets = [(1, 1), (1, -2), (-1, -2)]

        for magnification in ("10X", "20X"):
            with self.subTest(magnification=magnification):
                manager, _stage, focus, _processor = self._manager()
                plane = manager._calibrate_scan_focus_plane(
                    center_x=50000,
                    center_y=25000,
                    scan_offsets=scan_offsets,
                    magnification=magnification,
                )

                self.assertEqual(len(focus.calls), 3)
                expected_focus_xy = [
                    manager._scan_target_xy(
                        50000,
                        25000,
                        offset_x,
                        offset_y,
                        magnification,
                    )
                    for offset_x, offset_y in expected_focus_offsets
                ]
                self.assertEqual(
                    [(x, y) for x, y, _profile in focus.calls],
                    expected_focus_xy,
                )
                self.assertTrue(
                    all(
                        profile
                        == manager.SCAN_AUTOFOCUS_PROFILES[magnification]
                        for _x, _y, profile in focus.calls
                    )
                )
                for offset_x, offset_y in scan_offsets:
                    target_x, target_y = manager._scan_target_xy(
                        50000,
                        25000,
                        offset_x,
                        offset_y,
                        magnification,
                    )
                    expected_z = 35.0 + 0.012 * target_x - 0.007 * target_y
                    self.assertAlmostEqual(
                        plane.z_at(target_x, target_y),
                        expected_z,
                    )

    def test_scan_position_arms_capture_after_xy_and_predicted_z_settle(self):
        manager, stage, _focus, frame_processor = self._manager()
        plane = FocusPlane.from_samples([
            (0.0, 0.0, 10.0),
            (100.0, 0.0, 12.0),
            (0.0, 100.0, 7.0),
        ])

        actual_z = manager._move_to_scan_position(25.0, 40.0, plane)

        self.assertEqual(len(stage.xy_moves), 1)
        self.assertEqual(len(stage.z_moves), 1)
        self.assertAlmostEqual(stage.z_moves[0][0], plane.z_at(25.0, 40.0))
        self.assertAlmostEqual(actual_z, round(plane.z_at(25.0, 40.0), 1))
        self.assertEqual(
            frame_processor.capture_barriers,
            [((25, 40), stage.motion_sequence)],
        )


if __name__ == "__main__":
    unittest.main()
