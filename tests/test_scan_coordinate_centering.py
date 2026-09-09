import sys
import unittest
from pathlib import Path

import numpy as np


APP_DIR = Path(__file__).resolve().parents[1] / "App"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from Scanning import coordinate_generator  # noqa: E402


class _HighResolutionApp:
    @staticmethod
    def get_resolution():
        return "HIGH"


class ScanCoordinateCenteringTests(unittest.TestCase):
    def test_even_and_odd_rectangular_axes_are_centered(self):
        coordinates, total = coordinate_generator.generate_rect_coords(10, 13)

        self.assertEqual(total, 130)
        x_offsets = sorted({x for x, _ in coordinates})
        y_offsets = sorted({y for _, y in coordinates})
        self.assertEqual(x_offsets, [
            -4.5, -3.5, -2.5, -1.5, -0.5,
            0.5, 1.5, 2.5, 3.5, 4.5,
        ])
        self.assertEqual(y_offsets, list(range(-6, 7)))
        self.assertAlmostEqual(sum(x_offsets) / len(x_offsets), 0.0)
        self.assertAlmostEqual(sum(y_offsets) / len(y_offsets), 0.0)

    def test_div2_grid_remains_centered(self):
        coordinates, _ = coordinate_generator.generate_rect_coords(4, 2, div2=True)

        self.assertEqual(
            sorted({x for x, _ in coordinates}),
            [-0.75, -0.25, 0.25, 0.75],
        )
        self.assertEqual(
            sorted({y for _, y in coordinates}),
            [-0.25, 0.25],
        )

    def test_affine_fit_recovers_rotated_stitched_map_geometry(self):
        expected = np.array([
            [-3.8, 0.12, 24000.0],
            [-0.08, -3.9, 1800.0],
        ])
        map_points = np.array([
            [1000.0, 1000.0],
            [9000.0, 1000.0],
            [1000.0, 9000.0],
            [9000.0, 9000.0],
            [5000.0, 5000.0],
        ])
        homogeneous = np.column_stack((map_points, np.ones(len(map_points))))
        stage_points = homogeneous @ expected.T

        fitted = coordinate_generator.fit_map_to_stage_affine(
            map_points,
            stage_points,
        )

        np.testing.assert_allclose(fitted, expected, atol=1e-10)
        np.testing.assert_allclose(
            coordinate_generator.map_point_to_stage(fitted, 4000, 6000),
            (9520.0, -21920.0),
            atol=1e-10,
        )

    def test_full_scan_uses_stitched_affine_and_target_objective_offset(self):
        true_map = np.zeros((10000, 10000, 3), dtype=np.uint8)
        chip = (4900, 4900, 200, 200)
        map_to_2x_stage = np.array([
            [-4.0, 0.0, 21000.0],
            [0.0, -4.0, 22000.0],
        ])

        ten_x = coordinate_generator.generate_10x_scan_coordinates(
            _HighResolutionApp(),
            [chip],
            scan_center_x=999999,
            scan_center_y=999999,
            scale=2,
            true_map=true_map.copy(),
            camera_size=(5440, 3648),
            map_to_stage_affine=map_to_2x_stage,
        )[0]
        twenty_x = coordinate_generator.generate_20x_scan_coordinates(
            _HighResolutionApp(),
            [chip],
            scan_center_x=999999,
            scan_center_y=999999,
            scale=2,
            true_map=true_map.copy(),
            camera_size=(5440, 3648),
            map_to_stage_affine=map_to_2x_stage,
        )[0]

        # Map center is (1000, 2000) in the 2X frame. Full-scan planning
        # translates it into the selected objective's parcentric frame.
        self.assertEqual(ten_x[:2], [1350, 2075])
        self.assertEqual(twenty_x[:2], [1400, 2060])

    def test_nominal_20x_fallback_uses_same_map_axis_direction_as_10x(self):
        true_map = np.zeros((10000, 10000, 3), dtype=np.uint8)
        chip_right_of_center = (5000, 4900, 200, 200)

        ten_x = coordinate_generator.generate_10x_scan_coordinates(
            _HighResolutionApp(),
            [chip_right_of_center],
            10000,
            20000,
            2,
            true_map.copy(),
            (5440, 3648),
        )[0]
        twenty_x = coordinate_generator.generate_20x_scan_coordinates(
            _HighResolutionApp(),
            [chip_right_of_center],
            10000,
            20000,
            2,
            true_map.copy(),
            (5440, 3648),
        )[0]

        # A map point to the right corresponds to decreasing stage X for
        # every objective; only the parcentric offsets differ afterward.
        self.assertLess(ten_x[0], 10000 + 350)
        self.assertLess(twenty_x[0], 10000 + 400)


if __name__ == "__main__":
    unittest.main()
