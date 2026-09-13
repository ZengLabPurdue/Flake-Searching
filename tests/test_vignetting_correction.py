from datetime import datetime
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np


APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from imaging import image_metadata, vignetting_corrector  # noqa: E402
from imaging.flatfield_storage import (  # noqa: E402
    VIGNETTE_SUBSTRATE_LABELS,
    VIGNETTE_SUBSTRATES,
    normalize_vignette_substrate,
    save_vignette_filter,
    vignette_substrate_label,
    vignette_filter_path,
)
from imaging.frame_processing import FrameProcessor  # noqa: E402


class _FakeApp:
    @staticmethod
    def get_magnification():
        return "2X"

    @staticmethod
    def get_resolution():
        return "HIGH"

    @staticmethod
    def get_vignette_substrate():
        return "285nm_8in"


class VignettingCorrectionTests(unittest.TestCase):
    def test_replacing_flatfield_preserves_previous_calibration_in_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            flatfield_dir = root / "active"
            backup_dir = root / "backups"
            active_path = vignette_filter_path(
                "10X",
                "HIGH",
                substrate="100nm",
                flatfield_dir=flatfield_dir,
            )
            active_path.parent.mkdir(parents=True)
            previous = np.full((5, 7, 3), (30, 60, 90), dtype=np.uint8)
            replacement = np.full((5, 7, 3), (80, 100, 120), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(active_path), previous))

            saved_path, backup_path = save_vignette_filter(
                replacement,
                "10X",
                "HIGH",
                substrate="100nm",
                flatfield_dir=flatfield_dir,
                backup_dir=backup_dir,
                timestamp=datetime(2026, 7, 29, 12, 34, 56, 789),
            )

            self.assertEqual(saved_path, active_path)
            self.assertEqual(
                backup_path.name,
                "vignette_filter_10x_100nm_high_20260729_123456_000789.png",
            )
            np.testing.assert_array_equal(
                cv2.imread(str(saved_path), cv2.IMREAD_COLOR),
                replacement,
            )
            np.testing.assert_array_equal(
                cv2.imread(str(backup_path), cv2.IMREAD_COLOR),
                previous,
            )
            self.assertEqual(
                list(flatfield_dir.glob("*.tmp.png")),
                [],
            )

    def test_first_flatfield_save_does_not_create_an_empty_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.full((3, 4, 3), 100, dtype=np.uint8)

            saved_path, backup_path = save_vignette_filter(
                image,
                "2X",
                "100NM_HIGH",
                flatfield_dir=root / "active",
                backup_dir=root / "backups",
            )

            self.assertTrue(saved_path.is_file())
            self.assertIsNone(backup_path)
            self.assertFalse((root / "backups").exists())

    def test_substrate_specific_filename_uses_supported_substrate_names(self):
        self.assertEqual(
            VIGNETTE_SUBSTRATES,
            ("100nm", "285nm_4in", "285nm_8in"),
        )
        self.assertEqual(
            VIGNETTE_SUBSTRATE_LABELS,
            ("100nm", "285nm (4in)", "285nm (8in)"),
        )
        self.assertEqual(
            normalize_vignette_substrate("285nm (4in)"),
            "285nm_4in",
        )
        self.assertEqual(
            vignette_substrate_label("285nm_8in"),
            "285nm (8in)",
        )
        path = vignette_filter_path(
            "10X",
            "HIGH",
            substrate="285nm (4in)",
        )
        self.assertEqual(
            path.name,
            "vignette_filter_10x_285nm_4in_high.png",
        )

    def test_substrate_specific_save_does_not_replace_another_substrate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            flatfield_dir = root / "active"
            four_inch_path = vignette_filter_path(
                "10X",
                "HIGH",
                substrate="285nm_4in",
                flatfield_dir=flatfield_dir,
            )
            eight_inch_path = vignette_filter_path(
                "10X",
                "HIGH",
                substrate="285nm_8in",
                flatfield_dir=flatfield_dir,
            )
            four_inch_path.parent.mkdir(parents=True)
            four_inch = np.full((3, 4, 3), 40, dtype=np.uint8)
            eight_inch = np.full((3, 4, 3), 80, dtype=np.uint8)
            replacement = np.full((3, 4, 3), 120, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(four_inch_path), four_inch))
            self.assertTrue(cv2.imwrite(str(eight_inch_path), eight_inch))

            saved_path, _backup_path = save_vignette_filter(
                replacement,
                "10X",
                "HIGH",
                substrate="285nm_4in",
                flatfield_dir=flatfield_dir,
                backup_dir=root / "backups",
            )

            self.assertEqual(saved_path, four_inch_path)
            np.testing.assert_array_equal(
                cv2.imread(str(saved_path), cv2.IMREAD_COLOR),
                replacement,
            )
            np.testing.assert_array_equal(
                cv2.imread(str(eight_inch_path), cv2.IMREAD_COLOR),
                eight_inch,
            )

    def test_flatfield_filename_rejects_path_traversal(self):
        with self.assertRaises(ValueError):
            vignette_filter_path("../10X", "HIGH")

    def test_flatfield_filename_rejects_unknown_substrate(self):
        with self.assertRaisesRegex(ValueError, "Unknown vignette substrate"):
            vignette_filter_path(
                "10X",
                "HIGH",
                substrate="glass",
            )

    def test_frame_processor_looks_up_the_selected_substrate(self):
        processor = FrameProcessor.__new__(FrameProcessor)
        processor.app = _FakeApp()

        path = processor.get_vignette_filter_path()

        self.assertEqual(
            path.name,
            "vignette_filter_2x_285nm_8in_high.png",
        )

    def test_saved_png_records_the_selected_vignette_substrate(self):
        with tempfile.TemporaryDirectory() as directory:
            processor = FrameProcessor.__new__(FrameProcessor)
            processor.app = _FakeApp()
            image = np.full((3, 4, 3), 100, dtype=np.uint8)

            path = processor.save_image(
                image=image,
                save_dir=directory,
                filename="capture.png",
            )
            metadata = image_metadata.read_png_metadata(path)

            self.assertEqual(
                metadata["vignette_substrate"],
                "285nm_8in",
            )

    def test_per_channel_gain_removes_spatial_color_cast(self):
        flatfield = np.array(
            [
                [[40, 90, 140], [70, 100, 130], [60, 60, 80]],
                [[50, 80, 120], [80, 120, 160], [70, 90, 100]],
                [[30, 70, 100], [60, 100, 150], [50, 80, 90]],
            ],
            dtype=np.uint8,
        )

        gain = vignetting_corrector.create_vignette_gain(
            flatfield,
            reference_point=(1, 1),
            reference_radius=0,
        )
        corrected = vignetting_corrector.apply_vignette_gain(
            flatfield,
            gain,
        )

        expected = np.broadcast_to(
            flatfield[1, 1],
            flatfield.shape,
        )
        self.assertEqual(gain.shape, flatfield.shape)
        np.testing.assert_allclose(
            corrected,
            expected,
            atol=1,
        )

    def test_uniform_flatfield_preserves_existing_color_balance(self):
        flatfield = np.empty((4, 5, 3), dtype=np.uint8)
        flatfield[:] = (80, 120, 160)
        image = np.array(
            [
                [[10, 30, 90], [100, 80, 20]],
                [[25, 50, 75], [200, 150, 100]],
            ],
            dtype=np.uint8,
        )
        image = np.tile(image, (2, 3, 1))[:4, :5]

        gain = vignetting_corrector.create_vignette_gain(
            flatfield,
            reference_point=(2, 2),
        )
        corrected = vignetting_corrector.apply_vignette_gain(
            image,
            gain,
        )

        np.testing.assert_array_equal(corrected, image)

    def test_legacy_scalar_gain_remains_supported(self):
        image = np.array(
            [
                [[10, 20, 30], [20, 40, 60]],
                [[30, 60, 90], [40, 80, 120]],
            ],
            dtype=np.uint8,
        )
        gain = np.array(
            [
                [2.0, 0.5],
                [1.0, 1.5],
            ],
            dtype=np.float32,
        )

        corrected = vignetting_corrector.apply_vignette_gain(
            image,
            gain,
        )

        expected = np.clip(
            np.rint(image.astype(np.float32) * gain[:, :, None]),
            0,
            255,
        ).astype(np.uint8)
        np.testing.assert_array_equal(corrected, expected)

    def test_frame_processor_crops_a_color_gain_to_a_cropped_frame(self):
        processor = FrameProcessor.__new__(FrameProcessor)
        processor.app = _FakeApp()
        processor.vignette_gain = np.ones(
            (10, 10, 3),
            dtype=np.float32,
        )
        processor.load_vignette_filter = lambda _shape=None: None
        image = np.full((6, 6, 3), (20, 40, 80), dtype=np.uint8)

        corrected = processor.apply_vignette_filter(image)

        np.testing.assert_array_equal(corrected, image)


if __name__ == "__main__":
    unittest.main()
