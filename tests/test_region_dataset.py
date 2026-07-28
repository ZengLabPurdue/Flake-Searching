from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np


APP_DIR = Path(__file__).resolve().parents[1] / "App"
sys.path.insert(0, str(APP_DIR))

from Scanning.region_dataset import (  # noqa: E402
    NEW_FLAKE_FILTER_DIR,
    build_region_training_image,
    save_region_training_images,
)


class RegionDatasetTests(unittest.TestCase):
    def test_default_dataset_folder_is_the_existing_labeled_data_folder(self):
        self.assertEqual(
            NEW_FLAKE_FILTER_DIR,
            APP_DIR.parent / "Labeled Data" / "New Flake Filter",
        )

    @staticmethod
    def _detection(x=300, y=300, width=40, height=30):
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[4:-4, 5:-5] = 255
        return {
            "bounding_box": {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            },
            "instance_mask": mask,
        }

    def test_training_image_places_rgb_left_and_binary_mask_right(self):
        image = np.zeros((800, 900, 3), dtype=np.uint8)
        image[:, :] = (25, 80, 140)

        combined = build_region_training_image(image, self._detection())

        self.assertEqual(combined.shape, (512, 1024, 3))
        self.assertTrue(np.any(np.all(combined[:, :512] == (25, 80, 140), axis=2)))
        mask_values = np.unique(combined[:, 512:])
        self.assertEqual(mask_values.tolist(), [0, 255])
        self.assertTrue(np.array_equal(
            combined[:, 512:, 0],
            combined[:, 512:, 1],
        ))

    def test_large_region_is_preserved_when_standardized_to_512(self):
        image = np.full((900, 900, 3), 100, dtype=np.uint8)
        detection = self._detection(
            x=100,
            y=120,
            width=600,
            height=550,
        )

        combined = build_region_training_image(image, detection)

        mask = combined[:, 512:, 0]
        ys, xs = np.nonzero(mask)
        self.assertGreater(len(xs), 0)
        self.assertGreater(xs.max() - xs.min(), 300)
        self.assertGreater(ys.max() - ys.min(), 250)

    def test_save_uses_one_png_per_region_without_sidecars(self):
        image = np.full((700, 700, 3), 90, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            paths = save_region_training_images(
                image,
                [self._detection(), self._detection(x=400, y=400)],
                directory,
                "img_10x_7.png",
            )

            self.assertEqual(len(paths), 2)
            self.assertEqual(
                sorted(path.name for path in Path(directory).iterdir()),
                [
                    "img_10x_7_region_001.png",
                    "img_10x_7_region_002.png",
                ],
            )
            loaded = cv2.imread(str(paths[0]))
            self.assertEqual(loaded.shape, (512, 1024, 3))


if __name__ == "__main__":
    unittest.main()
