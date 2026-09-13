from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
import tempfile
import unittest


os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY_ROOT / "tools" / "contrast_viewer.py"
SPEC = spec_from_file_location("contrast_viewer_contour_finder", MODULE_PATH)
contour_finder = module_from_spec(SPEC)
SPEC.loader.exec_module(contour_finder)


class ContourContrastViewerTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_range_volume_is_a_transparent_box(self):
        figure = plt.figure()
        axis = figure.add_subplot(111, projection="3d")

        corners = contour_finder._plot_contrast_range_box(
            axis,
            np.asarray((-20, -10, 0), dtype=float),
            np.asarray((5, 15, 25), dtype=float),
            (0.2, 0.4, 0.6),
        )
        figure.canvas.draw()

        volumes = [
            collection
            for collection in axis.collections
            if isinstance(collection, Poly3DCollection)
        ]
        self.assertEqual(corners.shape, (8, 3))
        self.assertEqual(len(volumes), 1)
        self.assertTrue(np.allclose(
            volumes[0].get_facecolors()[0],
            (0.2, 0.4, 0.6, contour_finder.RANGE_BOX_ALPHA),
        ))
        self.assertTrue(np.allclose(
            volumes[0].get_edgecolors()[0],
            (0.2, 0.4, 0.6, contour_finder.RANGE_BOX_EDGE_ALPHA),
        ))
        self.assertTrue(np.allclose(
            volumes[0].get_linewidths(),
            contour_finder.REFERENCE_VOLUME_LINE_WIDTH,
        ))

    def test_profile_discovery_lists_profile_directories_in_name_order(self):
        with tempfile.TemporaryDirectory() as directory:
            profiles_directory = Path(directory)
            for profile_name in ("Zulu", "alpha"):
                profile_directory = profiles_directory / profile_name
                profile_directory.mkdir()
                (profile_directory / "profile.json").write_text(
                    "{}",
                    encoding="utf-8",
                )
            (profiles_directory / "ignored.json").write_text(
                "{}",
                encoding="utf-8",
            )

            profiles = contour_finder._discover_profile_paths(
                profiles_directory,
            )

        self.assertEqual(
            [profile.parent.name for profile in profiles],
            ["alpha", "Zulu"],
        )

    def test_sample_reference_uses_constant_sphere_and_connected_label(self):
        center = np.asarray((-20, -10, 0), dtype=float)
        class_color = np.asarray((0.2, 0.4, 0.6), dtype=float)
        configuration = {
            "classes": [{
                "name": "Thin",
                "group": "Graphene",
                "identify": True,
                "classification_type": "sample",
                "contrast_rgb": center,
                "tolerance": 6,
                "display_color_rgb": class_color,
            }],
        }

        figure = contour_finder.plot_contour_contrasts_3d(
            [],
            profile_configuration=configuration,
            display=False,
        )
        axis = figure.axes[0]
        figure.canvas.draw()
        spheres = [
            collection
            for collection in axis.collections
            if isinstance(collection, Poly3DCollection)
        ]

        self.assertEqual(len(spheres), 1)
        self.assertTrue(np.allclose(
            spheres[0].get_facecolors()[0],
            (*class_color, contour_finder.REFERENCE_SPHERE_ALPHA),
        ))
        self.assertTrue(np.allclose(
            spheres[0].get_linewidths(),
            contour_finder.REFERENCE_VOLUME_LINE_WIDTH,
        ))
        self.assertEqual(
            [text.get_text() for text in axis.texts],
            ["Graphene / Thin"],
        )
        self.assertNotIn("[", axis.texts[0].get_text())
        self.assertEqual(len(axis.lines), 1)
        line_x, line_y, line_z = axis.lines[0].get_data_3d()
        line_start = np.asarray((line_x[0], line_y[0], line_z[0]))
        line_end = np.asarray((line_x[1], line_y[1], line_z[1]))
        self.assertAlmostEqual(
            np.linalg.norm(line_start - center),
            contour_finder.REFERENCE_SPHERE_RADIUS,
        )
        self.assertGreater(
            np.linalg.norm(line_end - center),
            contour_finder.REFERENCE_SPHERE_RADIUS,
        )

    def test_matched_regions_keep_normal_measured_color_markers(self):
        regions = [
            {
                "contrast_rgb": (-20, -10, 0),
                "average_color_rgb": (255, 0, 0),
                "matched_class": "Matched",
            },
            {
                "contrast_rgb": (5, 10, 15),
                "average_color_rgb": (0, 255, 0),
                "matched_class": None,
            },
        ]

        figure = contour_finder.plot_contour_contrasts_3d(
            regions,
            display=False,
        )
        axis = figure.axes[0]
        region_points = axis.collections[0]
        figure.canvas.draw()
        colors = region_points.get_facecolors()

        self.assertEqual(len(region_points.get_paths()), 1)
        self.assertTrue(np.allclose(
            region_points.get_linewidths(),
            contour_finder.REGION_POINT_EDGE_WIDTH,
        ))
        self.assertTrue(np.allclose(
            colors[:, :3],
            np.asarray(((1, 0, 0), (0, 1, 0))),
        ))


if __name__ == "__main__":
    unittest.main()
