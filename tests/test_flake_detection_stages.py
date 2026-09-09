from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import KeyEvent


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY_ROOT / "Flake Recognition" / "flake_detection_stages.py"
SPEC = spec_from_file_location("flake_detection_stages", MODULE_PATH)
flake_detection_stages = module_from_spec(SPEC)
SPEC.loader.exec_module(flake_detection_stages)


class FlakeDetectionStageViewerTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    @staticmethod
    def _details(height=8, width=10):
        binary = lambda: np.zeros((height, width), dtype=np.uint8)
        rgb = lambda: np.zeros((height, width, 3), dtype=np.uint8)
        return {
            "image_rgb": rgb(),
            "closed_edges": binary(),
            "external_edge_mask": binary(),
            "internal_edge_mask": binary(),
            "contour_img": rgb(),
            "all_external_mask": binary(),
            "unfiltered_external_mask": binary(),
            "external_mask": binary(),
            "all_internal_mask": binary(),
            "internal_mask": binary(),
            "matched_region_mask": binary(),
            "classified_image_without_legend": rgb(),
            "internal_contours": [
                np.asarray([[[2, 2]], [[3, 2]], [[3, 3]]], dtype=np.int32),
            ],
            "region_results": [],
            "combined_region_results": [],
        }

    def test_stage_panels_separate_external_and_internal_contour_lines(self):
        details = self._details()
        details["external_edge_mask"][1, 2] = 255
        details["internal_edge_mask"][5, 7] = 255
        details["image_rgb"][:] = (10, 20, 30)
        details["all_external_mask"][3, 4] = 255
        # This later classification mask must not drive the contour panel.
        details["internal_mask"][1, 2] = 255

        panels = flake_detection_stages.build_stage_panels(
            details,
            external_contour_count=2,
        )
        panels_by_key = {panel["key"]: panel for panel in panels}
        external_image = panels_by_key["external_contours"]["image"]
        internal_image = panels_by_key["internal_contours"]["image"]

        self.assertEqual(len(panels), 12)
        self.assertIs(external_image, details["external_edge_mask"])
        self.assertIs(internal_image, details["internal_edge_mask"])
        self.assertEqual(external_image[1, 2], 255)
        self.assertEqual(external_image[5, 7], 0)
        self.assertEqual(internal_image[5, 7], 255)
        self.assertEqual(internal_image[1, 2], 0)
        self.assertTrue(np.allclose(
            panels_by_key["external_contours"]["cmap"](1.0)[:3],
            np.asarray(flake_detection_stages.EXTERNAL_CONTOUR_RGB) / 255,
        ))
        self.assertTrue(np.allclose(
            panels_by_key["internal_contours"]["cmap"](1.0)[:3],
            np.asarray(flake_detection_stages.INTERNAL_CONTOUR_RGB) / 255,
        ))
        self.assertIn("(2)", panels_by_key["external_contours"]["title"])
        self.assertIn("(1)", panels_by_key["internal_contours"]["title"])
        masked_background = panels_by_key["masked_background"]
        self.assertIs(masked_background["image"], details["image_rgb"])
        self.assertIs(
            masked_background["overlay_mask"],
            details["all_external_mask"],
        )
        detail_sources = {
            "external_candidates_before_filters": "unfiltered_external_mask",
            "external_candidates_after_filters": "external_mask",
            "internal_matches_before_filters": "all_internal_mask",
            "accepted_internal_regions": "internal_mask",
            "accepted_matched_regions": "matched_region_mask",
        }
        for panel_key, detail_key in detail_sources.items():
            self.assertIs(panels_by_key[panel_key]["image"], details[detail_key])

    def test_stage_figure_contains_the_current_twelve_stage_grid(self):
        details = self._details()

        figure = flake_detection_stages.create_stage_figure(
            details,
            external_contour_count=1,
            image_name="sample.png",
            profile_name="Test Profile",
            pixel_size_um=0.5,
            pixel_size_source="PNG metadata",
        )
        titles = [axis.get_title() for axis in figure.axes]

        self.assertEqual(len(figure.axes), 12)
        self.assertTrue(any("External Contours" in title for title in titles))
        self.assertTrue(any("Internal Contours" in title for title in titles))
        self.assertTrue(any("Final Pixel Classification" in title for title in titles))
        self.assertIn("sample.png", figure._suptitle.get_text())
        self.assertIn("Test Profile", figure._suptitle.get_text())
        self.assertIn("PNG metadata", figure._suptitle.get_text())

    def test_optional_navigator_moves_one_full_size_stage_at_a_time(self):
        details = self._details()
        figure = flake_detection_stages.create_stage_navigator_figure(
            details,
            external_contour_count=1,
            image_name="sample.png",
        )
        controller = figure._stage_navigator

        self.assertIs(controller.figure, figure)
        self.assertEqual(controller.index, 0)
        self.assertEqual(controller.current_panel["key"], "original")
        self.assertEqual(len(controller.axis.images), 1)
        self.assertIn("Stage 1 of 12", controller.axis.get_title())

        controller.next()
        self.assertEqual(controller.index, 1)
        self.assertEqual(controller.current_panel["key"], "closed_edges")
        controller.previous()
        controller.previous()
        self.assertEqual(controller.index, 0)

        right = KeyEvent("key_press_event", figure.canvas, key="right")
        figure.canvas.callbacks.process("key_press_event", right)
        self.assertEqual(controller.index, 1)

        controller.show_stage(5)
        self.assertEqual(controller.current_panel["key"], "masked_background")
        self.assertEqual(len(controller.axis.images), 2)
        controller.next()
        self.assertEqual(len(controller.axis.images), 1)

        controller.show_stage(11)
        controller.next()
        self.assertEqual(controller.index, 11)
        self.assertIn("Final Pixel Classification", controller.axis.get_title())

    def test_closing_each_full_size_stage_advances_through_all_panels(self):
        details = self._details()
        details["image_rgb"][:] = (10, 20, 30)
        panels = flake_detection_stages.build_stage_panels(
            details,
            external_contour_count=1,
        )
        first_figure = flake_detection_stages.create_single_stage_figure(
            panels[0],
            0,
            len(panels),
            image_name="sample.png",
        )

        self.assertEqual(len(first_figure.axes), 1)
        self.assertEqual(first_figure.axes[0].get_title(), "")
        self.assertIsNone(first_figure._suptitle)
        self.assertEqual(first_figure.texts, [])
        default_filename = first_figure.canvas.get_default_filename()
        self.assertIn("sample", default_filename)
        self.assertIn("01", default_filename)
        self.assertIn("Original", default_filename)
        with tempfile.TemporaryDirectory() as directory:
            saved_stage = Path(directory) / "stage.png"
            first_figure.savefig(saved_stage, dpi=40)
            saved_rgb = cv2.cvtColor(
                cv2.imread(str(saved_stage), cv2.IMREAD_COLOR),
                cv2.COLOR_BGR2RGB,
            )
            self.assertTrue(np.all(saved_rgb == (10, 20, 30)))
        with (
            patch.object(flake_detection_stages.plt, "show") as show,
            patch("builtins.print"),
        ):
            flake_detection_stages.show_stage_figures_sequentially(
                details,
                external_contour_count=1,
                image_name="sample.png",
                first_figure=first_figure,
            )

        self.assertEqual(show.call_count, 12)
        self.assertTrue(all(
            call.kwargs == {"block": True}
            for call in show.call_args_list
        ))
        self.assertNotIn(first_figure.number, plt.get_fignums())

    def test_pixel_size_reports_override_and_missing_metadata_fallback(self):
        override_value, override_source = (
            flake_detection_stages._pixel_size_for_image(
                Path("unused.png"),
                override=0.25,
            )
        )
        fallback_value, fallback_source = (
            flake_detection_stages._pixel_size_for_image(
                Path("untagged.jpg"),
            )
        )

        self.assertEqual(override_value, 0.25)
        self.assertEqual(override_source, "command-line override")
        self.assertEqual(
            fallback_value,
            flake_detection_stages.DEFAULT_PIXEL_SIZE_UM,
        )
        self.assertEqual(fallback_source, "10X HIGH fallback")

    def test_viewer_delegates_to_the_production_pipeline_without_nested_display(self):
        image_bgr = np.full((8, 10, 3), 127, dtype=np.uint8)
        details = self._details()
        external_contours = [
            np.asarray([[[1, 1]], [[8, 1]], [[8, 6]]], dtype=np.int32),
        ]
        profile_path = Path("example-profile")

        with tempfile.TemporaryDirectory() as directory:
            save_path = Path(directory) / "stages.png"
            with (
                patch.object(
                    flake_detection_stages._app_contour_finder,
                    "find_flakes",
                    return_value=(
                        details["classified_image_without_legend"],
                        external_contours,
                        details,
                    ),
                ) as find_flakes,
                patch.object(flake_detection_stages.plt, "show") as show,
            ):
                figure, returned_details = (
                    flake_detection_stages.visualize_detection_stages(
                        image_bgr,
                        profile_path=profile_path,
                        edge_threshold=12,
                        area_threshold=345,
                        contrast_threshold=4,
                        region_flood_fill_threshold=11,
                        pixel_size_um=0.25,
                        display=False,
                        save_path=save_path,
                    )
                )
            self.assertTrue(save_path.is_file())

        call_args, call_kwargs = find_flakes.call_args
        self.assertIs(call_args[0], image_bgr)
        self.assertFalse(call_kwargs["display"])
        self.assertTrue(call_kwargs["return_details"])
        self.assertEqual(call_kwargs["profile_path"], profile_path)
        self.assertEqual(call_kwargs["edge_threshold"], 12)
        self.assertEqual(call_kwargs["area_threshold"], 345)
        self.assertEqual(call_kwargs["contrast_threshold"], 4)
        self.assertEqual(call_kwargs["region_flood_fill_threshold"], 11)
        self.assertFalse(call_kwargs["draw_legend"])
        self.assertIs(returned_details, details)
        self.assertIs(returned_details["external_contours"], external_contours)
        show.assert_not_called()
        self.assertEqual(len(figure.axes), 1)
        self.assertEqual(figure.axes[0].get_title(), "")
        self.assertIsNone(figure._suptitle)
        self.assertEqual(figure.texts, [])

    def test_nested_shape_produces_both_current_contour_types(self):
        image_bgr = np.full((100, 100, 3), 200, dtype=np.uint8)
        cv2.rectangle(image_bgr, (10, 10), (89, 89), (80, 80, 80), -1)
        cv2.rectangle(image_bgr, (30, 30), (69, 69), (200, 200, 200), -1)

        figure, details = flake_detection_stages.visualize_detection_stages(
            image_bgr,
            edge_threshold=10,
            area_threshold=20,
            display=False,
        )

        self.assertGreater(np.count_nonzero(details["external_edge_mask"]), 0)
        self.assertGreater(np.count_nonzero(details["internal_edge_mask"]), 0)
        panel_keys = {
            panel["key"]
            for panel in flake_detection_stages.build_stage_panels(
                details,
                external_contour_count=len(details["external_contours"]),
            )
        }
        self.assertIn("external_contours", panel_keys)
        self.assertIn("internal_contours", panel_keys)


if __name__ == "__main__":
    unittest.main()
