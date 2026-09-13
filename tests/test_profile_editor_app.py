from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY_ROOT / "profile_editor" / "profile_editor_app.py"
SPEC = spec_from_file_location("profile_editor_app", MODULE_PATH)
profile_editor_app = module_from_spec(SPEC)
SPEC.loader.exec_module(profile_editor_app)
from config import PIXEL_SIZE
from imaging.image_metadata import save_png
from scanning.contour_finder import find_flakes


class ProfileEditorAppTests(unittest.TestCase):
    def test_saved_image_calibration_controls_size_filter_like_scan(self):
        panel = object.__new__(profile_editor_app.ScanProfilePanel)
        panel.app = SimpleNamespace(get_magnification=lambda: "10X", get_resolution=lambda: "LOW")
        panel.scan_profile = SimpleNamespace(filters=[])
        image = np.full((100, 100, 3), 180, dtype=np.uint8)
        image[30:70, 30:70] = 150
        configuration = {
            "classes": [{"name": "Flake", "classification_type": "sample", "class_index": 0,
                         "identify": True, "group": "", "contrast_rgb": np.array([-30., -30., -30.]),
                         "tolerance": 5}],
            "filters": [{"id": "size", "name": "Small flakes", "type": "size", "enabled": True,
                         "minimum_um": None, "maximum_um": 100}],
        }
        with tempfile.TemporaryDirectory() as directory:
            panel.preview_source_path = Path(directory) / "cropped_scan.png"
            save_png(panel.preview_source_path, image, {"magnification": "20X", "resolution": "HIGH"})
            actual = panel._preview_pixel_size()
            self.assertEqual(actual, PIXEL_SIZE["20X"]["HIGH"])
            self.assertIn("image metadata", panel._preview_calibration_description())
            common = dict(return_details=True, draw_legend=False, color_seed=0,
                          profile_configuration=configuration)
            _, _, preview = find_flakes(image, pixel_size_um=actual, **common)
            _, _, scan = find_flakes(image, pixel_size_um=PIXEL_SIZE["20X"]["HIGH"], **common)
            _, _, old_preview = find_flakes(image, pixel_size_um=PIXEL_SIZE["10X"]["LOW"], **common)
        self.assertTrue(preview["combined_region_results"])
        self.assertFalse(old_preview["combined_region_results"])
        self.assertEqual(preview["combined_region_results"][0]["area_um2"],
                         scan["combined_region_results"][0]["area_um2"])

    def test_untagged_image_uses_manual_preview_calibration(self):
        panel = object.__new__(profile_editor_app.ScanProfilePanel)
        panel.app = SimpleNamespace(get_magnification=lambda: "10X", get_resolution=lambda: "MED")
        panel.preview_source_path = None
        self.assertEqual(panel._preview_pixel_size(), PIXEL_SIZE["10X"]["MED"])

    def test_invalid_saved_calibration_is_not_silently_replaced_by_menu(self):
        panel = object.__new__(profile_editor_app.ScanProfilePanel)
        panel.app = SimpleNamespace(get_magnification=lambda: "10X", get_resolution=lambda: "HIGH")
        with tempfile.TemporaryDirectory() as directory:
            panel.preview_source_path = Path(directory) / "invalid.png"
            save_png(panel.preview_source_path, np.zeros((10, 10, 3), dtype=np.uint8),
                     {"magnification": "UNKNOWN", "resolution": "HIGH"})
            with self.assertRaises(ValueError):
                panel._preview_pixel_size()

    def test_standalone_editor_uses_the_shared_profile_directory(self):
        profile = profile_editor_app.ScanProfile()

        self.assertEqual(
            profile.profiles_dir.resolve(),
            (REPOSITORY_ROOT / "app" / "Profiles").resolve(),
        )

    def test_display_click_maps_back_to_the_source_image(self):
        bounds = (100, 50, 400, 200, 1600, 800)

        self.assertEqual(
            profile_editor_app.display_to_source_point(bounds, 300, 150),
            (800, 400),
        )
        self.assertEqual(
            profile_editor_app.display_to_source_point(bounds, 499, 249),
            (1596, 796),
        )
        self.assertIsNone(
            profile_editor_app.display_to_source_point(bounds, 99, 150),
        )
        self.assertIsNone(
            profile_editor_app.display_to_source_point(bounds, 300, 250),
        )

    def test_profile_panel_shell_contract_is_present(self):
        required_methods = (
            "close_all_panels",
            "display_image",
            "display_image_message",
            "display_to_image_point",
            "get_magnification",
            "get_resolution",
            "get_view",
            "set_active_scan_profile",
            "set_view",
        )

        for method_name in required_methods:
            with self.subTest(method=method_name):
                self.assertTrue(
                    callable(
                        getattr(
                            profile_editor_app.ProfileEditorApp,
                            method_name,
                            None,
                        )
                    )
                )

    def test_standalone_toolbar_was_removed(self):
        self.assertFalse(
            hasattr(profile_editor_app.ProfileEditorApp, "_build_toolbar")
        )

    def test_preview_focus_colors_only_the_exact_selected_shape(self):
        panel_type = profile_editor_app.ScanProfilePanel
        image = np.full((120, 160, 3), 255, dtype=np.uint8)
        image[10:20, 10:20] = (90, 80, 70)
        image[45:55, 60:80] = (12, 34, 56)
        instance_mask = np.zeros((10, 20), dtype=np.uint8)
        instance_mask[2:8, 3:17] = 255
        class_index_map = np.zeros(image.shape[:2], dtype=np.int32)
        class_index_map[10:20, 10:20] = 1
        class_index_map[45:55, 60:80] = 2
        region = {
            "matched_class": "Graphene",
            "matched_group": "Thin",
            "area_um2": 12.5,
            "size_um": 5.25,
            "width_um": 4.0,
            "height_um": 3.0,
            "perimeter_um": 14.0,
            "perimeter_area_ratio_per_um": 1.12,
            "bounding_box": {"x": 60, "y": 45, "width": 20, "height": 10},
            "instance_mask": instance_mask,
            "matched_color_rgb": (12, 34, 56),
        }

        highlighted = panel_type._isolate_preview_region(
            image,
            region,
            {"class_index_map": class_index_map},
        )
        statistics = panel_type._format_preview_region_stats(region)

        self.assertEqual(highlighted.shape, image.shape)
        self.assertTrue(np.all(image[10:20, 10:20] == (90, 80, 70)))
        self.assertTrue(np.all(highlighted[10:20, 10:20] == 255))
        selected = highlighted[45:55, 60:80]
        self.assertTrue(np.all(selected[instance_mask > 0] == (12, 34, 56)))
        self.assertTrue(np.all(selected[instance_mask == 0] == 255))
        self.assertFalse(np.any(np.all(highlighted == (255, 0, 255), axis=2)))
        self.assertIn("Class: Graphene", statistics)
        self.assertIn("Area: 12.500 µm²", statistics)
        self.assertIn("Longest side: 5.250 µm", statistics)
        self.assertIn("P:A: 1.120 1/µm", statistics)

    def test_preview_export_saves_the_rendered_png_with_selected_options(self):
        panel_type = profile_editor_app.ScanProfilePanel
        panel = object.__new__(panel_type)
        panel.preview_result_rgb = np.full(
            (8, 10, 3),
            (12, 34, 56),
            dtype=np.uint8,
        )
        panel.preview_regions = []
        panel.preview_region_index = None
        panel.preview_details = {}
        panel.preview_source_path = Path("source.png")
        panel.scan_profile = type("_Profile", (), {"name": "Test Profile"})()
        panel.profile_name_var = type(
            "_Name",
            (),
            {"get": lambda _self: "Test Profile"},
        )()
        statuses = []
        panel.status_var = type(
            "_Status",
            (),
            {"set": lambda _self, value: statuses.append(value)},
        )()
        dialog = type(
            "_Dialog",
            (),
            {"destroy": lambda self: setattr(self, "destroyed", True)},
        )()
        panel_globals = panel_type._save_preview_image.__globals__

        with tempfile.TemporaryDirectory() as directory:
            save_path = Path(directory) / "preview.png"
            with patch.object(
                panel_globals["filedialog"],
                "asksaveasfilename",
                return_value=str(save_path),
            ):
                panel._save_preview_image(
                    dialog,
                    show_all=True,
                    show_area=False,
                    show_perimeter_area_ratio=False,
                    show_legend=False,
                )

            self.assertTrue(save_path.is_file())
            metadata = panel_globals["image_metadata"].read_png_metadata(save_path)
            self.assertEqual(metadata["profile_name"], "Test Profile")
            self.assertEqual(metadata["display_all_regions"], "true")
            self.assertEqual(metadata["display_area"], "false")
            self.assertEqual(metadata["display_perimeter_area_ratio"], "false")
            self.assertEqual(metadata["display_legend"], "false")
        self.assertTrue(dialog.destroyed)
        self.assertIn("Saved profile preview", statuses[-1])

    def test_filter_bound_parser_accepts_open_inclusive_ranges(self):
        class StubVariable:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        panel_type = profile_editor_app.ScanProfilePanel
        panel = object.__new__(panel_type)
        panel.filter_bound_vars = {
            "minimum_um": StubVariable("2.5"),
            "maximum_um": StubVariable(""),
        }

        self.assertEqual(
            panel._get_filter_bounds(
                panel_type.FILTER_LABEL_TO_TYPE["Size"],
            ),
            {"minimum_um": 2.5, "maximum_um": None},
        )
        self.assertEqual(
            panel._get_filter_bounds(
                panel_type.FILTER_LABEL_TO_TYPE["Neural Network Flake Filter"],
            ),
            {},
        )

        panel.filter_bound_vars["maximum_um"] = StubVariable("1")
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            panel._get_filter_bounds(panel_type.FILTER_LABEL_TO_TYPE["Size"])

    def test_shared_panel_exposes_contrast_range_controls(self):
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk is unavailable: {exc}")
        root.withdraw()
        try:
            app = profile_editor_app.ProfileEditorApp(root)
            root.update_idletasks()
            panel = app.scan_profile_panel
            self.assertTrue(hasattr(panel, "preview_legend_check"))
            self.assertTrue(hasattr(panel, "save_preview_button"))
            self.assertTrue(panel.save_preview_button.instate(["disabled"]))
            self.assertTrue(panel.preview_legend_var.get())
            panel.new_class()
            panel.classification_type_var.set("Contrast range")
            panel._on_classification_type_changed()
            panel._set_range_values((-20, -10, 0), (5, 15, 25))

            self.assertTrue(panel.range_options.winfo_manager())
            self.assertFalse(panel.region_options.winfo_manager())
            self.assertEqual(
                panel._get_contrast_range(),
                ((-20, -10, 0), (5, 15, 25)),
            )
        finally:
            root.destroy()

    def test_shared_panel_exposes_geometry_filters_and_preview_menu(self):
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk is unavailable: {exc}")
        root.withdraw()
        try:
            app = profile_editor_app.ProfileEditorApp(root)
            root.update_idletasks()
            menu = root.nametowidget(root.cget("menu"))
            menu_labels = [
                menu.entrycget(index, "label")
                for index in range(menu.index("end") + 1)
            ]
            panel = app.scan_profile_panel
            panel.new_filter()
            panel.filter_bound_vars["minimum_um"].set("2.5")
            panel.filter_bound_vars["maximum_um"].set("12")

            self.assertIn("Preview", menu_labels)
            self.assertTrue(panel.filter_options.winfo_manager())
            self.assertEqual(
                panel._get_filter_bounds(
                    profile_editor_app.ScanProfilePanel.FILTER_LABEL_TO_TYPE["Size"]
                ),
                {"minimum_um": 2.5, "maximum_um": 12.0},
            )
            self.assertFalse(hasattr(panel, "profile_minimum_size_var"))
            self.assertFalse(hasattr(panel, "minimum_size_var"))

            panel.filter_type_var.set("Neural Network Flake Filter")
            panel.filter_enabled_var.set(True)
            panel._on_filter_type_changed()
            self.assertFalse(panel.filter_enabled_var.get())
            self.assertTrue(panel.filter_enabled_check.instate(["disabled"]))
            self.assertTrue(panel.neural_filter_options.winfo_manager())
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
