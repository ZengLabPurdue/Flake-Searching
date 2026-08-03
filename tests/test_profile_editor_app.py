from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import tkinter as tk
import unittest

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY_ROOT / "Profile Editor" / "profile_editor_app.py"
SPEC = spec_from_file_location("profile_editor_app", MODULE_PATH)
profile_editor_app = module_from_spec(SPEC)
SPEC.loader.exec_module(profile_editor_app)


class ProfileEditorAppTests(unittest.TestCase):
    def test_standalone_editor_uses_the_shared_profile_directory(self):
        profile = profile_editor_app.ScanProfile()

        self.assertEqual(
            profile.profiles_dir.resolve(),
            (REPOSITORY_ROOT / "App" / "Profiles").resolve(),
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
        self.assertIn("P:A: 1.120 1/µm", statistics)

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
