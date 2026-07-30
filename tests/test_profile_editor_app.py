from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import tkinter as tk
import unittest


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


if __name__ == "__main__":
    unittest.main()
