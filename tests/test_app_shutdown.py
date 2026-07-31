from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPOSITORY_ROOT / "App"
sys.path.insert(0, str(APP_DIR))
MODULE_PATH = APP_DIR / "app.py"
SPEC = spec_from_file_location("flake_search_app_shutdown", MODULE_PATH)
app_module = module_from_spec(SPEC)
with patch.object(sys, "argv", ["app.py", "1", "2"]):
    SPEC.loader.exec_module(app_module)


class AppShutdownTests(unittest.TestCase):
    def test_close_focuses_objective_one_before_disconnecting_hardware(self):
        events = []

        class Root:
            @staticmethod
            def after_cancel(_job):
                return None

            @staticmethod
            def destroy():
                events.append("destroy")

        class ScanManager:
            @staticmethod
            def is_scan_running():
                return False

        class FocusController:
            focus_thread = None

        class TurretController:
            @staticmethod
            def return_to_first_objective_and_focus():
                events.append("return-and-focus")

            @staticmethod
            def close():
                events.append("turret-close")

        class ViewScansPanel:
            @staticmethod
            def shutdown():
                events.append("viewer-close")

        class FrameProcessor:
            @staticmethod
            def close():
                events.append("camera-close")

        class Stage:
            @staticmethod
            def disconnect():
                events.append("stage-close")

        app = app_module.App.__new__(app_module.App)
        app.root = Root()
        app.scan_manager = ScanManager()
        app.focus_controller = FocusController()
        app.turret_controller = TurretController()
        app.view_scans_panel = ViewScansPanel()
        app.frame_processor = FrameProcessor()
        app.stage = Stage()
        app._closing = False
        app._shutdown_requested = False
        app._focus_shutdown_started_at = None
        app._shutdown_returning_to_objective = False
        app._shutdown_focus_prepared = False
        app._ui_poll_job = None
        app._map_render_job = None

        app.on_close()

        self.assertEqual(
            events,
            [
                "return-and-focus",
                "viewer-close",
                "camera-close",
                "turret-close",
                "stage-close",
                "destroy",
            ],
        )
        self.assertTrue(app._closing)
        self.assertTrue(app._shutdown_focus_prepared)


if __name__ == "__main__":
    unittest.main()
