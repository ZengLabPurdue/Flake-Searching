import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "App"))
from Imaging.image_metadata import save_png
from Scanning.flake_inspection import FlakeInspection, centered_candidate, prepare_positions
from Scanning.scan_manager import ScanManager
from UI.panels.view_scans_panel import ViewScansPanel


class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.scan = Path(self.temp.name)
        self.source = "All Images/10x/wafer 1 (100, 200)/Raw/img_10x_12.png"
        path = self.scan / self.source
        path.parent.mkdir(parents=True)
        save_png(path, np.zeros((100, 100, 3), dtype=np.uint8), {
            "magnification": "10X", "resolution": "HIGH", "stage_z_um": 60000,
        })
        self.record = {
            "classification": "Thin", "source_image": self.source, "magnification": "10X",
            "bounding_box_px": {"x": 45, "y": 45, "width": 10, "height": 10},
            "location": {"stage_position_um": {"x": 110, "y": 220}},
            "statistics": {"area_um2": 10},
        }
        self.manifest = {"detection_model": "Region Detection", "profile_path": "profile",
                         "detections": [self.record]}
        (self.scan / "flakes_found.json").write_text(json.dumps(self.manifest))
        self.region = {"matched_class": "Thin", "area_um2": 10,
                       "bounding_box": {"x": 45, "y": 45, "width": 10, "height": 10},
                       "instance_mask": np.full((10, 10), 255, dtype=np.uint8)}

    def manager(self):
        events = []
        manager = SimpleNamespace()
        manager._stop_event = threading.Event()
        manager._check_cancelled = lambda: None
        manager._wait_for_existing_focus = lambda: None
        manager._wait_for_stage = lambda: None
        manager._move_stage_xy = lambda x, y: events.append(("move", x, y))
        manager._move_stage_z = lambda z: events.append(("z", z))
        manager.stage = SimpleNamespace(
            get_xy_position=lambda **kw: (110, 220),
            get_position=lambda **kw: (110, 220, 60000),
            wait_until_not_busy=lambda: None,
        )
        manager.turret_controller = SimpleNamespace(
            change_objective=lambda position, **kw: events.append(("return", position)),
        )
        manager.mag = "10X"
        def change(position, **kw):
            manager.mag = {2: "10X", 3: "20X", 5: "100X"}[position]
            events.append(("objective", position))
        manager._change_objective = change
        manager.app = SimpleNamespace(
            get_resolution=lambda: "HIGH", get_magnification=lambda: manager.mag,
            focus_controller=SimpleNamespace(start_auto_focus_thread=lambda **kw: events.append(("focus",))),
        )
        def save(image, save_dir, filename, **kw):
            path = Path(save_dir) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            save_png(path, image)
            events.append(("capture", manager.mag, kw.get("vignette_applied")))
            return path
        manager.frame_processor = SimpleNamespace(
            arm_capture_after_motion=lambda position: None,
            capture_frame=lambda **kw: np.full((100, 100, 3), 120, dtype=np.uint8),
            apply_vignette_filter=lambda image: image,
            save_image=save,
        )
        manager.ui_dispatch = lambda callback, *args: callback(*args)
        return manager, events

    def test_positions_are_persisted_with_stable_ids_and_chip_offsets(self):
        first = prepare_positions(self.scan)
        record = first["detections"][0]
        self.assertEqual(record["location"]["chip_relative_um"], {"x": 10, "y": 20})
        self.assertEqual(record["location"]["scan_relative_um"], {"x": 0, "y": 0})
        self.assertEqual(record["flake_id"], prepare_positions(self.scan)["detections"][0]["flake_id"])

    def test_centering_rejects_ambiguous_wrong_class_and_remote_candidates(self):
        centered_candidate([self.region], self.record, (100, 100, 3), 1)
        for candidates in ([self.region, copy.deepcopy(self.region)], [],
                           [dict(self.region, matched_class="Other")],
                           [dict(self.region, bounding_box={"x": 100, "y": 100, "width": 10, "height": 10})]):
            with self.assertRaises(ValueError):
                centered_candidate(candidates, self.record, (100, 100, 3), 1)

    def test_confirm_captures_both_objectives_and_exports_the_pinned_record(self):
        manager, events = self.manager()
        def notify(event, value):
            if event == "ready":
                session.confirm.set()
            elif event == "saved":
                manager._stop_event.set()
        session = FlakeInspection(self.scan, self.record, notify)
        self.record["classification"] = "Changed UI selection"
        def export(scan_path, record):
            self.assertEqual(record["classification"], "Thin")
            self.assertTrue((scan_path / record["image_20x"]).is_file())
            self.assertTrue((scan_path / record["image_100x"]).is_file())
            return {"flake_id": "db_flake", "json": self.scan / "database.json"}
        with patch("Scanning.flake_inspection.contour_finder.region_classifier.load_profile_configuration", return_value={}), \
             patch("Scanning.flake_inspection.contour_finder.find_flakes", return_value=(None, None, {"combined_region_results": [self.region]})), \
             patch("Scanning.flake_inspection.flake_detection.save_database", side_effect=export) as exporter:
            session.run(manager)
        exporter.assert_called_once()
        self.assertEqual([e for e in events if e[0] == "objective"],
                         [("objective", 2), ("objective", 3), ("objective", 5)])
        self.assertIn(("capture", "20X", True), events)
        self.assertIn(("capture", "100X", False), events)
        self.assertEqual(events[-1], ("return", 2))
        saved = json.loads((self.scan / "flakes_found.json").read_text())["detections"][0]
        self.assertEqual(saved["database_export"]["flake_id"], "db_flake")

    def test_leaving_preview_never_exports_or_captures_final_100x(self):
        manager, events = self.manager()
        session = FlakeInspection(self.scan, self.record,
                                  lambda event, value: manager._stop_event.set() if event == "ready" else None)
        with patch("Scanning.flake_inspection.contour_finder.region_classifier.load_profile_configuration", return_value={}), \
             patch("Scanning.flake_inspection.contour_finder.find_flakes", return_value=(None, None, {"combined_region_results": [self.region]})), \
             patch("Scanning.flake_inspection.flake_detection.save_database") as exporter:
            session.run(manager)
        exporter.assert_not_called()
        self.assertNotIn(("capture", "100X", False), events)
        self.assertEqual(events[-1], ("return", 2))

    def test_next_leaves_inspection_before_selecting_another_flake(self):
        panel = ViewScansPanel.__new__(ViewScansPanel)
        panel.inspection = SimpleNamespace()
        view = ["Flake Inspection"]
        panel.app = SimpleNamespace(scan_manager=SimpleNamespace(stop_scan=Mock()),
                                    get_view=lambda: view[0],
                                    set_view=Mock(side_effect=lambda mode: view.__setitem__(0, mode)))
        panel.inspection_panel = SimpleNamespace(grid_remove=Mock())
        panel._queue_panel_resize = Mock()
        panel._update_region_highlight = Mock()
        panel.current_regions = [self.record, dict(self.record, classification="Other")]
        panel.view_region_index = 0
        panel._show_current_region = Mock()
        panel.root = SimpleNamespace(focus_set=Mock())
        panel.next_region()
        self.assertIsNone(panel.inspection)
        self.assertEqual(panel.view_region_index, 1)
        panel._show_current_region.assert_called_once_with(fit_image=True)
        panel.app.scan_manager.stop_scan.assert_called_once()
        panel.app.set_view.assert_called_once_with("Scan Results")

    def test_late_ready_notification_cannot_enable_an_abandoned_inspection(self):
        panel = ViewScansPanel.__new__(ViewScansPanel)
        panel.inspection = None
        panel.inspection_status = Mock()
        panel.confirm_inspection_button = Mock()
        panel._inspection_update(object(), "ready", "Ready")
        panel.inspection_status.set.assert_not_called()
        panel.confirm_inspection_button.state.assert_not_called()

    def test_invalid_profile_does_not_move_the_microscope(self):
        manager, events = self.manager()
        notifications = []
        session = FlakeInspection(self.scan, self.record, lambda *args: notifications.append(args))
        with patch("Scanning.flake_inspection.contour_finder.region_classifier.load_profile_configuration",
                   side_effect=ValueError("Profile missing")):
            session.run(manager)
        self.assertFalse(events)
        self.assertIn(("error", "Profile missing"), notifications)
        self.assertEqual(notifications[-1], ("finished", None))

    def test_ambiguous_centering_waits_for_user_point_before_high_power(self):
        manager, events = self.manager()
        def notify(event, value):
            if event == "center":
                self.assertNotIn(("objective", 3), events)
                session.center_point = (50, 50)
                session.center_selected.set()
            elif event == "ready":
                manager._stop_event.set()
        session = FlakeInspection(self.scan, self.record, notify)
        with patch("Scanning.flake_inspection.contour_finder.region_classifier.load_profile_configuration", return_value={}), \
             patch("Scanning.flake_inspection.contour_finder.find_flakes", return_value=(None, None, {"combined_region_results": []})), \
             patch("Scanning.flake_inspection.flake_detection.save_database") as exporter:
            session.run(manager)
        self.assertEqual(session.record["centering_method"], "user_selected")
        self.assertIn(("objective", 5), events)
        exporter.assert_not_called()

    def test_managed_worker_claims_and_releases_hardware_for_inspection(self):
        manager = ScanManager.__new__(ScanManager)
        manager._scan_thread_lock = threading.Lock()
        manager._scan_thread = None
        manager._scan_execution_thread_id = None
        manager.scan_running = False
        manager._stop_event = threading.Event()
        manager.set_scan_running = Mock()
        manager.ui_dispatch = lambda fn, *args: fn(*args)
        errors = []
        ownership = []
        session = SimpleNamespace(run=lambda m: ownership.append(m.is_scan_worker_thread()))
        worker = manager.start_scan(flake_inspection=session, on_error=errors.append)
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(ownership, [True])
        self.assertEqual(errors, [])
        self.assertEqual([c.args[0] for c in manager.set_scan_running.call_args_list], [True, False])


if __name__ == "__main__":
    unittest.main()
