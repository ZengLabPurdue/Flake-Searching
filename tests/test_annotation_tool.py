import copy
import json
from pathlib import Path
import sys
import tempfile
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.annotation_tool import Project, discover_images, candidate_filters, identified_region_preview
from imaging.image_metadata import save_png


class AnnotationProjectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.profile = self.root / "source-profile.json"
        self.profile_data = {
            "schema": "flake-search.scan-search-profile", "version": 7,
            "classes": [{"name": "Flake", "classification_type": "range", "identify": True,
                         "contrast_range_rgb": {"minimum": [-100, -100, -100], "maximum": [-10, -10, -10]}}],
            "filters": [
                {"id": "reject-pa", "name": "P:A", "type": "perimeter_area_ratio", "enabled": True, "minimum_per_um": None, "maximum_per_um": 0.000001},
                {"id": "reject-area", "name": "Area", "type": "size", "enabled": True, "minimum_um": 100000, "maximum_um": None},
                {"id": "future-model", "name": "ML", "type": "neural_network", "enabled": True},
            ],
        }
        self.profile.write_text(json.dumps(self.profile_data), encoding="utf-8")
        self.source = self.root / "raw.png"
        image = np.full((160, 180, 3), 180, np.uint8)
        image[40:115, 50:130] = 130
        save_png(self.source, image, {"magnification": "10X", "resolution": "HIGH", "vignette_applied": True})

    def create(self, **settings):
        return Project.create(self.root / "Project", self.profile, settings, "A usable region of the target material")

    def imported(self):
        project = self.create()
        self.assertTrue(project.import_image(self.source, "sample A"))
        self.assertGreater(len(project.data["images"][0]["regions"]), 0)
        return project

    def test_real_detection_bypasses_filters_and_preserves_profile_and_raw(self):
        before_profile, before_image = self.profile.read_bytes(), self.source.read_bytes()
        project = self.imported()
        image = project.data["images"][0]
        region = image["regions"][0]
        self.assertEqual(region["excluded_by"], [])
        self.assertTrue(all(not result["passed"] for result in region["original_filter_results"]))
        self.assertEqual(project.asset(image["asset"]).read_bytes(), before_image)
        self.assertEqual(self.profile.read_bytes(), before_profile)
        self.assertEqual(self.source.read_bytes(), before_image)
        self.assertEqual((project.folder / "profile.json").read_bytes(), before_profile)
        self.assertIn("PNG metadata", image["calibration"])

    def test_resume_duplicate_and_original_profile_changes_do_not_change_project(self):
        project = self.imported()
        project.label(0, 0, "bad", "Debris")
        self.profile.write_text("{}", encoding="utf-8")
        loaded = Project.open(project.folder / "project.json")
        self.assertFalse(loaded.import_image(self.source, "another scan"))
        region = loaded.data["images"][0]["regions"][0]
        self.assertEqual((region["label"], region["reason"]), ("bad", "Debris"))
        self.assertEqual(len(loaded.data["images"]), 1)

    def test_excluded_candidates_retained_and_pa_opt_in(self):
        project = self.create(minimum_area_um2=1e8, use_pa_filter=True)
        project.import_image(self.source, "sample")
        region = project.data["images"][0]["regions"][0]
        self.assertEqual(len(region["excluded_by"]), 2)
        self.assertIn("P:A", region["excluded_by"])
        self.assertEqual(region["label"], "unlabeled")

    def test_minimum_area_is_inclusive(self):
        settings = {"minimum_area_um2": 50, "use_pa_filter": False, "use_other_geometry": False}
        _, excluded = candidate_filters({"area_um2": 50}, [], settings)
        self.assertEqual(excluded, [])
        _, excluded = candidate_filters({"area_um2": 49.9}, [], settings)
        self.assertTrue(excluded)

    def test_identified_preview_uses_masks_exact_colors_and_visible_candidates(self):
        regions = [
            {"bounding_box": {"x": index * 4, "y": 1, "width": 3, "height": 3}, "label": label}
            for index, label in enumerate(("unlabeled", "good", "bad", "unsure"))
        ]
        mask = np.full((3, 3), 255, np.uint8)
        mask[1, 1] = 0
        raw = np.full((6, 16, 3), 42, np.uint8)
        preview = identified_region_preview(raw, regions, range(4), lambda region: mask)
        for index, color in enumerate(((255, 255, 255), (0, 255, 0), (255, 0, 0), (128, 128, 128))):
            self.assertEqual(tuple(preview[1, index * 4]), color)
            self.assertEqual(tuple(preview[2, index * 4 + 1]), (42, 42, 42))
        hidden = identified_region_preview(raw, regions, [1], lambda region: mask)
        np.testing.assert_array_equal(hidden[:, :4], raw[:, :4])
        self.assertEqual(tuple(hidden[1, 4]), (0, 255, 0))
        self.assertTrue(np.all(raw == 42))

    def test_missing_metadata_requires_explicit_fallback(self):
        no_metadata = self.root / "plain.png"
        save_png(no_metadata, np.full((50, 50, 3), 180, np.uint8))
        project = self.create()
        with self.assertRaisesRegex(ValueError, "missing calibration"):
            project.import_image(no_metadata, "sample")
        self.assertEqual(project.data["images"], [])
        project.data["settings"]["allow_calibration_fallback"] = True
        project.import_image(no_metadata, "sample")
        self.assertIn("Fallback", project.data["images"][0]["calibration"])

    def test_full_review_and_undo_preserve_group_split(self):
        project = self.imported()
        with self.assertRaisesRegex(ValueError, "Resolve"):
            project.set_reviewed(0, True)
        for index in range(len(project.data["images"][0]["regions"])):
            project.label(0, index, "good")
        project.add_box(0, {"x": 5, "y": 5, "width": 15, "height": 20})
        project.assign_split(0, "test")
        project.undo()
        self.assertEqual(project.data["images"][0]["split"], "test")
        self.assertEqual(project.data["images"][0]["manual_good_boxes"], [])
        project.set_reviewed(0, True)
        project.label(0, 0, "unsure")
        self.assertFalse(project.data["images"][0]["fully_reviewed"])
        project.undo()
        self.assertTrue(project.data["images"][0]["fully_reviewed"])

    def test_group_assignment_covers_current_and_future_images(self):
        project = self.imported()
        other = self.root / "other.png"
        save_png(other, np.full((50, 50, 3), 160, np.uint8), {"magnification": "10X", "resolution": "HIGH"})
        project.assign_split(0, "validation")
        project.import_image(other, "sample A")
        self.assertEqual([i["split"] for i in project.data["images"]], ["validation"] * 2)
        project.assign_split(1, "train")
        self.assertEqual([i["split"] for i in project.data["images"]], ["train"] * 2)

    def test_failed_save_rolls_back_edit(self):
        project = self.imported()
        before = copy.deepcopy(project.data["images"])
        with patch.object(project, "save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                project.label(0, 0, "good")
        self.assertEqual(project.data["images"], before)
        self.assertEqual(project.history, [])

    def test_export_has_masks_crops_and_reviewed_image_references(self):
        project = self.imported()
        for index in range(len(project.data["images"][0]["regions"])):
            project.label(0, index, "good")
        project.add_box(0, {"x": 5, "y": 5, "width": 10, "height": 10})
        project.set_reviewed(0, True)
        project.assign_split(0, "test")
        target = project.export()
        data = json.loads((target / "dataset.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["ground_truth"]), 1)
        self.assertEqual(len(data["ground_truth"][0]["missed_good_boxes"]), 1)
        sample = cv2.imread(str(target / data["samples"][0]["sample"]))
        self.assertEqual(sample.shape, (512, 1024, 3))
        self.assertEqual(set(np.unique(sample[:, 512:])), {0, 255})
        self.assertEqual(data["samples"][0]["split"], "test")
        self.assertEqual((target / data["images"][0]["asset"]).read_bytes(), self.source.read_bytes())
        self.assertTrue((target / data["images"][0]["regions"][0]["mask"]).is_file())

    def test_unsure_and_unreviewed_are_not_exported_as_training_or_reference(self):
        project = self.imported()
        project.label(0, 0, "unsure")
        target = project.export()
        data = json.loads((target / "dataset.json").read_text(encoding="utf-8"))
        self.assertEqual(data["samples"], [])
        self.assertEqual(data["ground_truth"], [])
        self.assertEqual(data["images"][0]["regions"][0]["label"], "unsure")

    def test_scan_discovery_ignores_processed_and_other_images(self):
        scan = self.root / "scan"
        for name in ("All Images/10x/chip/Raw/a.png", "All Images/20x/chip/Raw/b.jpg", "All Images/10x/chip/Processed/a.png", "Flakes Found/a.png"):
            p = scan / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
        self.assertEqual(len(discover_images(scan, scan=True)), 2)

    def test_assets_cannot_escape_or_use_absolute_paths(self):
        project = self.imported()
        for path in ("../raw.png", str(self.source), str(project.asset(project.data["images"][0]["asset"]))):
            with self.assertRaises(ValueError):
                project.asset(path)

    def test_modified_region_id_cannot_write_outside_export(self):
        project = self.imported()
        project.data["images"][0]["regions"][0]["id"] = "../../outside"
        project.save()
        with self.assertRaisesRegex(ValueError, "region ID"):
            Project.open(project.folder)

    def test_nonempty_project_folder_and_changed_snapshot_are_rejected(self):
        project = self.imported()
        with self.assertRaisesRegex(ValueError, "empty"):
            self.create()
        (project.folder / "profile.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed"):
            Project.open(project.folder)

    def test_cancelled_export_has_no_finished_manifest(self):
        project = self.imported()
        with self.assertRaises(InterruptedError):
            project.export(cancelled=lambda: True)
        self.assertEqual(list((project.folder / "Exports").rglob("dataset.json")), [])

    def test_empty_image_can_be_a_fully_reviewed_negative_reference(self):
        project = self.create()
        blank = self.root / "blank.png"
        save_png(blank, np.full((100, 100, 3), 180, np.uint8), {"magnification": "10X", "resolution": "HIGH"})
        project.import_image(blank, "negative sample")
        self.assertEqual(project.data["images"][0]["regions"], [])
        project.set_reviewed(0, True)
        data = json.loads((project.export() / "dataset.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["ground_truth"]), 1)
        self.assertEqual(data["ground_truth"][0]["good_regions"], [])


class AnnotationInterfaceTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk is unavailable: {exc}")
        self.fixture = AnnotationProjectTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        from tools.annotation_tool import AnnotationApp
        self.app = AnnotationApp(self.root)
        self.addCleanup(self.app.close)
        self.app.load(self.fixture.imported())
        self.root.geometry("1100x760+0+0")
        self.root.update()

    def test_label_full_image_box_undo_and_resume(self):
        self.app.reason.set("Clean target region")
        self.app.label("good")
        self.assertEqual(self.app.item()["regions"][0]["label"], "good")
        self.assertEqual(self.app.item()["regions"][0]["reason"], "Clean target region")
        self.app.draw_mode.set(True)
        self.app.update_draw_mode()
        scale = self.app.canvas.scale
        ox, oy = self.app.canvas.offset
        self.app.image_press(SimpleNamespace(x=ox + 5 * scale, y=oy + 5 * scale))
        self.app.image_release(SimpleNamespace(x=ox + 25 * scale, y=oy + 35 * scale))
        self.assertEqual(len(self.app.item()["manual_good_boxes"]), 1)
        self.app.reviewed.set(True)
        self.app.mark_reviewed()
        self.assertTrue(Project.open(self.app.project.folder).data["images"][0]["fully_reviewed"])
        self.app.undo()
        self.assertFalse(self.app.item()["fully_reviewed"])
        self.assertGreater(self.app.canvas.winfo_width(), 500)
        self.assertGreater(self.app.crop.winfo_width(), 200)

    def test_zoom_keeps_pointer_source_position_and_busy_blocks_labels(self):
        canvas = self.app.canvas
        event = SimpleNamespace(x=canvas.winfo_width() / 2, y=canvas.winfo_height() / 2, delta=120)
        before = canvas.point(event.x, event.y)
        canvas.zoom(event)
        after = canvas.point(event.x, event.y)
        self.assertAlmostEqual(before[0], after[0])
        self.assertAlmostEqual(before[1], after[1])
        self.app.busy = True
        self.app.label("bad")
        self.assertEqual(self.app.item()["regions"][0]["label"], "unlabeled")
        self.app.busy = False

    def test_crop_mask_toggle_and_label_changes_preserve_raw_crop(self):
        original = self.app.rgb.copy()
        raw_crop = self.app.crop.rgb.copy()
        scale, offset = self.app.canvas.scale, self.app.canvas.offset[:]
        self.app.crop_mask_check.invoke()
        self.assertTrue(self.app.show_crop_mask.get())
        self.assertEqual(set(np.unique(self.app.crop.rgb)), {0, 255})
        binary_mask = self.app.crop.rgb.copy()
        self.assertEqual(self.app.canvas.scale, scale)
        self.assertEqual(self.app.canvas.offset, offset)
        region = self.app.item()["regions"][0]
        mask = self.app.project.mask(region)
        ys, xs = np.nonzero(mask)
        x, y = region["bounding_box"]["x"] + xs[0], region["bounding_box"]["y"] + ys[0]
        self.assertEqual(tuple(self.app.canvas.rgb[y, x]), (255, 255, 255))
        self.app.label("bad")
        self.assertEqual(tuple(self.app.canvas.rgb[y, x]), (255, 0, 0))
        np.testing.assert_array_equal(self.app.crop.rgb, binary_mask)
        self.app.crop_mask_check.invoke()
        np.testing.assert_array_equal(self.app.crop.rgb, raw_crop)
        np.testing.assert_array_equal(self.app.rgb, original)
        export = self.app.project.export()
        data = json.loads((export / "dataset.json").read_text(encoding="utf-8"))
        paired = cv2.cvtColor(cv2.imread(str(export / data["samples"][0]["sample"])), cv2.COLOR_BGR2RGB)
        np.testing.assert_array_equal(paired[:, :512], raw_crop)
        np.testing.assert_array_equal(paired[:, 512:], binary_mask)

    def test_display_all_controls_masks_and_numbers_without_rectangles(self):
        item = self.app.item()
        second = copy.deepcopy(item["regions"][0])
        second.update(id="second", mask="Masks/second.png", label="good", bounding_box={"x": 0, "y": 0, "width": 2, "height": 2})
        save_png(self.app.project.folder / second["mask"], np.full((2, 2), 255, np.uint8))
        item["regions"].append(second)
        self.app.refresh_regions()
        self.assertFalse(self.app.display_all.get())
        self.assertEqual(len(self.app.canvas.find_withtag("region_number")), 1)
        np.testing.assert_array_equal(self.app.canvas.rgb[:2, :2], self.app.rgb[:2, :2])
        self.app.display_all_check.invoke()
        self.assertEqual(len(self.app.canvas.find_withtag("region_number")), 2)
        self.assertEqual(tuple(self.app.canvas.rgb[0, 0]), (0, 255, 0))
        self.assertNotIn("rectangle", [self.app.canvas.type(i) for i in self.app.canvas.find_all()])
        self.app.display_all_check.invoke()
        self.app.move_region(1)
        self.assertEqual([self.app.canvas.itemcget(i, "text") for i in self.app.canvas.find_withtag("region_number")], ["2"])
        self.assertNotIn("rectangle", [self.app.canvas.type(i) for i in self.app.canvas.find_all()])

    def test_dense_numbers_stay_separate_during_pan_and_zoom(self):
        canvas = self.app.canvas
        canvas.overlays = [({"x": 75 + i % 4, "y": 70 + i // 4, "width": 10, "height": 10},
                            "white", i == 19, str(i + 1)) for i in range(20)]

        def check_numbers():
            numbers = canvas.find_withtag("region_number")
            self.assertEqual(len(numbers), 20)
            self.assertEqual(canvas.itemcget(numbers[0], "text"), "1")
            self.assertEqual(len({canvas.itemcget(n, "font") for n in numbers}), 1)
            boxes = [canvas.bbox(n) for n in numbers]
            for i, (left, top, right, bottom) in enumerate(boxes):
                for x1, y1, x2, y2 in boxes[i + 1:]:
                    self.assertTrue(right + 4 <= x1 or x2 + 4 <= left or bottom + 4 <= y1 or y2 + 4 <= top)
            return boxes

        canvas.render()
        before = check_numbers()
        canvas.pan_start(SimpleNamespace(x=200, y=200))
        canvas.pan(SimpleNamespace(x=250, y=240))
        self.assertNotEqual(before, check_numbers())
        canvas.zoom(SimpleNamespace(x=400, y=350, delta=120))
        check_numbers()
        canvas.offset = [-10000, -10000]
        canvas.render()
        check_numbers()

    def test_raw_image_left_drag_pans_and_click_selects(self):
        canvas = self.app.canvas
        before = canvas.offset[:]
        self.app.image_press(SimpleNamespace(x=300, y=300))
        self.app.image_drag(SimpleNamespace(x=370, y=330))
        self.app.image_release(SimpleNamespace(x=370, y=330))
        self.assertEqual(canvas.offset, [before[0] + 70, before[1] + 30])
        self.assertFalse(canvas.fit_mode)
        region = self.app.item()["regions"][0]
        ys, xs = np.nonzero(self.app.cached_mask(region))
        event = SimpleNamespace(x=canvas.offset[0] + (region["bounding_box"]["x"] + xs[0] + .5) * canvas.scale,
                                y=canvas.offset[1] + (region["bounding_box"]["y"] + ys[0] + .5) * canvas.scale)
        self.app.region_index = None
        self.app.image_press(event)
        self.app.image_release(event)
        self.assertEqual(self.app.region_index, 0)

    def test_old_default_upgrade_preserves_labels_and_counts_visible_candidates(self):
        project = self.app.project
        project.data["settings"]["minimum_area_um2"] = 50
        second = copy.deepcopy(self.app.item()["regions"][0])
        second.update(id="small", area_um2=150, label="good")
        self.app.item()["regions"].append(second)
        self.app.load(project)
        self.assertEqual(project.data["settings"]["minimum_area_um2"], 200)
        self.assertEqual(second["label"], "good")
        self.assertEqual(self.app.visible_indices, [0])
        self.assertIn("0/1 labeled · 1 filtered out", self.app.image_info.get())
        self.app.excluded.set(True)
        self.app.refresh_regions()
        self.assertIn("1/2 labeled · 0 filtered out", self.app.image_info.get())

    def test_panel_scrolls_without_scrollbar_and_preserves_crop_zoom(self):
        scroller = self.app.panel_scroller
        self.assertFalse(any(isinstance(w, ttk.Scrollbar) for w in scroller.master.winfo_children()))
        self.root.geometry("1100x680+0+0")
        self.root.update()
        before = scroller.yview()
        self.app.scroll_panel(SimpleNamespace(widget=self.app.panel, delta=-120))
        self.assertGreater(scroller.yview()[0], before[0])
        before = scroller.yview()
        self.app.scroll_panel(SimpleNamespace(widget=self.app.crop, delta=-120))
        self.assertEqual(scroller.yview(), before)

    def test_add_scan_discovers_only_raw_and_preserves_existing_labels(self):
        self.app.project.label(0, 0, "good")
        scan = self.fixture.root / "Another scan"
        for name in ("All Images/10x/chip/Raw/new.png", "All Images/10x/chip/Processed/new.png"):
            path = scan / name
            path.parent.mkdir(parents=True, exist_ok=True)
            save_png(path, np.full((100, 100, 3), 175, np.uint8), {"magnification": "10X", "resolution": "HIGH"})
        with patch("tools.annotation_tool.filedialog.askdirectory", return_value=str(scan)), \
             patch("tkinter.simpledialog.askstring", return_value="sample B"), \
             patch.object(self.app, "start_import") as start:
            self.app.add_scan()
        sources, group = start.call_args.args
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].parent.name, "Raw")
        self.app.project.import_image(sources[0], group)
        loaded = Project.open(self.app.project.folder)
        self.assertEqual(len(loaded.data["images"]), 2)
        self.assertEqual(loaded.data["images"][0]["regions"][0]["label"], "good")
        self.assertEqual(loaded.data["images"][1]["group"], "sample B")

    def test_single_left_panel_and_menu_actions_without_toolbars(self):
        self.assertLess(self.app.sidebar.winfo_rootx(), self.app.canvas.winfo_rootx())
        self.assertEqual(self.app.main_frame.winfo_children(), [self.app.sidebar, self.app.canvas])
        self.assertEqual(self.root.pack_slaves(), [self.app.main_frame])
        self.assertEqual([self.app.menu_bar.entrycget(i, "label") for i in range(self.app.menu_bar.index("end") + 1)], ["File", "View", "Review", "Help"])
        def check_buttons(widget):
            for child in widget.winfo_children():
                if child.winfo_class() == "TButton":
                    self.assertEqual(child.cget("style"), "Normal.TButton")
                self.assertNotEqual(child.winfo_class(), "Button")
                check_buttons(child)
        check_buttons(self.app.panel)


if __name__ == "__main__":
    unittest.main()
