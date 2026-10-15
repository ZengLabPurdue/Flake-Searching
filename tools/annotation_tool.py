"""Standalone region annotation: project storage, detector integration, and scanner-style UI."""

from __future__ import annotations

import argparse
import copy
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
from uuid import uuid4

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from config import PIXEL_SIZE
from imaging.image_metadata import read_png_metadata
from scanning.contour_finder import find_flakes
from scanning.region_classifier import load_profile_configuration, _region_passes_filters
from scanning.region_dataset import build_region_training_image


SCHEMA = "flake-search.annotation-project"
LABELS = ("unlabeled", "good", "bad", "unsure")
SPLITS = ("unassigned", "train", "validation", "test")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
DEFAULT_SETTINGS = {
    "minimum_area_um2": 200.0,
    "use_pa_filter": False,
    "use_other_geometry": False,
    "allow_calibration_fallback": False,
    "magnification": "10X",
    "resolution": "HIGH",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def read_image(path):
    image = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_png(path, image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"Cannot encode image: {path}")
    Path(path).write_bytes(encoded.tobytes())


def discover_images(folder, scan=False):
    """Scans include only saved Raw tiles, never processed or crop outputs."""
    folder = Path(folder).resolve()
    base = folder / "All Images" if scan else folder
    if not base.is_dir():
        raise ValueError(f"Image folder does not exist: {base}")
    return sorted(
        p for p in base.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        and (not scan or p.parent.name == "Raw")
    )


def validate_settings(settings):
    result = {**DEFAULT_SETTINGS, **settings}
    area = float(result["minimum_area_um2"])
    if not math.isfinite(area) or area < 0:
        raise ValueError("Minimum area must be a finite number of at least zero.")
    result["minimum_area_um2"] = area
    if result["magnification"] not in PIXEL_SIZE or result["resolution"] not in PIXEL_SIZE[result["magnification"]]:
        raise ValueError("Select a valid fallback magnification and resolution.")
    return result


def image_calibration(path, settings):
    metadata = read_png_metadata(path)
    mag = metadata.get("magnification", "").upper()
    resolution = metadata.get("resolution", "").upper()
    if mag in PIXEL_SIZE and resolution in PIXEL_SIZE[mag]:
        return float(PIXEL_SIZE[mag][resolution]), f"PNG metadata: {mag} / {resolution}", metadata
    if not settings["allow_calibration_fallback"]:
        raise ValueError(f"{Path(path).name}: missing calibration metadata. Enable an explicit fallback in a new project.")
    mag, resolution = settings["magnification"], settings["resolution"]
    return float(PIXEL_SIZE[mag][resolution]), f"Fallback: {mag} / {resolution}", metadata


def candidate_filters(region, filters, settings):
    """Record original decisions without discarding potential training examples."""
    geometry = [item for item in filters if item["type"] != "neural_network"]
    _, original_results = _region_passes_filters(region, geometry)
    by_id = {item["id"]: item for item in geometry}
    excluded = []
    if region["area_um2"] < settings["minimum_area_um2"]:
        excluded.append(f"Area below {settings['minimum_area_um2']:g} µm²")
    for result in original_results:
        item = by_id[result["id"]]
        apply = settings["use_pa_filter"] if item["type"] == "perimeter_area_ratio" else settings["use_other_geometry"]
        if apply and result["enabled"] and not result["passed"]:
            excluded.append(item["name"])
    return original_results, excluded


class Project:
    def __init__(self, folder, data):
        self.folder = Path(folder).resolve()
        self.data = data
        self.history = []

    @classmethod
    def create(cls, folder, profile_path, settings, definition):
        folder = Path(folder).resolve()
        # A project owns only a new/empty directory. Never overlay existing data.
        if folder.exists() and any(folder.iterdir()):
            raise ValueError("Choose a new or empty project directory.")
        for protected in (APP / "Flatfields", APP / "Calibration Backups" / "Flatfields", APP / "Profiles", APP / "Scans", APP / "Saved Images"):
            if folder == protected.resolve() or protected.resolve() in folder.parents:
                raise ValueError("Choose a project location outside microscope data directories.")
        profile_path = Path(profile_path)
        if profile_path.is_dir():
            profile_path /= "profile.json"
        config = load_profile_configuration(profile_path)
        if not any(c.get("identify", True) for c in config["classes"]):
            raise ValueError("The profile must have at least one enabled color class.")
        settings = validate_settings(settings)
        if not definition.strip():
            raise ValueError("Describe what reviewers should label as a good region.")
        profile_bytes = profile_path.read_bytes()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "profile.json").write_bytes(profile_bytes)
        for name in ("Images", "Masks"):
            (folder / name).mkdir()
        data = {
            "schema": SCHEMA, "version": 1, "created_at": now(),
            "profile_source": str(profile_path.resolve()),
            "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
            "settings": settings, "good_definition": definition.strip(),
            "images": [], "cursor": {"image": 0, "region": 0},
        }
        project = cls(folder, data)
        project.save()
        return project

    @classmethod
    def open(cls, folder):
        folder = Path(folder)
        if folder.is_file():
            folder = folder.parent
        data = json.loads((folder / "project.json").read_text(encoding="utf-8"))
        if data.get("schema") != SCHEMA or data.get("version") != 1:
            raise ValueError("This is not a supported annotation project.")
        project = cls(folder, data)
        validate_settings(data["settings"])
        profile_bytes = (folder / "profile.json").read_bytes()
        if hashlib.sha256(profile_bytes).hexdigest() != data["profile_sha256"]:
            raise ValueError("The project's frozen profile has been changed.")
        for item in data["images"]:
            if item["split"] not in SPLITS:
                raise ValueError("Invalid dataset split in project.")
            project.asset(item["asset"])
            for region in item["regions"]:
                if not re.fullmatch(r"[A-Za-z0-9_-]+", region["id"]):
                    raise ValueError("Invalid region ID in project.")
                project.asset(region["mask"])
        return project

    def asset(self, relative):
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Project assets must use relative paths without '..'.")
        path = (self.folder / relative).resolve()
        if self.folder not in path.parents:
            raise ValueError("Project asset path escapes the project directory.")
        if not path.is_file():
            raise FileNotFoundError(f"Project asset is missing: {relative}")
        return path

    def save(self):
        self.data["updated_at"] = now()
        temporary = self.folder / ".project.json.tmp"
        temporary.write_text(json.dumps(self.data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(self.folder / "project.json")

    def import_image(self, source, group):
        """Commit one image at a time; failed/cancelled batches remain resumable."""
        source = Path(source).resolve()
        source_bytes = source.read_bytes()
        digest = hashlib.sha256(source_bytes).hexdigest()
        if any(item["sha256"] == digest for item in self.data["images"]):
            return False
        settings = self.data["settings"]
        pixel_size, calibration, metadata = image_calibration(source, settings)
        image = cv2.imdecode(np.frombuffer(source_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read image: {source}")
        config = load_profile_configuration(self.folder / "profile.json")
        original_filters = config["filters"]
        config["filters"] = []
        _, _, details = find_flakes(
            image, profile_configuration=config, pixel_size_um=pixel_size,
            return_details=True, draw_legend=False, color_seed=0,
        )
        image_id = digest[:24]
        asset = f"Images/{image_id}{source.suffix.lower()}"
        (self.folder / asset).write_bytes(source_bytes)
        regions = []
        for index, detected in enumerate(details["combined_region_results"]):
            region_id = f"{image_id}_{index:05d}"
            mask_path = f"Masks/{region_id}.png"
            write_png(self.folder / mask_path, detected["instance_mask"])
            results, excluded = candidate_filters(detected, original_filters, settings)
            regions.append({
                "id": region_id, "mask": mask_path,
                "bounding_box": detected["bounding_box"],
                "class": detected["matched_class"], "class_group": detected.get("matched_group"),
                "area_um2": detected["area_um2"],
                "width_um": detected["width_um"], "height_um": detected["height_um"],
                "perimeter_area_ratio_per_um": detected["perimeter_area_ratio_per_um"],
                "original_filter_results": results, "excluded_by": excluded,
                "label": "unlabeled", "reason": "", "reviewed_at": None,
            })
        split = next((item["split"] for item in self.data["images"] if item["group"] == group), "unassigned")
        item = {
            "id": image_id, "sha256": digest, "asset": asset, "source": str(source),
            "width": int(image.shape[1]), "height": int(image.shape[0]),
            "pixel_size_um": pixel_size, "calibration": calibration, "metadata": metadata,
            "group": group, "split": split, "regions": regions,
            "manual_good_boxes": [], "fully_reviewed": False,
        }
        self.data["images"].append(item)
        try:
            self.save()
        except Exception:
            self.data["images"].pop()
            raise
        return True

    def _change(self, image_index, function):
        """Persist an annotation edit and retain a bounded session undo history."""
        previous = copy.deepcopy(self.data["images"][image_index])
        try:
            function(self.data["images"][image_index])
            self.save()
        except Exception:
            self.data["images"][image_index] = previous
            raise
        self.history.append((image_index, previous))
        self.history = self.history[-100:]

    def label(self, image_index, region_index, label, reason=""):
        if label not in LABELS:
            raise ValueError("Unknown annotation label.")
        def edit(item):
            item["regions"][region_index].update(label=label, reason=reason.strip(), reviewed_at=now())
            item["fully_reviewed"] = False
        self._change(image_index, edit)

    def add_box(self, image_index, box, note=""):
        item = self.data["images"][image_index]
        x, y, w, h = [int(box[key]) for key in ("x", "y", "width", "height")]
        if x < 0 or y < 0 or w < 2 or h < 2 or x + w > item["width"] or y + h > item["height"]:
            raise ValueError("Draw a box of at least 2 × 2 pixels inside the image.")
        def edit(image):
            image["manual_good_boxes"].append({"id": uuid4().hex, "bounding_box": {"x": x, "y": y, "width": w, "height": h}, "note": note, "created_at": now()})
            image["fully_reviewed"] = False
        self._change(image_index, edit)

    def remove_box(self, image_index, box_id):
        def edit(image):
            image["manual_good_boxes"] = [b for b in image["manual_good_boxes"] if b["id"] != box_id]
            image["fully_reviewed"] = False
        self._change(image_index, edit)

    def set_reviewed(self, image_index, reviewed):
        if reviewed and any(r["label"] in {"unlabeled", "unsure"} for r in self.data["images"][image_index]["regions"] if not r["excluded_by"]):
            raise ValueError("Resolve all included candidates before marking the full image reviewed.")
        self._change(image_index, lambda item: item.update(fully_reviewed=bool(reviewed)))

    def assign_split(self, image_index, split):
        if split not in SPLITS:
            raise ValueError("Unknown dataset split.")
        group = self.data["images"][image_index]["group"]
        previous = [(item, item["split"]) for item in self.data["images"] if item["group"] == group]
        for item, _ in previous:
            item["split"] = split
        try:
            self.save()
        except Exception:
            for item, old in previous:
                item["split"] = old
            raise

    def undo(self):
        if not self.history:
            return None
        index, previous = self.history[-1]
        current = self.data["images"][index]
        # Dataset assignment is a group-wide operation, not an annotation edit.
        previous["split"] = current["split"]
        self.data["images"][index] = previous
        try:
            self.save()
        except Exception:
            self.data["images"][index] = current
            raise
        self.history.pop()
        return index

    def mask(self, region):
        mask = cv2.imdecode(np.frombuffer(self.asset(region["mask"]).read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
        box = region["bounding_box"]
        if mask is None or mask.shape != (box["height"], box["width"]):
            raise ValueError("A stored region mask is invalid.")
        return mask

    def export(self, progress=None, cancelled=None):
        """Export crops and full-image reference annotations; never overwrite a run."""
        target = self.folder / "Exports" / (datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8])
        if self.folder not in target.resolve().parents:
            raise ValueError("The Exports folder must stay inside the annotation project.")
        target.mkdir(parents=True)
        records, images, ground_truth = [], [], []
        for index, item in enumerate(self.data["images"]):
            if cancelled and cancelled():
                raise InterruptedError(f"Export cancelled. Partial files are in {target}; no dataset.json was written.")
            rgb = cv2.cvtColor(read_image(self.asset(item["asset"])), cv2.COLOR_BGR2RGB)
            destination = target / item["asset"]
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(self.asset(item["asset"]), destination)
            reference = copy.deepcopy(item)
            reference["good_definition"] = self.data["good_definition"]
            images.append(reference)
            if item["fully_reviewed"]:
                ground_truth.append({
                    "image_id": item["id"], "asset": item["asset"],
                    "group": item["group"], "split": item["split"],
                    "good_regions": [
                        {"id": r["id"], "bounding_box": r["bounding_box"], "mask": r["mask"]}
                        for r in item["regions"] if r["label"] == "good" and not r["excluded_by"]
                    ],
                    "missed_good_boxes": copy.deepcopy(item["manual_good_boxes"]),
                })
            for region in item["regions"]:
                mask_destination = target / region["mask"]
                mask_destination.parent.mkdir(exist_ok=True)
                shutil.copy2(self.asset(region["mask"]), mask_destination)
                if region["label"] not in {"good", "bad"}:
                    continue
                detection = {"bounding_box": region["bounding_box"], "instance_mask": self.mask(region)}
                sample = build_region_training_image(rgb, detection)
                if sample is None:
                    raise ValueError(f"Cannot build crop for {region['id']}")
                relative = f"Samples/{item['split']}/{region['label']}/{region['id']}.png"
                path = target / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                write_png(path, cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))
                records.append({"sample": relative, "image_id": item["id"], "region_id": region["id"], "label": region["label"], "group": item["group"], "split": item["split"]})
            if progress:
                progress(index + 1, len(self.data["images"]))
        manifest = {
            "schema": "flake-search.annotation-export", "version": 1,
            "created_at": now(), "good_definition": self.data["good_definition"],
            "settings": self.data["settings"], "profile_sha256": self.data["profile_sha256"],
            "samples": records, "images": images, "ground_truth": ground_truth,
            "reference_scope": "Fully reviewed images only; qualifying good candidates plus manually added good boxes. Exclusions are recorded per candidate.",
            "sample_format": "512x1024 RGB image: 512x512 source crop on left, aligned binary mask on right",
        }
        shutil.copy2(self.folder / "profile.json", target / "profile.json")
        (target / "dataset.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return target


REGION_COLORS_RGB = {
    "unlabeled": (255, 255, 255),
    "good": (0, 255, 0),
    "bad": (255, 0, 0),
    "unsure": (128, 128, 128),
}
COLORS = {label: "#%02x%02x%02x" % rgb for label, rgb in REGION_COLORS_RGB.items()}


def identified_region_preview(image_rgb, regions, visible_indices, load_mask):
    """Color actual region pixels while preserving the raw image's context."""
    preview = image_rgb.copy()
    for index in visible_indices:
        region = regions[index]
        box = region["bounding_box"]
        x, y, width, height = (box[key] for key in ("x", "y", "width", "height"))
        mask = load_mask(region)
        preview[y:y + height, x:x + width][mask > 0] = REGION_COLORS_RGB[region["label"]]
    return preview


class ImageCanvas(tk.Canvas):
    """Image-space coordinates with pointer-centred zoom and middle-button pan."""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg="#f0f0f0", highlightthickness=0, **kwargs)
        self.rgb = None
        self.scale = 1.0
        self.offset = [0.0, 0.0]
        self.photo = None
        self.empty_message = "Create or open an annotation project"
        self.overlays = []
        self.fit_mode = True
        self.bind("<Configure>", lambda e: self.fit() if self.fit_mode else self.render())
        self.bind("<MouseWheel>", self.zoom)
        self.bind("<Button-4>", lambda e: self.zoom(e, 1.2))
        self.bind("<Button-5>", lambda e: self.zoom(e, 1 / 1.2))
        self.bind("<ButtonPress-2>", self.pan_start)
        self.bind("<B2-Motion>", self.pan)

    def set_image(self, rgb):
        self.rgb = rgb
        self.fit()

    def fit(self):
        self.fit_mode = True
        if self.rgb is not None:
            h, w = self.rgb.shape[:2]
            cw, ch = max(1, self.winfo_width()), max(1, self.winfo_height())
            self.scale = min(cw / w, ch / h)
            self.offset = [(cw - w * self.scale) / 2, (ch - h * self.scale) / 2]
        self.render()

    def point(self, x, y, clamp=False):
        if self.rgb is None:
            return None
        px, py = (x - self.offset[0]) / self.scale, (y - self.offset[1]) / self.scale
        h, w = self.rgb.shape[:2]
        if clamp:
            return max(0, min(w, px)), max(0, min(h, py))
        return (px, py) if 0 <= px < w and 0 <= py < h else None

    def zoom(self, event, factor=None):
        if self.rgb is None:
            return
        source = self.point(event.x, event.y, clamp=True)
        factor = factor or (1.2 if event.delta > 0 else 1 / 1.2)
        self.scale = max(0.02, min(12, self.scale * factor))
        self.offset = [event.x - source[0] * self.scale, event.y - source[1] * self.scale]
        self.fit_mode = False
        self.render()

    def pan_start(self, event):
        self.pan_anchor = (event.x, event.y, *self.offset)

    def pan(self, event):
        x, y, ox, oy = self.pan_anchor
        self.offset = [ox + event.x - x, oy + event.y - y]
        self.fit_mode = False
        self.render()

    def render(self):
        self.delete("all")
        if self.rgb is None:
            self.create_text(max(100, self.winfo_width() / 2), max(80, self.winfo_height() / 2), text=self.empty_message, fill="#555555", font=("TkDefaultFont", 14))
            return
        h, w = self.rgb.shape[:2]
        cw, ch = max(1, self.winfo_width()), max(1, self.winfo_height())
        ox, oy = self.offset
        x1, y1 = max(0, int(-ox / self.scale)), max(0, int(-oy / self.scale))
        x2, y2 = min(w, int((cw - ox) / self.scale) + 1), min(h, int((ch - oy) / self.scale) + 1)
        if x2 > x1 and y2 > y1:
            viewport = self.rgb[y1:y2, x1:x2]
            resized = cv2.resize(viewport, (max(1, round((x2 - x1) * self.scale)), max(1, round((y2 - y1) * self.scale))), interpolation=cv2.INTER_AREA if self.scale < 1 else cv2.INTER_NEAREST)
            self.photo = ImageTk.PhotoImage(Image.fromarray(resized))
            self.create_image(ox + x1 * self.scale, oy + y1 * self.scale, image=self.photo, anchor="nw")
        placed = []
        # Keep each number at its region; only nudge downward enough to
        # separate neighboring numbers. No viewport search or label hiding.
        for box, color, selected, label in self.overlays:
            x, y = ox + box["x"] * self.scale, oy + box["y"] * self.scale
            number = self.create_text(x + 2, y + 2, text=label, fill=color,
                                      anchor="nw", font=("TkDefaultFont", 10, "bold"), tags="region_number")
            while True:
                left, top, right, bottom = self.bbox(number)
                collisions = [bounds for bounds in placed
                              if left < bounds[2] + 4 and right + 4 > bounds[0]
                              and top < bounds[3] + 4 and bottom + 4 > bounds[1]]
                if not collisions:
                    break
                self.move(number, 0, max(bounds[3] for bounds in collisions) + 4 - top)
            placed.append(self.bbox(number))


class NewProjectDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("New Annotation Project")
        self.configure(bg="white")
        self.resizable(False, False)
        self.transient(parent)
        self.result = None
        self.sources = []
        self.source_description = tk.StringVar(value="No source images selected")
        self.profile = tk.StringVar()
        self.folder = tk.StringVar()
        self.minimum = tk.StringVar(value="200")
        self.pa = tk.BooleanVar(value=False)
        self.geometry = tk.BooleanVar(value=False)
        self.fallback = tk.BooleanVar(value=False)
        self.mag = tk.StringVar(value="10X")
        self.res = tk.StringVar(value="HIGH")
        self.group = tk.StringVar()
        self.definition = tk.StringVar(value="The selected region is a usable flake of its assigned profile class.")
        body = tk.Frame(self, bg="white", padx=16, pady=12)
        body.pack(fill="both", expand=True)
        self.row(body, "Project folder", self.folder, lambda: self.folder.set(filedialog.askdirectory(parent=self, title="Choose a new or empty project folder", mustexist=False) or self.folder.get()))
        profile_paths = sorted((APP / "Profiles").glob("*/profile.json"))
        if profile_paths:
            self.profile.set(str(profile_paths[0]))
        self.row(body, "Profile JSON", self.profile, lambda: self.profile.set(filedialog.askopenfilename(parent=self, initialdir=APP / "Profiles", filetypes=[("Scan profile", "*.json")]) or self.profile.get()))
        buttons = tk.Frame(body, bg="white")
        buttons.pack(fill="x", pady=(8, 4))
        for label, mode in (("Choose scan", "scan"), ("Choose folder", "folder"), ("Choose images", "files")):
            ttk.Button(buttons, style="Normal.TButton", text=label, command=lambda m=mode: self.choose_sources(m)).pack(side="left", padx=(0, 5))
        tk.Label(body, textvariable=self.source_description, bg="white", wraplength=580, anchor="w", justify="left").pack(fill="x")
        self.row(body, "Sample / scan group", self.group)
        self.row(body, "Good region means", self.definition)
        self.row(body, "Minimum area (µm²)", self.minimum)
        tk.Label(body, text="Regions below this area are hidden by default, but remain available for review.", bg="white", fg="#555555").pack(anchor="w")
        for label, var in (("Apply the profile's P:A filter", self.pa), ("Apply the profile's other geometry filters", self.geometry), ("Allow calibration fallback for images without metadata", self.fallback)):
            tk.Checkbutton(body, text=label, variable=var, bg="white", anchor="w").pack(anchor="w", pady=2)
        row = tk.Frame(body, bg="white")
        row.pack(fill="x", pady=4)
        tk.Label(row, text="Fallback", bg="white").pack(side="left")
        ttk.Combobox(row, textvariable=self.mag, values=("2X", "10X", "20X", "100X"), state="readonly", width=9).pack(side="left", padx=8)
        ttk.Combobox(row, textvariable=self.res, values=("HIGH", "MED", "LOW"), state="readonly", width=9).pack(side="left")
        tk.Label(body, text="The project freezes a copy of the profile. Neural-network filtering is off.\nUse the same group for images of the same physical sample to keep dataset splits together.", bg="white", fg="#555555", justify="left").pack(anchor="w", pady=8)
        ttk.Button(body, style="Normal.TButton", text="Create project and detect regions", command=self.accept).pack(fill="x", pady=4)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def row(self, body, label, variable, browse=None):
        row = tk.Frame(body, bg="white")
        row.pack(fill="x", pady=4)
        tk.Label(row, text=label, bg="white", width=22, anchor="w").pack(side="left")
        ttk.Entry(row, textvariable=variable, width=57).pack(side="left", fill="x", expand=True)
        if browse:
            ttk.Button(row, style="Normal.TButton", text="Browse…", command=browse, width=9).pack(side="left", padx=(5, 0))

    def choose_sources(self, mode):
        try:
            if mode == "files":
                selected = filedialog.askopenfilenames(parent=self, filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp")])
                if not selected:
                    return
                paths = [Path(p) for p in selected]
                group = str(paths[0].parent)
            else:
                selected = filedialog.askdirectory(parent=self, title="Choose a saved scan" if mode == "scan" else "Choose a folder of source images")
                if not selected:
                    return
                paths = discover_images(selected, scan=mode == "scan")
                group = selected
                if mode == "scan":
                    manifest = Path(selected) / "flakes_found.json"
                    if manifest.is_file():
                        import json
                        hint = json.loads(manifest.read_text(encoding="utf-8")).get("profile_path")
                        if hint and Path(hint).exists():
                            self.profile.set(str(Path(hint) / "profile.json" if Path(hint).is_dir() else hint))
            self.sources = paths
            self.group.set(group)
            self.source_description.set(f"{len(paths)} source images selected — {group}")
        except Exception as exc:
            messagebox.showerror("Cannot select images", str(exc), parent=self)

    def accept(self):
        try:
            if not self.sources or not self.folder.get() or not self.group.get().strip():
                raise ValueError("Choose source images, an empty project folder, and a sample / scan group.")
            destination = Path(self.folder.get()).resolve()
            if any(destination == p.resolve().parent or destination in p.resolve().parents for p in self.sources):
                raise ValueError("Keep the annotation project separate from your input images.")
            settings = {**DEFAULT_SETTINGS, "minimum_area_um2": float(self.minimum.get()), "use_pa_filter": self.pa.get(), "use_other_geometry": self.geometry.get(), "allow_calibration_fallback": self.fallback.get(), "magnification": self.mag.get(), "resolution": self.res.get()}
            project = Project.create(destination, self.profile.get(), settings, self.definition.get())
            self.result = (project, self.sources, self.group.get().strip())
            self.destroy()
        except Exception as exc:
            messagebox.showerror("Cannot create project", str(exc), parent=self)


class AnnotationApp:
    def __init__(self, root, project=None):
        self.root = root
        root.title("Flake Search — Region Annotation")
        width = min(1460, root.winfo_screenwidth() - 80)
        height = min(920, root.winfo_screenheight() - 100)
        root.geometry(f"{width}x{height}")
        root.minsize(min(1100, width), min(680, height))
        self.project = None
        self.image_index = 0
        self.region_index = None
        self.rgb = None
        self.busy = False
        self.closing = False
        self.events = queue.Queue()
        self.cancelled = threading.Event()
        self.visible_indices = []
        self.excluded = tk.BooleanVar(value=False)
        self.display_all = tk.BooleanVar(value=False)
        self.show_crop_mask = tk.BooleanVar(value=False)
        self._mask_cache = {}
        self.draw_mode = tk.BooleanVar(value=False)
        self.auto_next = tk.BooleanVar(value=True)
        self.reviewed = tk.BooleanVar(value=False)
        self.reason = tk.StringVar()
        self.split = tk.StringVar(value="unassigned")
        self.status = tk.StringVar(value="Create a project to label regions, or open an existing project to resume.")
        self.details = tk.StringVar(value="No region selected")
        self.image_info = tk.StringVar(value="")
        self._build_ui()
        self._bind_keys()
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.poll_job = root.after(100, self.poll)
        if project:
            self.load(Project.open(project))

    def _build_ui(self):
        style = ttk.Style(self.root)
        # Match the scanner's Stage and Focus panels, including native button
        # borders, padding, and hover/pressed states supplied by the Tk theme.
        style.configure("Normal.TButton", font="TkDefaultFont", background="white", relief="flat")
        style.configure("Profile.TCheckbutton", background="white", font="TkDefaultFont")

        self.main_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.main_frame.pack(fill="both", expand=True)
        self.sidebar = tk.Frame(self.main_frame, bg="#f0f0f0", width=300)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        background = tk.Frame(self.sidebar, bg="white")
        background.pack(fill="both", expand=True, padx=2)
        scroller = tk.Canvas(background, bg="white", highlightthickness=0, width=282)
        scroller.pack(side="left", fill="both", expand=True)
        self.panel_scroller = scroller
        panel = tk.Frame(scroller, bg="white", padx=8)
        window = scroller.create_window(0, 0, window=panel, anchor="nw")
        panel.bind("<Configure>", lambda e: scroller.configure(scrollregion=scroller.bbox("all")))
        scroller.bind("<Configure>", lambda e: scroller.itemconfigure(window, width=e.width))
        self.panel = panel
        self.root.bind("<MouseWheel>", self.scroll_panel, add="+")
        self.root.bind("<Button-4>", self.scroll_panel, add="+")
        self.root.bind("<Button-5>", self.scroll_panel, add="+")
        self.canvas = ImageCanvas(self.main_frame)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.image_press)
        self.canvas.bind("<B1-Motion>", self.image_drag)
        self.canvas.bind("<ButtonRelease-1>", self.image_release)

        tk.Label(panel, text="Region Annotation", bg="white", fg="black", font=("TkDefaultFont", 13)).pack(pady=(8, 7))
        self.heading(panel, "Image")
        tk.Label(panel, textvariable=self.image_info, bg="white", fg="black", wraplength=255, justify="left", anchor="w").pack(fill="x", pady=(4, 3))
        row = tk.Frame(panel, bg="white")
        row.pack(fill="x", pady=(0, 5))
        self.button(row, "Previous image", lambda: self.move_image(-1)).pack(side="left", expand=True, fill="x")
        self.button(row, "Next image", lambda: self.move_image(1)).pack(side="left", expand=True, fill="x")
        self.display_all_check = ttk.Checkbutton(panel, text="Display all", variable=self.display_all, command=self.refresh_overlays, style="Profile.TCheckbutton")
        self.display_all_check.pack(anchor="w", pady=(0, 5))
        self.crop = ImageCanvas(panel, height=155)
        self.crop.empty_message = "No region selected"
        self.crop.pack(fill="x", pady=(0, 5))
        self.crop_mask_check = ttk.Checkbutton(panel, text="Show mask", variable=self.show_crop_mask, command=self.refresh_crop, style="Profile.TCheckbutton")
        self.crop_mask_check.pack(anchor="w", pady=(0, 5))
        self.heading(panel, "Selected region")
        tk.Label(panel, textvariable=self.details, bg="white", fg="black", justify="left", anchor="w", wraplength=255).pack(fill="x", pady=(4, 5))
        list_frame = tk.Frame(panel, bg="white")
        list_frame.pack(fill="x")
        self.region_list = tk.Listbox(list_frame, height=4, exportselection=False, width=28)
        region_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.region_list.yview)
        self.region_list.configure(yscrollcommand=region_scroll.set)
        self.region_list.pack(side="left", fill="x", expand=True)
        region_scroll.pack(side="right", fill="y")
        self.region_list.bind("<<ListboxSelect>>", self.select_list)
        row = tk.Frame(panel, bg="white")
        row.pack(fill="x", pady=(3, 0))
        self.button(row, "Previous region", lambda: self.move_region(-1)).pack(side="left", expand=True, fill="x")
        self.button(row, "Next region", lambda: self.move_region(1)).pack(side="left", expand=True, fill="x")
        row = tk.Frame(panel, bg="white")
        row.pack(fill="x", pady=(4, 3))
        for label, title in (("good", "Good [G]"), ("bad", "Bad [B]"), ("unsure", "Unsure [U]")):
            self.button(row, title, lambda value=label: self.label(value), width=8).pack(side="left", expand=True, fill="x")
        tk.Label(panel, text="Note (saved with label)", bg="white", fg="black").pack(anchor="w", pady=(3, 2))
        ttk.Entry(panel, textvariable=self.reason).pack(fill="x")
        ttk.Checkbutton(panel, text="Advance after labeling", variable=self.auto_next, style="Profile.TCheckbutton").pack(anchor="w", pady=(5, 0))
        self.heading(panel, "Full-image review", pady=(9, 4))
        ttk.Checkbutton(panel, text="Draw a missed good flake", variable=self.draw_mode, command=self.update_draw_mode, style="Profile.TCheckbutton").pack(anchor="w")
        row = tk.Frame(panel, bg="white")
        row.pack(fill="x", pady=4)
        self.manual_list = ttk.Combobox(row, state="readonly", width=19)
        self.manual_list.pack(side="left", fill="x", expand=True)
        self.manual_list.bind("<<ComboboxSelected>>", lambda e: self.select_manual())
        self.button(row, "Remove", self.remove_box, width=7).pack(side="left", padx=(3, 0))
        ttk.Checkbutton(panel, text="Entire image reviewed", variable=self.reviewed, command=self.mark_reviewed, style="Profile.TCheckbutton").pack(anchor="w")
        row = tk.Frame(panel, bg="white")
        row.pack(fill="x", pady=(7, 4))
        tk.Label(row, text="Group split", bg="white", fg="black").pack(side="left")
        ttk.Combobox(row, textvariable=self.split, values=SPLITS, state="readonly", width=13).pack(side="right")
        self.split.trace_add("write", self.split_changed)
        self.progress = ttk.Progressbar(panel, mode="determinate", length=240)
        self.status_label = tk.Label(panel, textvariable=self.status, bg="white", fg="#555555", anchor="w", wraplength=255, justify="left")
        self.status_label.pack(fill="x", pady=(9, 8))
        self._build_menu()

    @staticmethod
    def button(parent, text, command, width=None):
        options = {"text": text, "command": command, "style": "Normal.TButton"}
        if width is not None:
            options["width"] = width
        return ttk.Button(parent, **options)

    @staticmethod
    def heading(parent, text, size=10, pady=0):
        tk.Label(parent, text=text, bg="white", fg="black", font=("TkDefaultFont", size, "bold")).pack(anchor="w", pady=pady)

    def _build_menu(self):
        self.menu_bar = tk.Menu(self.root, tearoff=0)
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        for label, command in (("New project…", self.new), ("Open project…", self.open), ("Add scan…", self.add_scan), ("Add image folder…", self.add_folder), ("Add source images…", self.add_images)):
            file_menu.add_command(label=label, command=command)
        file_menu.add_separator()
        file_menu.add_command(label="Export dataset", command=self.export)
        file_menu.add_command(label="Project details", command=self.project_details)
        file_menu.add_separator()
        file_menu.add_command(label="Cancel operation", command=self.cancelled.set)
        file_menu.add_command(label="Quit", command=self.close)
        self.menu_bar.add_cascade(label="File", menu=file_menu)

        view_menu = tk.Menu(self.menu_bar, tearoff=0)
        view_menu.add_command(label="Fit image", accelerator="F", command=self.canvas.fit)
        view_menu.add_separator()
        for label, variable, command in (
            ("Display all", self.display_all, self.refresh_overlays),
            ("Show mask in cropped preview", self.show_crop_mask, self.refresh_crop),
            ("Show excluded candidates", self.excluded, self.refresh_regions),
        ):
            view_menu.add_checkbutton(label=label, variable=variable, command=command)
        self.menu_bar.add_cascade(label="View", menu=view_menu)

        review_menu = tk.Menu(self.menu_bar, tearoff=0)
        review_menu.add_command(label="Undo annotation", accelerator="Z", command=self.undo)
        review_menu.add_command(label="Clear selected label", command=lambda: self.label("unlabeled"))
        review_menu.add_separator()
        for label, value in (("Good", "good"), ("Bad", "bad"), ("Unsure", "unsure")):
            review_menu.add_command(label=label, accelerator=label[0], command=lambda v=value: self.label(v))
        self.menu_bar.add_cascade(label="Review", menu=review_menu)
        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        help_menu.add_command(label="Annotation controls", command=lambda: messagebox.showinfo(
            "Annotation controls", "G / B / U: label a region\nLeft / Right: previous / next region\nZ: undo annotation\nF: fit image\nMouse wheel: zoom\nMiddle-button drag: pan\n\nUse View for overlays and excluded candidates.\nEnable Draw a missed good flake, then drag on the image.\nMark Entire image reviewed after checking the full field.", parent=self.root))
        self.menu_bar.add_cascade(label="Help", menu=help_menu)
        self.root.configure(menu=self.menu_bar)

    def project_details(self):
        if not self.project or self.busy:
            return
        settings = self.project.data["settings"]
        group = self.item()["group"] if self.item() else "No imported images"
        messagebox.showinfo("Annotation project", f"Project: {self.project.folder}\n\nGood means: {self.project.data['good_definition']}\n\nCurrent group: {group}\nMinimum area: {settings['minimum_area_um2']:g} µm²\nP:A filter: {'on' if settings['use_pa_filter'] else 'off'}\nOther geometry filters: {'on' if settings['use_other_geometry'] else 'off'}\n\nProfile: {self.project.data['profile_source']}", parent=self.root)

    def scroll_panel(self, event):
        widget = event.widget
        # Keep the crop's zoom and the list/combobox's native wheel behavior.
        if isinstance(widget, (ImageCanvas, tk.Listbox, ttk.Combobox, ttk.Scrollbar)):
            return
        while widget is not None and widget is not self.sidebar:
            widget = getattr(widget, "master", None)
        if widget is None:
            return
        bounds = self.panel_scroller.bbox("all")
        if bounds and bounds[3] - bounds[1] > self.panel_scroller.winfo_height():
            direction = -1 if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0 else 1
            self.panel_scroller.yview_scroll(direction * 3, "units")
            return "break"

    def _bind_keys(self):
        actions = {"g": lambda: self.label("good"), "b": lambda: self.label("bad"), "u": lambda: self.label("unsure"), "Left": lambda: self.move_region(-1), "Right": lambda: self.move_region(1), "f": self.canvas.fit, "z": self.undo}
        for key, action in actions.items():
            self.root.bind(f"<KeyPress-{key}>", lambda e, a=action: self.hotkey(e, a))

    def hotkey(self, event, action):
        if event.widget.winfo_class() not in {"Entry", "TEntry", "Text", "TCombobox"} and not self.busy:
            action()
            return "break"

    def error(self, exc):
        messagebox.showerror("Region Annotation", str(exc), parent=self.root)

    def new(self):
        if self.busy:
            return
        dialog = NewProjectDialog(self.root)
        self.root.wait_window(dialog)
        if dialog.result:
            project, sources, group = dialog.result
            self.load(project)
            self.start_import(sources, group)

    def open(self):
        if self.busy:
            return
        selected = filedialog.askopenfilename(parent=self.root, title="Open annotation project", filetypes=[("Annotation project", "project.json")])
        if selected:
            try:
                self.load(Project.open(selected))
            except Exception as exc:
                self.error(exc)

    def load(self, project):
        # Upgrade the previous default, preserving explicit custom cutoffs.
        if project.data["settings"]["minimum_area_um2"] == 50:
            project.data["settings"]["minimum_area_um2"] = 200.0
            for item in project.data["images"]:
                for region in item["regions"]:
                    region["excluded_by"] = [reason for reason in region["excluded_by"]
                                             if not reason.startswith("Area below ")]
                    if region["area_um2"] < 200:
                        region["excluded_by"].insert(0, "Area below 200 µm²")
            project.save()
        self.project = project
        cursor = project.data.get("cursor", {})
        self.image_index = min(max(0, cursor.get("image", 0)), max(0, len(project.data["images"]) - 1))
        self.region_index = cursor.get("region", 0)
        self.root.title(f"Flake Search — Region Annotation — {project.folder.name}")
        self.load_image()

    def item(self):
        if not self.project or not self.project.data["images"]:
            return None
        return self.project.data["images"][self.image_index]

    def load_image(self):
        item = self.item()
        self._mask_cache = {}
        try:
            self.rgb = cv2.cvtColor(read_image(self.project.asset(item["asset"])), cv2.COLOR_BGR2RGB) if item else None
            self.canvas.set_image(self.rgb)
            self.refresh_regions()
        except Exception as exc:
            self.rgb = None
            self.canvas.set_image(None)
            self.error(exc)

    def refresh_regions(self):
        if self.busy:
            return
        item = self.item()
        self.region_list.delete(0, "end")
        self.visible_indices = []
        if item:
            for index, region in enumerate(item["regions"]):
                if self.excluded.get() or not region["excluded_by"]:
                    self.visible_indices.append(index)
                    self.region_list.insert("end", f"{index + 1}. {region['label'].upper()} · {region['class']}" + (" · excluded" if region["excluded_by"] else ""))
            if self.region_index not in self.visible_indices:
                self.region_index = self.visible_indices[0] if self.visible_indices else None
            labeled = sum(item["regions"][i]["label"] != "unlabeled" for i in self.visible_indices)
            hidden = len(item["regions"]) - len(self.visible_indices)
            self.image_info.set(f"Image {self.image_index + 1}/{len(self.project.data['images'])}: {Path(item['source']).name}\n{labeled}/{len(self.visible_indices)} labeled · {hidden} filtered out\n{item['calibration']}")
            self.reviewed.set(item["fully_reviewed"])
            self.updating_split = True
            self.split.set(item["split"])
            self.updating_split = False
            self.manual_list["values"] = [f"M{i + 1}: missed good flake" for i in range(len(item["manual_good_boxes"]))]
            if item["manual_good_boxes"]:
                self.manual_list.current(len(item["manual_good_boxes"]) - 1)
            else:
                self.manual_list.set("")
            self.status.set(f"Edits save automatically.\nMinimum area: {self.project.data['settings']['minimum_area_um2']:g} µm²")
        else:
            self.region_index = None
            self.image_info.set("No imported images")
            self.manual_list.set("")
        self.select_region()

    def select_list(self, _event):
        if self.busy:
            return
        selection = self.region_list.curselection()
        if selection and selection[0] < len(self.visible_indices):
            self.region_index = self.visible_indices[selection[0]]
            self.select_region()

    def select_region(self):
        item = self.item()
        self.region_list.selection_clear(0, "end")
        if item and self.region_index in self.visible_indices:
            index = self.visible_indices.index(self.region_index)
            self.region_list.selection_set(index)
            self.region_list.see(index)
            region = item["regions"][self.region_index]
            self.reason.set(region["reason"])
            failures = [r["name"] for r in region["original_filter_results"] if r["enabled"] and not r["passed"]]
            self.details.set(f"Region {self.region_index + 1} · {region['class']} · {region['label'].upper()}\nArea {region['area_um2']:.1f} µm² · P:A {region['perimeter_area_ratio_per_um']:.4f} µm⁻¹\nOriginal geometry: {'reject — ' + ', '.join(failures) if failures else 'pass'}" + (f"\nExcluded: {', '.join(region['excluded_by'])}" if region["excluded_by"] else ""))
        else:
            self.details.set("No visible candidates. Inspect the full image for missed good flakes.")
            self.reason.set("")
        self.refresh_crop()
        self.refresh_overlays()

    def refresh_crop(self):
        item = self.item()
        if self.busy:
            return
        if item and self.region_index is not None and self.rgb is not None:
            try:
                from scanning.region_dataset import build_region_training_image
                region = item["regions"][self.region_index]
                sample = build_region_training_image(self.rgb, {"bounding_box": region["bounding_box"], "instance_mask": self.cached_mask(region)})
                # These are the exact aligned RGB and mask halves exported for
                # training. The viewer never paints label colors into the crop.
                crop = sample[:, 512:] if self.show_crop_mask.get() else sample[:, :512]
                self.crop.set_image(crop)
            except Exception as exc:
                self.error(exc)
        else:
            self.crop.set_image(None)

    def cached_mask(self, region):
        if region["id"] not in self._mask_cache:
            self._mask_cache[region["id"]] = self.project.mask(region)
        return self._mask_cache[region["id"]]

    def select_manual(self):
        if self.busy:
            return
        self.region_index = None
        self.region_list.selection_clear(0, "end")
        self.reason.set("")
        self.details.set(f"Missed good flake M{self.manual_list.current() + 1} (reference box)")
        self.refresh_crop()
        self.refresh_overlays()

    def refresh_overlays(self):
        if self.busy:
            return
        overlays = []
        item = self.item()
        displayed_indices = self.visible_indices if self.display_all.get() else [i for i in self.visible_indices if i == self.region_index]
        if self.rgb is not None:
            try:
                self.canvas.rgb = identified_region_preview(
                    self.rgb, item["regions"], displayed_indices, self.cached_mask,
                ) if item else self.rgb
            except Exception as exc:
                self.error(exc)
                return
        if item:
            for index in displayed_indices:
                region = item["regions"][index]
                overlays.append((region["bounding_box"], COLORS[region["label"]], index == self.region_index, str(index + 1)))
            for index, manual in enumerate(item["manual_good_boxes"]):
                selected = self.region_index is None and index == self.manual_list.current()
                if self.display_all.get() or selected:
                    overlays.append((manual["bounding_box"], COLORS["good"], selected, f"M{index + 1}"))
        self.canvas.overlays = overlays
        self.canvas.render()

    def move_region(self, step):
        if self.busy or not self.visible_indices:
            return
        index = self.visible_indices.index(self.region_index) if self.region_index in self.visible_indices else 0
        self.region_index = self.visible_indices[max(0, min(len(self.visible_indices) - 1, index + step))]
        self.select_region()

    def move_image(self, step):
        if self.busy or not self.item():
            return
        self.image_index = max(0, min(len(self.project.data["images"]) - 1, self.image_index + step))
        self.region_index = None
        self.load_image()

    def label(self, label):
        if self.busy or not self.item() or self.region_index is None:
            return
        try:
            current = self.region_index
            self.project.label(self.image_index, current, label, self.reason.get())
            self.refresh_regions()
            if self.auto_next.get() and label != "unlabeled":
                self.move_region(1)
            self.save_cursor()
        except Exception as exc:
            self.error(exc)

    def save_cursor(self):
        if self.project:
            self.project.data["cursor"] = {"image": self.image_index, "region": self.region_index}
            self.project.save()

    def undo(self):
        if self.busy or not self.project:
            return
        try:
            index = self.project.undo()
            if index is not None:
                self.image_index = index
                self.load_image()
        except Exception as exc:
            self.error(exc)

    def update_draw_mode(self):
        self.canvas.config(cursor="crosshair" if self.draw_mode.get() else "")

    def image_press(self, event):
        self.drag_start = None
        self.pan_press = None
        if self.busy or not self.item():
            return
        if self.draw_mode.get():
            self.drag_start = self.canvas.point(event.x, event.y)
        else:
            self.pan_press = (event.x, event.y)
            self.pan_moved = False
            self.canvas.pan_start(event)

    def select_at(self, event):
        point = self.canvas.point(event.x, event.y)
        if point is None:
            return
        matches = []
        for index in self.visible_indices:
            region = self.item()["regions"][index]
            box = region["bounding_box"]
            x, y = int(point[0] - box["x"]), int(point[1] - box["y"])
            if 0 <= x < box["width"] and 0 <= y < box["height"] and self.cached_mask(region)[y, x]:
                matches.append((region["area_um2"], index))
        if matches:
            self.region_index = min(matches)[1]
            self.select_region()

    def image_drag(self, event):
        if not self.busy and getattr(self, "pan_press", None) is not None:
            x, y = self.pan_press
            if self.pan_moved or abs(event.x - x) + abs(event.y - y) >= 4:
                self.pan_moved = True
                self.canvas.configure(cursor="fleur")
                self.canvas.pan(event)
            return
        if self.busy or not getattr(self, "drag_start", None):
            return
        self.canvas.delete("draft")
        x, y = self.drag_start
        self.canvas.create_rectangle(self.canvas.offset[0] + x * self.canvas.scale, self.canvas.offset[1] + y * self.canvas.scale, event.x, event.y, outline=COLORS["good"], width=2, dash=(4, 2), tags="draft")

    def image_release(self, event):
        if getattr(self, "pan_press", None) is not None:
            self.pan_press = None
            self.update_draw_mode()
            if not self.busy and not self.pan_moved:
                self.select_at(event)
            return
        if self.busy or not getattr(self, "drag_start", None):
            return
        start, self.drag_start = self.drag_start, None
        self.canvas.delete("draft")
        end = self.canvas.point(event.x, event.y, clamp=True)
        if end is None:
            return
        x, y = int(min(start[0], end[0])), int(min(start[1], end[1]))
        box = {"x": x, "y": y, "width": int(max(start[0], end[0])) - x, "height": int(max(start[1], end[1])) - y}
        try:
            self.project.add_box(self.image_index, box)
            self.refresh_regions()
            self.select_manual()
        except Exception as exc:
            self.error(exc)

    def remove_box(self):
        if self.busy or not self.item():
            return
        index = self.manual_list.current()
        if index >= 0:
            try:
                self.project.remove_box(self.image_index, self.item()["manual_good_boxes"][index]["id"])
                self.refresh_regions()
            except Exception as exc:
                self.error(exc)

    def mark_reviewed(self):
        if self.busy or not self.item():
            return
        try:
            self.project.set_reviewed(self.image_index, self.reviewed.get())
        except Exception as exc:
            self.reviewed.set(self.item()["fully_reviewed"])
            self.error(exc)

    def split_changed(self, *_):
        if getattr(self, "updating_split", False) or self.busy or not self.item():
            return
        try:
            self.project.assign_split(self.image_index, self.split.get())
        except Exception as exc:
            self.error(exc)

    def add_images(self):
        if self.busy or not self.project:
            return
        paths = filedialog.askopenfilenames(parent=self.root, filetypes=[("Source images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp")])
        if not paths:
            return
        from tkinter.simpledialog import askstring
        group = askstring("Sample / scan group", "Use the same group for all views of the same physical sample:", initialvalue=self.item()["group"] if self.item() else str(Path(paths[0]).parent), parent=self.root)
        if group and group.strip():
            self.start_import([Path(p) for p in paths], group.strip())

    def add_scan(self):
        self.add_folder(scan=True)

    def add_folder(self, scan=False):
        if self.busy or not self.project:
            return
        selected = filedialog.askdirectory(parent=self.root, title="Add a saved scan" if scan else "Add a folder of source images")
        if not selected:
            return
        try:
            sources = discover_images(selected, scan=scan)
            if not sources:
                raise ValueError("No saved Raw images found in this scan." if scan else "No source images found in this folder.")
            from tkinter.simpledialog import askstring
            group = askstring(
                "Sample / scan group",
                "Use the same group for views of the same physical sample.\nThese images will use this project's existing profile and settings.",
                initialvalue=str(Path(selected).resolve()), parent=self.root,
            )
            if group and group.strip():
                self.start_import(sources, group.strip())
        except Exception as exc:
            self.error(exc)

    def start_import(self, sources, group):
        project = self.project
        def work():
            errors, imported, duplicates = [], 0, 0
            for index, source in enumerate(sources):
                if self.cancelled.is_set():
                    break
                self.events.put(("progress", index, len(sources), f"Detecting {Path(source).name} ({index + 1}/{len(sources)})"))
                try:
                    if project.import_image(source, group):
                        imported += 1
                    else:
                        duplicates += 1
                except Exception as exc:
                    errors.append(f"{source}: {exc}")
            return f"Imported {imported} images; skipped {duplicates} duplicates; {len(errors)} errors." + (" Cancelled between images." if self.cancelled.is_set() else ""), errors
        self.start_worker(work)

    def export(self):
        if self.busy or not self.item():
            return
        project = self.project
        def work():
            target = project.export(lambda n, total: self.events.put(("progress", n, total, f"Exporting {n}/{total} images")), self.cancelled.is_set)
            return f"Dataset exported to {target}", []
        self.start_worker(work)

    def start_worker(self, function):
        self.busy = True
        self.cancelled.clear()
        self.progress.pack(fill="x", pady=(7, 0), before=self.status_label)
        self.status.set("Working… annotation controls are paused until this operation finishes.")
        def worker():
            try:
                self.events.put(("done", *function()))
            except Exception as exc:
                self.events.put(("done", str(exc), [str(exc)]))
        threading.Thread(target=worker, daemon=False).start()

    def poll(self):
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "progress":
                    _, n, total, message = event
                    self.progress.config(maximum=max(1, total), value=n)
                    self.status.set(message)
                else:
                    _, message, errors = event
                    self.busy = False
                    self.progress.pack_forget()
                    self.progress.config(value=self.progress["maximum"])
                    self.load_image()
                    self.status.set(message)
                    if errors:
                        # Persist the complete report; the dialog remains readable.
                        try:
                            (self.project.folder / "last_operation_errors.txt").write_text("\n".join(errors), encoding="utf-8")
                        except OSError:
                            pass
                        messagebox.showwarning("Annotation operation", message + "\n\n" + "\n".join(errors[:6]), parent=self.root)
                    if self.closing:
                        self.close()
                        return
        except queue.Empty:
            pass
        self.poll_job = self.root.after(100, self.poll)

    def close(self):
        if self.busy:
            self.closing = True
            self.cancelled.set()
            self.status.set("Finishing the current image and saving before closing…")
            return
        try:
            self.save_cursor()
        except Exception as exc:
            self.error(exc)
            return
        self.root.after_cancel(self.poll_job)
        self.root.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Standalone region labeling and full-image reference annotation.")
    parser.add_argument("project", nargs="?", type=Path, help="Existing project folder or project.json")
    args = parser.parse_args(argv)
    root = tk.Tk()
    try:
        AnnotationApp(root, args.project)
    except Exception as exc:
        messagebox.showerror("Cannot open annotation project", str(exc), parent=root)
        root.destroy()
        return 1
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
