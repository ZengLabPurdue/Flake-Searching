import json
import math
import re
from pathlib import Path

import tkinter as tk
from tkinter import Frame, Label, filedialog, messagebox
from tkinter import ttk

import cv2
import numpy as np

from scanning import flake_detection
from ui.sparse_tile_viewer import SparseTileViewer


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
FILTERED_MAP_VIEW_PAIRS = {
    "Raw 2x": "Filtered 2x",
    "Raw 10x": "Processed 10x",
}
CAPTURE_IMAGE_VIEWS = {
    "10x Images": "10x",
    "20x Images": "20x",
}
INDIVIDUAL_IMAGE_VIEWS = {"Flakes Found", *CAPTURE_IMAGE_VIEWS}
SCAN_FOLDER_NAME_PATTERN = re.compile(
    r"^(?:Full|100x) Scan \(\d{4}-\d{2}-\d{2}\) \(\d{2}-\d{2}-\d{2}\)$"
)


def is_scan_folder_name(value):
    return bool(SCAN_FOLDER_NAME_PATTERN.fullmatch(str(value)))


def _path_parts(value):
    """Return platform-independent path components from a manifest value."""
    if not isinstance(value, (str, Path)):
        return ()
    return tuple(
        part
        for part in re.split(r"[\\/]+", str(value).strip())
        if part and part != "."
    )


def detection_result_key(record):
    """Map one saved detection to its Flakes Found chip/image key."""
    if not isinstance(record, dict):
        return None
    parts = _path_parts(record.get("result_image"))
    if len(parts) >= 2:
        return parts[-2].casefold(), parts[-1].casefold()

    parts = _path_parts(record.get("source_image"))
    if not parts:
        return None
    filename = parts[-1]
    chip = None
    for index in range(len(parts) - 2, 0, -1):
        if parts[index].casefold() in {"raw", "processed"}:
            chip = parts[index - 1]
            break
    if chip is None and len(parts) >= 2:
        chip = parts[-2]
    if not chip or not filename:
        return None
    return chip.casefold(), filename.casefold()


def result_image_key(path):
    path = Path(path)
    return path.parent.name.casefold(), path.name.casefold()


def _finite_number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first_number(mapping, *keys):
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        number = _finite_number(mapping.get(key))
        if number is not None:
            return number
    return None


def _format_number(value, unit, *, decimal_places=2):
    if value is None:
        return "—"
    text = f"{value:,.{decimal_places}f}"
    if decimal_places:
        text = text.rstrip("0").rstrip(".")
    return f"{text} {unit}" if unit else text


def format_region_statistics(record):
    """Format optional v2 measurements without rejecting legacy records."""
    if not isinstance(record, dict):
        record = {}
    statistics = record.get("statistics")
    if not isinstance(statistics, dict):
        statistics = {}
    bounding_box = record.get("bounding_box_px")
    if not isinstance(bounding_box, dict):
        bounding_box = {}

    area_um2 = _first_number(statistics, "area_um2")
    size_um = _first_number(statistics, "size_um")
    width_um = _first_number(statistics, "width_um", "bounding_width_um")
    height_um = _first_number(statistics, "height_um", "bounding_height_um")
    perimeter_um = _first_number(statistics, "perimeter_um")
    ratio_per_um = _first_number(
        statistics,
        "perimeter_area_ratio_per_um",
        "perimeter_to_area_ratio_per_um",
        "perimeter_area_ratio",
    )

    pixel_count = _first_number(statistics, "pixel_count", "area_px2")
    width_px = _first_number(statistics, "width_px")
    height_px = _first_number(statistics, "height_px")
    perimeter_px = _first_number(statistics, "perimeter_px")
    x_px = _first_number(bounding_box, "x")
    y_px = _first_number(bounding_box, "y")
    if width_px is None:
        width_px = _first_number(bounding_box, "width")
    if height_px is None:
        height_px = _first_number(bounding_box, "height")
    size_px = (
        max(width_px, height_px)
        if width_px is not None and height_px is not None
        else None
    )
    ratio_per_px = (
        perimeter_px / pixel_count
        if perimeter_px is not None and pixel_count is not None and pixel_count > 0
        else None
    )

    classification = record.get("classification")
    classification = str(classification).strip() if classification is not None else ""
    group = record.get("classification_group")
    group = str(group).strip() if group is not None else ""

    area_text = (
        _format_number(area_um2, "µm²", decimal_places=2)
        if area_um2 is not None
        else _format_number(pixel_count, "px²", decimal_places=0)
    )
    width_text = (
        _format_number(width_um, "µm", decimal_places=2)
        if width_um is not None
        else _format_number(width_px, "px", decimal_places=0)
    )
    height_text = (
        _format_number(height_um, "µm", decimal_places=2)
        if height_um is not None
        else _format_number(height_px, "px", decimal_places=0)
    )
    perimeter_text = (
        _format_number(perimeter_um, "µm", decimal_places=2)
        if perimeter_um is not None
        else _format_number(perimeter_px, "px", decimal_places=2)
    )
    size_text = (
        _format_number(size_um, "µm", decimal_places=2)
        if size_um is not None
        else _format_number(size_px, "px", decimal_places=0)
    )
    ratio_text = (
        _format_number(ratio_per_um, "µm⁻¹", decimal_places=4)
        if ratio_per_um is not None
        else _format_number(ratio_per_px, "px⁻¹", decimal_places=4)
    )

    return "\n".join((
        f"Class: {classification or '—'}",
        f"Group: {group or '—'}",
        f"Area: {area_text}",
        f"Longest side: {size_text}",
        f"Width: {width_text}",
        f"Height: {height_text}",
        f"Perimeter: {perimeter_text}",
        f"Perimeter / area: {ratio_text}",
        f"Pixels: {_format_number(pixel_count, 'px', decimal_places=0)}",
        "Box (px): "
        f"x {_format_number(x_px, '', decimal_places=0)}, "
        f"y {_format_number(y_px, '', decimal_places=0)}, "
        f"{_format_number(width_px, '', decimal_places=0)} × "
        f"{_format_number(height_px, '', decimal_places=0)}",
    ))


def _natural_sort_key(value):
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"(\d+)", str(value))
    )


class ViewScansPanel:
    def __init__(
        self,
        parent,
        root,
        app,
    ):
        self.parent = parent
        self.root = root
        self.app = app

        self.view_chip_index = 0
        self.view_image_index = 0
        self.view_scan_path = None
        self.view_folder = None
        self.image_files = None
        self.selected_view = None
        self.sparse_map_mode = False
        self.chip_root = None
        self.chip_source_folder = None
        self.capture_images_mode = False
        self.capture_chip_folder = None
        self.magnification_roots = {}
        self.available_result_views = set()
        self.region_records_by_image = {}
        self.current_regions = []
        self.current_region_key = None
        self.view_region_index = 0
        self.current_result_path = None
        self._processed_region_cache_path = None
        self._processed_region_cache_rgb = None
        self.inspection = None
        self.app.img_label.bind("<Button-1>", self._inspection_image_click, add="+")
        self.tile_viewer = SparseTileViewer(self.app.scan_results_canvas)

        self.results_menu = None
        self.open_scan_menu = None

        self._build_panel()
        self.frame.place_forget()

    def _build_panel(self):
        self.frame = Frame(
            self.parent,
            bg="white",
            width=204,
            height=225,
        )
        self.frame.place(relx=0.0, rely=0.0, anchor="nw")

        self.background = Frame(
            self.frame,
            bg="white",
            width=200,
            height=223,
        )
        self.background.place(x=2, y=0)

        self.content = Frame(self.background, bg="white", width=184)
        self.content.place(x=8, y=8, width=184)
        self.content.columnconfigure(0, weight=1)

        style = ttk.Style()
        style.configure(
            "ScanResults.TCheckbutton",
            background="white",
            foreground="black",
        )
        style.map("ScanResults.TCheckbutton", background=[
            ("disabled", "white"), ("pressed", "white"), ("active", "white"),
        ])
        style.configure(
            "Normal.TButton",
            font="TkDefaultFont",
            background="white",
            relief="flat",
        )

        self.title_label = Label(
            self.content,
            text="Scan Results",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 13),
        )
        self.title_label.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        self.scan_name_var = tk.StringVar(value="Scan: Not Selected")

        self.scan_name_label = Label(
            self.content,
            textvariable=self.scan_name_var,
            bg="white",
            fg="black",
            font="TkDefaultFont",
            justify="center",
            wraplength=176,
        )
        self.scan_name_label.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(0, 8),
        )

        self.chip_var = tk.StringVar()

        self.chip_dropdown = ttk.Combobox(
            self.content,
            textvariable=self.chip_var,
            state="readonly",
            width=1,
        )
        self.chip_dropdown.grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(0, 8),
        )
        self.chip_dropdown.bind("<<ComboboxSelected>>", self._on_chip_selected)

        self.image_var = tk.StringVar(value="Image: None")

        self.image_label = Label(
            self.content,
            textvariable=self.image_var,
            bg="white",
            fg="black",
            font="TkDefaultFont",
            justify="center",
            wraplength=176,
        )
        self.image_label.grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(0, 8),
        )

        self.region_var = tk.StringVar(value="Region: None")
        self.region_statistics_var = tk.StringVar(value="Statistics unavailable")
        self.region_group = Frame(self.content, bg="white")
        self.region_group.columnconfigure(0, weight=1)
        self.region_label = Label(
            self.region_group,
            textvariable=self.region_var,
            bg="white",
            fg="black",
            font=("TkDefaultFont", 10, "bold"),
            justify="center",
            wraplength=176,
        )
        self.region_label.grid(row=0, column=0, sticky="ew")
        self.region_statistics_label = Label(
            self.region_group,
            textvariable=self.region_statistics_var,
            bg="white",
            fg="#333333",
            justify="left",
            anchor="w",
            wraplength=176,
        )
        self.region_statistics_label.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(3, 5),
        )
        self.region_button_panel = Frame(self.region_group, bg="white")
        self.region_button_panel.grid(row=2, column=0, sticky="ew")
        self.region_button_panel.columnconfigure(0, weight=1)
        self.region_button_panel.columnconfigure(1, weight=1)
        self.btn_previous_region = ttk.Button(
            self.region_button_panel,
            style="Normal.TButton",
            text="\u25c2 Region",
            command=self.previous_region,
        )
        self.btn_next_region = ttk.Button(
            self.region_button_panel,
            style="Normal.TButton",
            text="Region \u25b8",
            command=self.next_region,
        )
        self.btn_previous_region.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 2),
        )
        self.btn_next_region.grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(2, 0),
        )
        self.display_everything_var = tk.BooleanVar(value=False)
        self.display_everything_checkbox = ttk.Checkbutton(
            self.region_group,
            text="Display everything",
            variable=self.display_everything_var,
            command=self._toggle_region_display,
            style="ScanResults.TCheckbutton",
        )
        self.display_everything_checkbox.grid(
            row=3,
            column=0,
            sticky="w",
            pady=(4, 0),
        )
        self.inspect_button = ttk.Button(
            self.region_group, text="Image under 100X", style="Normal.TButton",
            command=self.start_inspection,
        )
        self.inspect_button.grid(row=4, column=0, sticky="ew", pady=(5, 0))
        self.inspection_panel = tk.LabelFrame(self.content, text="Save Flake", bg="white",
                                             fg="black", borderwidth=0, highlightthickness=0)
        self.inspection_status = tk.StringVar(value="")
        Label(self.inspection_panel, textvariable=self.inspection_status, bg="white", fg="#333333",
                  wraplength=156).pack(fill="x", padx=6, pady=6)
        self.confirm_inspection_button = ttk.Button(
            self.inspection_panel, text="Save under 100X", command=self.confirm_inspection,
            style="Normal.TButton",
        )
        self.confirm_inspection_button.pack(fill="x", padx=6, pady=3)
        self.refocus_inspection_button = ttk.Button(
            self.inspection_panel, text="Refocus at 100X", command=self.refocus_inspection,
            style="Normal.TButton",
        )
        self.refocus_inspection_button.pack(fill="x", padx=6, pady=3)
        ttk.Button(self.inspection_panel, text="Back to results",
                   style="Normal.TButton",
                   command=self.end_inspection).pack(fill="x", padx=6, pady=3)

        self.filtered_map_var = tk.BooleanVar(value=False)
        self.filtered_map_checkbox = ttk.Checkbutton(
            self.content,
            text="Filtered map",
            variable=self.filtered_map_var,
            command=self._toggle_filtered_map,
            style="ScanResults.TCheckbutton",
        )

        self.raw_result_var = tk.BooleanVar(value=False)
        self.raw_result_checkbox = ttk.Checkbutton(
            self.content,
            text="Show raw image",
            variable=self.raw_result_var,
            command=self._toggle_raw_result,
            style="ScanResults.TCheckbutton",
        )

        self.button_panel = Frame(
            self.content,
            bg="white",
            width=80,
            height=32,
        )
        self.button_panel.grid_propagate(False)

        controls = Frame(self.button_panel, bg="white")
        controls.pack(expand=True, fill="both")

        self.btn_next = ttk.Button(controls, text="▸", style="Arrow.TButton")
        self.btn_previous = ttk.Button(controls, text="◂", style="Arrow.TButton")

        self.btn_next.bind("<ButtonPress-1>", self.next_image)
        self.btn_previous.bind("<ButtonPress-1>", self.previous_image)

        self.root.bind("<Left>", self.previous_image)
        self.root.bind("<Right>", self.next_image)

        controls.rowconfigure(0, weight=1)
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)

        self.btn_previous.grid(row=0, column=0, sticky="nsew")
        self.btn_next.grid(row=0, column=1, sticky="nsew")

        self._show_chip = True
        self._show_navigation = True
        self._show_filter_toggle = False
        self._show_raw_result_toggle = False
        self._show_region_details = False
        self._resize_job = None
        self.scan_name_var.trace_add("write", self._queue_panel_resize)
        self.image_var.trace_add("write", self._queue_panel_resize)
        self.region_var.trace_add("write", self._queue_panel_resize)
        self.region_statistics_var.trace_add("write", self._queue_panel_resize)
        self._layout_controls(
            show_chip=True,
            show_navigation=True,
            show_filter_toggle=False,
            show_region_details=False,
        )

    def _queue_panel_resize(self, *_args):
        if self._resize_job is None:
            self._resize_job = self.frame.after_idle(self._refresh_panel_height)

    def _refresh_panel_height(self):
        self._resize_job = None
        self.content.update_idletasks()
        visible_elements = (
            self.title_label,
            self.scan_name_label,
            self.chip_dropdown,
            self.image_label,
            self.region_group,
            self.filtered_map_checkbox,
            self.raw_result_checkbox,
            self.button_panel,
            self.inspection_panel,
        )
        bottom = max(
            (
                element.winfo_y() + element.winfo_height()
                for element in visible_elements
                if element.winfo_manager()
            ),
            default=0,
        )
        height = max(96, self.content.winfo_y() + bottom + 15)
        self.frame.configure(height=height)
        self.background.configure(height=max(1, height - 2))

    def _layout_controls(
        self,
        show_chip=None,
        show_navigation=None,
        show_filter_toggle=None,
        show_raw_result_toggle=None,
        show_region_details=None,
    ):
        if show_chip is not None:
            self._show_chip = bool(show_chip)
        if show_navigation is not None:
            self._show_navigation = bool(show_navigation)
        if show_filter_toggle is not None:
            self._show_filter_toggle = bool(show_filter_toggle)
        if show_raw_result_toggle is not None:
            self._show_raw_result_toggle = bool(show_raw_result_toggle)
        if show_region_details is not None:
            self._show_region_details = bool(show_region_details)
        if self._show_chip:
            self.chip_dropdown.grid()
        else:
            self.chip_dropdown.grid_remove()

        if self._show_region_details:
            self.region_group.grid(
                row=4,
                column=0,
                sticky="ew",
                pady=(0, 8),
            )
        else:
            self.region_group.grid_remove()

        if self._show_filter_toggle:
            self.filtered_map_checkbox.grid(
                row=5,
                column=0,
                sticky="w",
                pady=(0, 8),
            )
        else:
            self.filtered_map_checkbox.grid_remove()

        if self._show_raw_result_toggle:
            self.raw_result_checkbox.grid(
                row=6,
                column=0,
                sticky="w",
                pady=(0, 8),
            )
        else:
            self.raw_result_checkbox.grid_remove()

        self.button_panel.grid_remove()
        if self._show_navigation:
            self.button_panel.grid(
                row=7,
                column=0,
                pady=(0, 2),
            )
        self._queue_panel_resize()

    def add_to_menu(self, parent_menu):
        self.results_menu = tk.Menu(parent_menu, tearoff=0)

        self.results_menu.add_command(
            label="Open Scan...",
            command=self.open_scan
        )
        self.results_menu.add_separator()

        self.open_scan_menu = tk.Menu(self.results_menu, tearoff=0)
        self._rebuild_open_scan_menu()

        self.results_menu.add_cascade(
            label="View Scan",
            state="disabled",
            menu=self.open_scan_menu
        )

        self.results_menu.add_command(
            label="Classify Flakes",
            state="disabled",
            command=None
        )

        parent_menu.add_cascade(label="Results", menu=self.results_menu)

    def _rebuild_open_scan_menu(self):
        if self.open_scan_menu is None:
            return
        self.open_scan_menu.delete(0, "end")

        groups = []
        two_x_items = []
        if "Raw 2x" in self.available_result_views:
            two_x_items.append(("2x Map", "Raw 2x", "normal"))
        elif self.view_scan_path is None:
            two_x_items.append(("2x Map", "Raw 2x", "disabled"))
        if "Filtered 2x" in self.available_result_views:
            two_x_items.append(("Filtered 2x Map", "Filtered 2x", "normal"))
        if "Scan Windows" in self.available_result_views:
            two_x_items.append(("2x Scan Windows", "Scan Windows", "normal"))
        if two_x_items:
            groups.append(two_x_items)

        for magnification in ("10x", "20x"):
            items = []
            raw_view = f"Raw {magnification[:-1]}x"
            processed_view = f"Processed {magnification[:-1]}x"
            if raw_view in self.available_result_views:
                items.append((f"{magnification} Map", raw_view, "normal"))
            if processed_view in self.available_result_views:
                items.append(
                    (
                        f"Processed {magnification} Map",
                        processed_view,
                        "normal",
                    )
                )
            image_view = f"{magnification} Images"
            if image_view in self.available_result_views:
                items.append((image_view, image_view, "normal"))
            if items:
                groups.append(items)

        groups.append([
            (
                "Detected Flakes",
                "Flakes Found",
                (
                    "normal"
                    if "Flakes Found" in self.available_result_views
                    else "disabled"
                ),
            )
        ])

        for group_index, group in enumerate(groups):
            if group_index:
                self.open_scan_menu.add_separator()
            for label, selected_view, state in group:
                self.open_scan_menu.add_command(
                    label=label,
                    state=state,
                    command=lambda view=selected_view: self.set_view_folder(view),
                )

    def _discover_result_layers(self):
        roots = {}
        available = set()
        if self.view_scan_path is None:
            return roots, available

        all_images = self.view_scan_path / "All Images"
        try:
            candidates = [path for path in all_images.iterdir() if path.is_dir()]
        except OSError:
            candidates = []
        for candidate in candidates:
            normalized = re.sub(r"[\s_-]+", "", candidate.name).casefold()
            if normalized in {"2x", "10x", "20x"}:
                roots[normalized] = candidate

        maps = self.view_scan_path / "Maps"
        two_x_root = roots.get("2x")
        if (
            self._folder_has_images(self._named_child(two_x_root, "Raw"))
            or (maps / "map_2x.png").is_file()
        ):
            available.add("Raw 2x")
        if (
            self._folder_has_images(self._named_child(two_x_root, "Filtered"))
            or (maps / "map_2x_filtered.png").is_file()
        ):
            available.add("Filtered 2x")
        if (maps / "map_2x_scan_windows.png").is_file():
            available.add("Scan Windows")

        for magnification in ("10x", "20x"):
            root = roots.get(magnification)
            if self._chip_folders(root, source_name="Raw"):
                available.add(f"Raw {magnification[:-1]}x")
            if self._chip_folders(root, source_name="Processed"):
                available.add(f"Processed {magnification[:-1]}x")
            if self._capture_chip_folders(root):
                available.add(f"{magnification} Images")

        flakes_root = self.view_scan_path / "Flakes Found"
        if self._chip_folders(flakes_root, require_images=True):
            available.add("Flakes Found")
        return roots, available

    def _magnification_root(self, magnification, base_path):
        return (
            self.magnification_roots.get(magnification.casefold())
            or self._named_child(base_path, magnification)
            or base_path / magnification
        )

    @staticmethod
    def _named_child(parent, name):
        if parent is None or not parent.is_dir():
            return None
        try:
            for child in parent.iterdir():
                if child.is_dir() and child.name.casefold() == name.casefold():
                    return child
        except OSError:
            return None
        return None

    @staticmethod
    def _folder_has_images(folder):
        if folder is None or not folder.is_dir():
            return False
        try:
            return any(
                path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                for path in folder.iterdir()
            )
        except OSError:
            return False

    def _chip_folders(self, root, source_name=None, require_images=False):
        if root is None or not root.is_dir():
            return []
        try:
            folders = sorted(
                [path for path in root.iterdir() if path.is_dir()],
                key=lambda folder: _natural_sort_key(folder.name),
            )
        except OSError:
            return []

        matching = []
        for folder in folders:
            if source_name is not None:
                source_folder = self._named_child(folder, source_name)
                if not self._folder_has_images(source_folder):
                    continue
            elif require_images and not self._folder_has_images(folder):
                continue
            matching.append(folder)
        return matching

    def _capture_chip_folders(self, root):
        return [
            folder
            for folder in self._chip_folders(root)
            if (
                self._folder_has_images(self._named_child(folder, "Raw"))
                or self._folder_has_images(
                    self._named_child(folder, "Processed")
                )
            )
        ]

    def show(self):
        self.app.close_all_panels()
        self.frame.place(relx=0.0, rely=0.0, anchor="nw")

    def hide(self):
        self.end_inspection(restore_view=False)
        self.frame.place_forget()

    def pause(self):
        self.tile_viewer.pause()

    def shutdown(self):
        self.end_inspection(restore_view=False)
        self.tile_viewer.shutdown()

    def display_chip_dropdown(self, display=True):
        self._layout_controls(show_chip=display)

    @staticmethod
    def _map_layer_pair(selected_view):
        if selected_view in FILTERED_MAP_VIEW_PAIRS:
            return selected_view, FILTERED_MAP_VIEW_PAIRS[selected_view]
        for raw_view, filtered_view in FILTERED_MAP_VIEW_PAIRS.items():
            if selected_view == filtered_view:
                return raw_view, filtered_view
        return None

    def _configure_filtered_map_toggle(self, selected_view):
        pair = self._map_layer_pair(selected_view)
        available = bool(
            pair
            and pair[0] in self.available_result_views
            and pair[1] in self.available_result_views
        )
        self.filtered_map_var.set(
            bool(pair and selected_view == pair[1])
        )
        self._layout_controls(show_filter_toggle=available)

    def _toggle_filtered_map(self):
        pair = self._map_layer_pair(self.selected_view)
        if pair is None:
            self.filtered_map_var.set(False)
            return

        raw_view, filtered_view = pair
        target_view = (
            filtered_view
            if self.filtered_map_var.get()
            else raw_view
        )
        if target_view not in self.available_result_views:
            self.filtered_map_var.set(self.selected_view == filtered_view)
            return

        self.set_view_folder(
            target_view,
            view_state=self.tile_viewer.capture_view_state(),
            preserve_context=True,
        )

    def _toggle_raw_result(self):
        if (
            self.selected_view != "Flakes Found"
            and not self.capture_images_mode
        ):
            self.raw_result_var.set(False)
            return
        self.display_current_image(preserve_region=True)

    @staticmethod
    def _load_region_records(scan_path):
        """Load and index optional result metadata without blocking image use."""
        manifest_path = Path(scan_path) / "flakes_found.json"
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        detections = payload.get("detections")
        if not isinstance(detections, list):
            return {}

        detection_model = str(payload.get("detection_model", "")).casefold()
        indexed = {}
        for record in detections:
            if not isinstance(record, dict) or record.get("accepted") is False:
                continue
            if detection_model == "flake detection" and "class_id" in record:
                try:
                    is_good_flake = int(record["class_id"]) == 1
                except (TypeError, ValueError, OverflowError):
                    is_good_flake = False
                if not is_good_flake:
                    continue
            key = detection_result_key(record)
            if key is not None:
                indexed.setdefault(key, []).append(record)
        return indexed

    def _clear_current_regions(self):
        self.end_inspection()
        self.current_regions = []
        self.current_region_key = None
        self.view_region_index = 0
        self.current_result_path = None
        self._processed_region_cache_path = None
        self._processed_region_cache_rgb = None
        if hasattr(self, "region_var"):
            self.region_var.set("Region: None")
            self.region_statistics_var.set("Statistics unavailable")
            self._configure_region_buttons(0)
        viewer = getattr(self, "tile_viewer", None)
        if viewer is not None and hasattr(viewer, "set_highlight"):
            viewer.set_highlight(None)

    def _set_current_regions(self, result_path, preserve_region=False):
        self.end_inspection()
        key = result_image_key(result_path)
        same_image = key == self.current_region_key
        self.current_region_key = key
        self.current_regions = list(
            getattr(self, "region_records_by_image", {}).get(key, ())
        )
        if not preserve_region or not same_image:
            self.view_region_index = 0
        elif self.current_regions:
            self.view_region_index %= len(self.current_regions)
        else:
            self.view_region_index = 0
        self._show_current_region()

    def _configure_region_buttons(self, region_count):
        if not hasattr(self, "btn_previous_region"):
            return
        state = ["!disabled"] if region_count > 1 else ["disabled"]
        self.btn_previous_region.state(state)
        self.btn_next_region.state(state)
        if hasattr(self, "display_everything_checkbox"):
            display_state = ["!disabled"] if region_count else ["disabled"]
            self.display_everything_checkbox.state(display_state)
        self._configure_database_button()

    def _configure_database_button(self):
        inspect_button = getattr(self, "inspect_button", None)
        if inspect_button is not None:
            regions = getattr(self, "current_regions", ())
            record = regions[self.view_region_index % len(regions)] if regions else {}
            enabled = bool(record.get("location", {}).get("stage_position_um")) and "class_id" not in record
            enabled = enabled and not self.app.scan_manager.is_scan_running()
            inspect_button.state(["!disabled"] if enabled else ["disabled"])
        button = getattr(self, "save_database_button", None)
        if button is None:
            return
        regions = getattr(self, "current_regions", ())
        selected = (
            regions[self.view_region_index % len(regions)]
            if regions
            else None
        )
        enabled = (
            isinstance(selected, dict)
            and bool(selected.get("image_20x"))
            and bool(selected.get("image_100x"))
            and getattr(self, "view_scan_path", None) is not None
        )
        button.state(["!disabled"] if enabled else ["disabled"])

    def _show_current_region(self, fit_image=False):
        regions = getattr(self, "current_regions", ())
        if not regions:
            if hasattr(self, "region_var"):
                self.region_var.set("Region: None")
                self.region_statistics_var.set("Statistics unavailable")
                self._configure_region_buttons(0)
            self._update_region_highlight()
            return

        self.view_region_index %= len(regions)
        record = regions[self.view_region_index]
        if hasattr(self, "region_var"):
            self.region_var.set(
                f"Region {self.view_region_index + 1} of {len(regions)}"
            )
            self.region_statistics_var.set(format_region_statistics(record))
            self._configure_region_buttons(len(regions))
        self._update_region_highlight(fit_image=fit_image)

    def _toggle_region_display(self):
        self._update_region_highlight(fit_image=True)

    def _update_region_highlight(self, fit_image=False):
        """Render accepted flake masks in their class colors on black."""
        viewer = getattr(self, "tile_viewer", None)
        if viewer is None:
            return
        if hasattr(viewer, "set_highlight"):
            viewer.set_highlight(None)
        raw_var = getattr(self, "raw_result_var", None)
        if raw_var is not None and raw_var.get():
            return
        if not self.current_regions or self.current_result_path is None:
            return

        display_var = getattr(self, "display_everything_var", None)
        show_everything = (
            bool(display_var.get()) if display_var is not None else False
        )
        processed_path = self._processed_result_path(self.current_result_path)
        if processed_path is None:
            processed_path = self.current_result_path
        image_rgb = self._load_processed_region_image(processed_path)
        if image_rgb is None:
            return
        records = self.current_regions if show_everything else [
            self.current_regions[self.view_region_index % len(self.current_regions)]
        ]
        classified_rgb = np.zeros_like(image_rgb)
        for record in records:
            self._paint_region_mask(classified_rgb, image_rgb, record)
        image_rgb = classified_rgb

        view_state = None
        if not fit_image and hasattr(viewer, "capture_view_state"):
            view_state = viewer.capture_view_state()
        if hasattr(viewer, "load_rgb_image"):
            viewer.load_rgb_image(
                image_rgb,
                title=self.current_result_path.name,
                nearest=True,
                view_state=view_state,
            )

    def _load_processed_region_image(self, path):
        path = Path(path)
        if path == self._processed_region_cache_path:
            return self._processed_region_cache_rgb.copy()
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            return None
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self._processed_region_cache_path = path
        self._processed_region_cache_rgb = image_rgb
        return image_rgb.copy()

    @staticmethod
    def _decoded_region_mask(record):
        encoded = record.get("region_mask") if isinstance(record, dict) else None
        if not isinstance(encoded, dict):
            return None
        if encoded.get("encoding") != "foreground-runs-row-major":
            return None
        try:
            width = int(encoded["width"])
            height = int(encoded["height"])
            runs = list(encoded["runs"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if width <= 0 or height <= 0 or len(runs) % 2:
            return None
        flat = np.zeros(width * height, dtype=bool)
        try:
            for offset in range(0, len(runs), 2):
                start = int(runs[offset])
                length = int(runs[offset + 1])
                if start < 0 or length <= 0 or start + length > flat.size:
                    return None
                flat[start:start + length] = True
        except (TypeError, ValueError, OverflowError):
            return None
        return flat.reshape(height, width)

    @classmethod
    def _focused_region_image(cls, image_rgb, record):
        focused = np.zeros_like(image_rgb)
        return cls._paint_region_mask(focused, image_rgb, record)

    @classmethod
    def _paint_region_mask(cls, focused, image_rgb, record):
        """Add an exact accepted mask, with class-color fallback for old scans."""
        if not isinstance(record, dict):
            return focused
        box = record.get("bounding_box_px")
        if not isinstance(box, dict):
            return focused
        try:
            x = int(round(float(box["x"])))
            y = int(round(float(box["y"])))
            width = max(1, int(round(float(box["width"]))))
            height = max(1, int(round(float(box["height"]))))
        except (KeyError, TypeError, ValueError, OverflowError):
            return focused
        image_height, image_width = image_rgb.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(image_width, x + width), min(image_height, y + height)
        if x1 >= x2 or y1 >= y2:
            return focused

        source = image_rgb[y1:y2, x1:x2]
        local_mask = cls._decoded_region_mask(record)
        if local_mask is not None:
            mask_x1, mask_y1 = x1 - x, y1 - y
            local_mask = local_mask[
                mask_y1:mask_y1 + (y2 - y1),
                mask_x1:mask_x1 + (x2 - x1),
            ]
        if local_mask is None or local_mask.shape != source.shape[:2]:
            color = record.get("display_color_rgb")
            try:
                color = np.asarray(color, dtype=np.uint8).reshape(1, 1, 3)
                # Black and white belong to the old preview background, not
                # to a reliably recoverable class mask.
                if np.all(color == 0) or np.all(color == 255):
                    return focused
                local_mask = np.all(source == color, axis=2)
            except (TypeError, ValueError):
                return focused
        target = focused[y1:y2, x1:x2]
        color = record.get("display_color_rgb")
        try:
            color = tuple(int(channel) for channel in color)
            if len(color) != 3 or any(channel < 0 or channel > 255 for channel in color):
                color = (255, 255, 255)
        except (TypeError, ValueError):
            color = (255, 255, 255)
        target[local_mask] = color
        return focused

    def start_inspection(self):
        if not self.current_regions or self.view_scan_path is None:
            return
        from scanning.flake_inspection import FlakeInspection
        manager = self.app.scan_manager
        if manager.is_scan_running():
            messagebox.showinfo("Microscope busy", "Wait for the current operation to finish.", parent=self.root)
            return
        record = self.current_regions[self.view_region_index % len(self.current_regions)]
        session = FlakeInspection(
            self.view_scan_path, record,
            lambda event, value: self._inspection_update(session, event, value),
        )
        self.inspection = session
        self.inspection_panel.grid(row=8, column=0, sticky="ew", pady=(8, 0))
        self.inspection_status.set(f"{record.get('classification', 'Flake')}: preparing inspection…")
        self.confirm_inspection_button.state(["disabled"])
        self.refocus_inspection_button.state(["disabled"])
        self.inspect_button.state(["disabled"])
        self._queue_panel_resize()
        self.app.set_live_mapping(False)
        self.app.set_view("Flake Inspection")
        try:
            manager.start_scan(
                flake_inspection=session,
                on_error=lambda error: messagebox.showerror("Flake inspection", str(error), parent=self.root),
            )
        except Exception as error:
            self.end_inspection()
            messagebox.showerror("Flake inspection", str(error), parent=self.root)

    def _inspection_update(self, session, event, value):
        if event == "finished":
            # The worker releases its hardware lock immediately after this
            # notification. Refresh availability on a subsequent UI tick.
            self.root.after(100, self._configure_database_button)
        if getattr(self, "inspection", None) is not session:
            return
        if event == "finished":
            self.end_inspection()
            return
        if event == "error":
            messagebox.showerror("Flake inspection", value, parent=self.root)
            return
        if event == "center":
            self.inspection_status.set("Automatic centering was uncertain. Click inside the intended flake in this 10X image, or return to results.")
            self.confirm_inspection_button.state(["disabled"])
            self.refocus_inspection_button.state(["disabled"])
            self.app.set_view("Flake Centering")
            self.app.display_image(value, color_order="BGR")
            return
        self.inspection_status.set(value)
        ready = event == "ready" and not session.saved
        self.confirm_inspection_button.state(["!disabled"] if ready else ["disabled"])
        self.refocus_inspection_button.state(["!disabled"] if event in ("ready", "saved") else ["disabled"])
        if event == "saved":
            self.region_records_by_image = self._load_region_records(self.view_scan_path)
            self.current_regions = list(self.region_records_by_image.get(self.current_region_key, ()))

    def confirm_inspection(self):
        session = getattr(self, "inspection", None)
        if session is not None and session.ready and not session.saved:
            self.confirm_inspection_button.state(["disabled"])
            self.refocus_inspection_button.state(["disabled"])
            session.confirm.set()

    def _inspection_image_click(self, event):
        session = getattr(self, "inspection", None)
        if session is None or self.app.get_view() != "Flake Centering":
            return
        point = self.app.display_to_image_point(event.x, event.y)
        if point is None:
            return
        session.center_point = point
        self.app.set_view("Flake Inspection")
        self.inspection_status.set("Centering the selected point…")
        session.center_selected.set()

    def refocus_inspection(self):
        session = getattr(self, "inspection", None)
        if session is not None and session.ready:
            self.confirm_inspection_button.state(["disabled"])
            self.refocus_inspection_button.state(["disabled"])
            session.refocus.set()

    def end_inspection(self, restore_view=True):
        session = getattr(self, "inspection", None)
        if session is None:
            return
        self.inspection = None
        self.app.scan_manager.stop_scan()
        self.inspection_panel.grid_remove()
        self._queue_panel_resize()
        if restore_view and self.app.get_view() in ("Flake Inspection", "Flake Centering"):
            self.app.set_view("Scan Results")
            self._update_region_highlight(fit_image=True)

    def save_selected_region_to_database(self):
        if not self.current_regions or self.view_scan_path is None:
            return
        record = self.current_regions[
            self.view_region_index % len(self.current_regions)
        ]
        try:
            exported = flake_detection.save_database(self.view_scan_path, record)
        except (OSError, TypeError, ValueError) as exc:
            messagebox.showerror(
                "Database Export Error",
                str(exc),
                parent=self.root,
            )
            return
        messagebox.showinfo(
            "Database Export Complete",
            "Saved the selected 20X image, 100X image, and JSON to:\n"
            f"{exported['json'].parent}",
            parent=self.root,
        )

    def previous_region(self, event=None):
        self.end_inspection()
        if self.app.get_view() != "Scan Results" or not self.current_regions:
            return
        self.view_region_index = (
            self.view_region_index - 1
        ) % len(self.current_regions)
        self._show_current_region(fit_image=True)
        self.root.focus_set()

    def next_region(self, event=None):
        self.end_inspection()
        if self.app.get_view() != "Scan Results" or not self.current_regions:
            return
        self.view_region_index = (
            self.view_region_index + 1
        ) % len(self.current_regions)
        self._show_current_region(fit_image=True)
        self.root.focus_set()

    def open_scan(self):
        folder = filedialog.askdirectory(title="Select Scan Folder")
        if not folder:
            return

        path = Path(folder)
        folder_name = path.name

        if not is_scan_folder_name(folder_name):
            messagebox.showwarning(
                "Invalid Folder",
                "Selected folder is not a valid scan folder."
            )
            return

        self.view_scan_path = path
        self.region_records_by_image = self._load_region_records(path)
        self._clear_current_regions()
        self.scan_name_var.set(folder_name)
        self.selected_view = None
        self.chip_var.set("")
        self.filtered_map_var.set(False)
        self.raw_result_var.set(False)
        self._layout_controls(
            show_filter_toggle=False,
            show_raw_result_toggle=False,
            show_region_details=False,
        )
        self.magnification_roots, self.available_result_views = (
            self._discover_result_layers()
        )
        self._rebuild_open_scan_menu()

        if self.results_menu is not None:
            self.results_menu.entryconfig("View Scan", state="normal")
            self.results_menu.entryconfig("Classify Flakes", state="normal")

        # Detected flakes are the landing page even for a zero-flake scan; the
        # latter gets a quiet empty state instead of an interrupting warning.
        self.set_view_folder("Flakes Found")

    def set_view_folder(
        self,
        selected_view,
        view_state=None,
        preserve_context=False,
    ):
        if self.view_scan_path is None:
            messagebox.showwarning(
                "No Scan Selected",
                "Please open a scan folder first."
            )
            return

        self.app.set_view("Scan Results")
        self.show()

        previous_chip_name = (
            self.chip_var.get()
            if preserve_context
            else None
        )
        previous_view = self.selected_view
        if not preserve_context:
            self.view_chip_index = 0
            self.view_image_index = 0
        self.selected_view = selected_view
        if selected_view != previous_view or selected_view != "Flakes Found":
            self._clear_current_regions()
        self.sparse_map_mode = selected_view not in INDIVIDUAL_IMAGE_VIEWS
        self.chip_root = None
        self.chip_source_folder = None
        self.capture_images_mode = False
        self.capture_chip_folder = None
        self._configure_filtered_map_toggle(selected_view)
        if selected_view not in INDIVIDUAL_IMAGE_VIEWS:
            self.raw_result_var.set(False)
        self._layout_controls(
            show_chip=False,
            show_navigation=False,
            show_raw_result_toggle=False,
            show_region_details=False,
        )

        base_path = self.view_scan_path / "All Images"

        if selected_view == "Raw 2x":
            magnification_root = self._magnification_root("2x", base_path)
            self.chip_var.set("")
            self.view_folder = self._named_child(magnification_root, "Raw")
            self.display_chip_dropdown(False)
            if self.view_folder is None:
                self._layout_controls(
                    show_chip=False,
                    show_navigation=False,
                )
                self._load_flattened_map(
                    "map_2x.png",
                    "Raw 2x",
                    False,
                    view_state=view_state,
                )
                return

        elif selected_view == "Filtered 2x":
            magnification_root = self._magnification_root("2x", base_path)
            self.chip_var.set("")
            self.view_folder = self._named_child(magnification_root, "Filtered")
            self.display_chip_dropdown(False)
            self._layout_controls(
                show_chip=False,
                show_navigation=False,
            )
            if self.view_folder is not None:
                self.load_current_folder(view_state=view_state)
            else:
                self._load_flattened_map(
                    "map_2x_filtered.png",
                    "Filtered 2x Map",
                    True,
                    view_state=view_state,
                )
            return

        elif selected_view == "Scan Windows":
            self.chip_var.set("")
            self.view_folder = self.view_scan_path / "Maps"
            self.display_chip_dropdown(False)
            self._layout_controls(
                show_chip=False,
                show_navigation=False,
            )
            self._load_flattened_map(
                "map_2x_scan_windows.png",
                "2x Scan Windows",
                False,
                view_state=view_state,
            )
            return

        elif selected_view in CAPTURE_IMAGE_VIEWS:
            magnification = CAPTURE_IMAGE_VIEWS[selected_view]
            magnification_root = self._magnification_root(
                magnification,
                base_path,
            )
            chip_folder = self.get_capture_subfolder(
                magnification_root,
                self.view_chip_index,
                preferred_name=previous_chip_name,
            )
            if chip_folder is None:
                self._show_missing_folder(
                    f"No {magnification} image folders found."
                )
                return

            self.capture_images_mode = True
            self.capture_chip_folder = chip_folder
            self.view_folder = self._preferred_capture_folder(chip_folder)
            self.chip_root = magnification_root
            self.display_chip_dropdown(True)
            self.populate_chips_dropdown(
                magnification_root,
                selected_name=chip_folder.name,
                capture_images=True,
            )
            self._configure_capture_source_toggle()

        elif selected_view == "Raw 10x":
            magnification_root = self._magnification_root("10x", base_path)
            chip_folder = self.get_subfolder(
                magnification_root,
                self.view_chip_index,
                source_name="Raw",
                preferred_name=previous_chip_name,
            )
            if chip_folder is None:
                self._show_missing_folder("No 10x chip folders found.")
                return

            self.view_folder = self._named_child(chip_folder, "Raw")
            self.chip_root = magnification_root
            self.chip_source_folder = "Raw"
            self.display_chip_dropdown(True)
            self.populate_chips_dropdown(
                magnification_root,
                source_name="Raw",
                selected_name=chip_folder.name,
            )

        elif selected_view == "Processed 10x":
            magnification_root = self._magnification_root("10x", base_path)
            chip_folder = self.get_subfolder(
                magnification_root,
                self.view_chip_index,
                source_name="Processed",
                preferred_name=previous_chip_name,
            )
            if chip_folder is None:
                self._show_missing_folder("No 10x chip folders found.")
                return

            self.view_folder = self._named_child(chip_folder, "Processed")
            self.chip_root = magnification_root
            self.chip_source_folder = "Processed"
            self.display_chip_dropdown(True)
            self.populate_chips_dropdown(
                magnification_root,
                source_name="Processed",
                selected_name=chip_folder.name,
            )

        elif selected_view == "Raw 20x":
            magnification_root = self._magnification_root("20x", base_path)
            chip_folder = self.get_subfolder(
                magnification_root,
                self.view_chip_index,
                source_name="Raw",
            )
            if chip_folder is None:
                self._show_missing_folder("No 20x chip folders found.")
                return

            self.view_folder = self._named_child(chip_folder, "Raw")
            self.chip_root = magnification_root
            self.chip_source_folder = "Raw"
            self.display_chip_dropdown(True)
            self.populate_chips_dropdown(magnification_root, source_name="Raw")

        elif selected_view == "Processed 20x":
            magnification_root = self._magnification_root("20x", base_path)
            chip_folder = self.get_subfolder(
                magnification_root,
                self.view_chip_index,
                source_name="Processed",
            )
            if chip_folder is None:
                self._show_missing_folder("No 20x chip folders found.")
                return

            self.view_folder = self._named_child(chip_folder, "Processed")
            self.chip_root = magnification_root
            self.chip_source_folder = "Processed"
            self.display_chip_dropdown(True)
            self.populate_chips_dropdown(
                magnification_root,
                source_name="Processed",
            )

        elif selected_view == "Flakes Found":
            flakes_root = self.view_scan_path / "Flakes Found"
            chip_folder = self.get_subfolder(
                flakes_root,
                self.view_chip_index,
                require_images=True,
            )
            if chip_folder is None:
                self.view_folder = flakes_root
                self.image_files = []
                self.chip_var.set("")
                self._layout_controls(
                    show_chip=False,
                    show_navigation=False,
                    show_region_details=True,
                )
                self.image_var.set("Image: No flakes found")
                self._clear_current_regions()
                self.tile_viewer.clear("No flakes found in this scan")
                return

            self.view_folder = flakes_root / chip_folder.name
            self.chip_root = flakes_root
            self.display_chip_dropdown(True)
            self.populate_chips_dropdown(flakes_root, require_images=True)

        self._layout_controls(
            show_chip=self.chip_root is not None,
            show_navigation=not self.sparse_map_mode,
            show_raw_result_toggle=(
                selected_view == "Flakes Found"
                or (
                    self.capture_images_mode
                    and self._capture_has_both_sources()
                )
            ),
            show_region_details=selected_view == "Flakes Found",
        )
        self.load_current_folder(view_state=view_state)

    def load_current_folder(self, view_state=None):
        if self.view_folder is None or not self.view_folder.exists():
            self._show_missing_folder(f"Folder does not exist:\n{self.view_folder}")
            return

        if self.sparse_map_mode:
            title = self.selected_view
            if self.chip_var.get():
                title = f"{title} - {self.chip_var.get()}"
            self.image_var.set(f"Map: {title}")
            use_nearest = self.selected_view.startswith(("Processed", "Filtered"))
            fallback_name = self._flattened_map_name()
            overview_path = (
                self.view_scan_path / "Maps" / fallback_name
                if fallback_name is not None
                else None
            )

            def load_flattened_fallback():
                if fallback_name is not None:
                    self._load_flattened_map(
                        fallback_name,
                        title,
                        use_nearest,
                        view_state=view_state,
                    )
                else:
                    self.tile_viewer.clear("No positioned scan tiles were found")

            if self.tile_viewer.load_tiles(
                self.view_folder,
                title=title,
                nearest=use_nearest,
                on_empty=load_flattened_fallback,
                overview_path=overview_path,
                view_state=view_state,
            ):
                return
            if fallback_name is not None:
                self._load_flattened_map(
                    fallback_name,
                    title,
                    use_nearest,
                    view_state=view_state,
                )
            return

        if self.capture_images_mode:
            self.image_files = self._capture_image_files(
                self.capture_chip_folder
            )
        else:
            self.image_files = sorted(
                [
                    p for p in self.view_folder.iterdir()
                    if p.suffix.lower() in IMAGE_SUFFIXES
                ],
                key=self.image_sort_key
            )

        if not self.image_files:
            self.image_var.set("Image: None")
            self._clear_current_regions()
            messagebox.showwarning(
                "No Images Found",
                f"No images found in:\n{self.view_folder}"
            )
            return

        self.view_image_index = 0
        self.display_current_image()

    def display_current_image(self, preserve_region=False):
        if not self.image_files:
            return

        result_path = self.image_files[self.view_image_index]
        if self.selected_view == "Flakes Found":
            self.current_result_path = None
            self._set_current_regions(
                result_path,
                preserve_region=preserve_region,
            )
            self.current_result_path = result_path
        else:
            self._clear_current_regions()
        img_path = result_path
        showing_raw = (
            (
                self.selected_view == "Flakes Found"
                or self.capture_images_mode
            )
            and self.raw_result_var.get()
        )
        if self.capture_images_mode:
            source_name = "Raw" if showing_raw else "Processed"
            paired_path = self._capture_result_path(
                result_path,
                source_name,
            )
            if paired_path is not None:
                img_path = paired_path
            actual_source = (
                img_path.parent.name
                if img_path.parent.name.casefold() in {"raw", "processed"}
                else source_name
            )
            title = f"{actual_source} - {result_path.name}"
        elif showing_raw:
            raw_path = self._raw_result_path(result_path)
            if raw_path is not None:
                img_path = raw_path
            title = (
                f"Raw - {result_path.name}"
                if img_path != result_path
                else result_path.name
            )
        else:
            title = result_path.name
        self.image_var.set(f"Image: {title}")

        if not self.tile_viewer.load_image(img_path, title=title):
            messagebox.showwarning(
                "Image Error",
                f"Could not read image:\n{img_path}"
            )
            return
        if self.selected_view == "Flakes Found" and self.current_regions:
            self._update_region_highlight(fit_image=True)

    def _raw_result_path(self, result_path):
        if self.view_scan_path is None:
            return None
        chip_name = result_path.parent.name
        for magnification in ("10x", "20x"):
            root = self._magnification_root(
                magnification,
                self.view_scan_path / "All Images",
            )
            chip_folder = self._named_child(root, chip_name)
            raw_folder = self._named_child(chip_folder, "Raw")
            candidate = raw_folder / result_path.name if raw_folder else None
            if candidate is not None and candidate.is_file():
                return candidate
        return None

    def _processed_result_path(self, result_path):
        if self.view_scan_path is None:
            return None
        chip_name = result_path.parent.name
        for magnification in ("10x", "20x"):
            root = self._magnification_root(
                magnification,
                self.view_scan_path / "All Images",
            )
            chip_folder = self._named_child(root, chip_name)
            processed_folder = self._named_child(chip_folder, "Processed")
            candidate = (
                processed_folder / result_path.name
                if processed_folder is not None
                else None
            )
            if candidate is not None and candidate.is_file():
                return candidate
        return None

    def _capture_result_path(self, result_path, source_name):
        source_folder = self._named_child(
            self.capture_chip_folder,
            source_name,
        )
        candidate = (
            source_folder / result_path.name
            if source_folder is not None
            else None
        )
        if candidate is not None and candidate.is_file():
            return candidate
        return result_path if result_path.is_file() else None

    def _capture_image_files(self, chip_folder):
        files_by_name = {}
        for source_name in ("Raw", "Processed"):
            source_folder = self._named_child(chip_folder, source_name)
            if source_folder is None:
                continue
            try:
                source_files = [
                    path
                    for path in source_folder.iterdir()
                    if (
                        path.is_file()
                        and path.suffix.lower() in IMAGE_SUFFIXES
                    )
                ]
            except OSError:
                continue
            for path in source_files:
                files_by_name[path.name.casefold()] = path
        return sorted(files_by_name.values(), key=self.image_sort_key)

    def _capture_has_both_sources(self):
        return bool(
            self.capture_chip_folder
            and self._folder_has_images(
                self._named_child(self.capture_chip_folder, "Raw")
            )
            and self._folder_has_images(
                self._named_child(self.capture_chip_folder, "Processed")
            )
        )

    def _preferred_capture_folder(self, chip_folder):
        processed = self._named_child(chip_folder, "Processed")
        if self._folder_has_images(processed):
            return processed
        return self._named_child(chip_folder, "Raw")

    def _configure_capture_source_toggle(self):
        processed = self._folder_has_images(
            self._named_child(self.capture_chip_folder, "Processed")
        )
        self.raw_result_var.set(not processed)

    def previous_image(self, event=None):
        self.end_inspection()
        if self.app.get_view() != "Scan Results":
            return

        if self.sparse_map_mode:
            return

        if not self.image_files:
            return

        self.view_image_index = (self.view_image_index - 1) % len(self.image_files)
        self.display_current_image()
        self.root.focus_set()

    def next_image(self, event=None):
        self.end_inspection()
        if self.app.get_view() != "Scan Results":
            return

        if self.sparse_map_mode:
            return

        if not self.image_files:
            return

        self.view_image_index = (self.view_image_index + 1) % len(self.image_files)
        self.display_current_image()
        self.root.focus_set()

    def populate_chips_dropdown(
        self,
        chip_root,
        source_name=None,
        require_images=False,
        selected_name=None,
        capture_images=False,
    ):
        if not chip_root.exists():
            self.chip_dropdown["values"] = []
            self.chip_var.set("No chips found")
            return

        folders = (
            self._capture_chip_folders(chip_root)
            if capture_images
            else self._chip_folders(
                chip_root,
                source_name=source_name,
                require_images=require_images,
            )
        )
        chips = [path.name for path in folders]
        self.chip_dropdown["values"] = chips

        if chips:
            selected = (
                selected_name
                if selected_name in chips
                else chips[min(self.view_chip_index, len(chips) - 1)]
            )
            self.view_chip_index = chips.index(selected)
            self.chip_var.set(selected)
        else:
            self.chip_var.set("No chips found")

    def _on_chip_selected(self, _event=None):
        if self.chip_root is None:
            return
        selected_chip = self.chip_var.get()
        if not selected_chip:
            return
        chip_path = self.chip_root / selected_chip
        self.view_chip_index = list(self.chip_dropdown["values"]).index(selected_chip)
        self.view_image_index = 0
        if self.capture_images_mode:
            self.capture_chip_folder = chip_path
            self.view_folder = self._preferred_capture_folder(chip_path)
            self._configure_capture_source_toggle()
            self._layout_controls(
                show_raw_result_toggle=self._capture_has_both_sources(),
            )
        elif self.chip_source_folder is not None:
            self.view_folder = self._named_child(
                chip_path,
                self.chip_source_folder,
            )
        else:
            self.view_folder = chip_path
        self.load_current_folder()

    def get_capture_subfolder(
        self,
        path,
        index,
        preferred_name=None,
    ):
        folders = self._capture_chip_folders(path)
        if preferred_name:
            for folder in folders:
                if folder.name == preferred_name:
                    return folder
        return folders[index] if 0 <= index < len(folders) else None

    def _flattened_map_name(self):
        map_index = self.view_chip_index + 1
        wafer_match = re.search(
            r"\bwafer\s+(\d+)\b",
            self.chip_var.get(),
            re.IGNORECASE,
        )
        if wafer_match:
            map_index = int(wafer_match.group(1))
        names = {
            "Raw 2x": "map_2x.png",
            "Filtered 2x": "map_2x_filtered.png",
            "Raw 10x": f"map_10x_wafer_{map_index}.png",
            "Processed 10x": f"map_10x_processed_wafer_{map_index}.png",
            "Raw 20x": f"map_20x_wafer_{map_index}.png",
            "Processed 20x": f"map_20x_processed_wafer_{map_index}.png",
        }
        return names.get(self.selected_view)

    def _load_flattened_map(
        self,
        filename,
        title,
        nearest=False,
        view_state=None,
    ):
        map_path = self.view_scan_path / "Maps" / filename
        self.image_var.set(f"Map: {title}")
        if not map_path.is_file():
            self.tile_viewer.clear(f"Map not found: {filename}")
            messagebox.showwarning("Missing Map", f"Map does not exist:\n{map_path}")
            return False
        return self.tile_viewer.load_image(
            map_path,
            title=title,
            nearest=nearest,
            view_state=view_state,
        )

    def image_sort_key(self, p):
        match = re.search(r"_(\d+)\.", p.name)
        return int(match.group(1)) if match else -1

    def get_subfolder(
        self,
        path,
        index,
        source_name=None,
        require_images=False,
        preferred_name=None,
    ):
        if not path.exists():
            return None

        subfolders = self._chip_folders(
            path,
            source_name=source_name,
            require_images=require_images,
        )
        if preferred_name:
            for subfolder in subfolders:
                if subfolder.name == preferred_name:
                    return subfolder
        return subfolders[index] if 0 <= index < len(subfolders) else None

    def _show_missing_folder(self, message):
        self.image_files = []
        self.image_var.set("Image: None")
        self.tile_viewer.clear(message)
        messagebox.showwarning("Missing Folder", message)
