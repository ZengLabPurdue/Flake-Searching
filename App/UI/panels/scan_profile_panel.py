from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import tkinter as tk
from tkinter import Frame, Label, filedialog, messagebox
from tkinter import ttk

from config import PIXEL_SIZE
from Imaging import image_metadata
from Scanning import contour_finder
from Scanning.contour_extractor import get_region_from_point
from Scanning.contour_finder import region_contrast_rgb
from Scanning.scan_profile import (
    CLASSIFICATION_RANGE,
    CLASSIFICATION_SAMPLE,
    ScanProfile,
    ScanProfileError,
    build_region_overlay,
)


class ScanProfilePanel:
    IMAGE_TYPES = [("Images", "*.png *.jpg *.jpeg *.bmp")]
    CLASS_IMAGE_TYPES = [("Vignette-corrected PNG", "*.png")]
    DEFAULT_FLOOD_FILL_TOLERANCE = 15
    PREVIEW_LEGEND_POSITION = contour_finder.LEGEND_TOP_RIGHT
    SIMILAR_CLASS_CONTRAST_DISTANCE = 10.0
    CLASSIFICATION_LABELS = {
        CLASSIFICATION_SAMPLE: "Sampled color",
        CLASSIFICATION_RANGE: "Contrast range",
    }
    CLASSIFICATION_VALUES = {
        label: value for value, label in CLASSIFICATION_LABELS.items()
    }

    def __init__(self, parent, root, app, scan_profile: ScanProfile):
        self.parent = parent
        self.root = root
        self.app = app
        self.scan_profile = scan_profile

        self.mode = "create"
        self.class_edit_active = False
        self.editor_kind: str | None = None
        self.selected_item: tuple[str, int] | None = None
        self.editing_class_index: int | None = None
        self.pending_new_class = False
        self._updating_item_list = False
        self._display_items: list[tuple[str, int]] = []
        self.drag_class_index = None

        self.current_source_path: Path | None = None
        self.current_image_bgr: np.ndarray | None = None
        self.current_region_mask: np.ndarray | None = None
        self.current_seed_point: tuple[int, int] | None = None
        self.current_threshold: int | None = None
        self.current_contrast_rgb: tuple[int, int, int] | None = None

        self.preview_source_path: Path | None = None
        self.preview_image_bgr: np.ndarray | None = None
        self.preview_result_rgb: np.ndarray | None = None

        self.profile_name_var = tk.StringVar()
        self.profile_minimum_size_var = tk.StringVar()
        self.profile_maximum_size_var = tk.StringVar()
        self.class_name_var = tk.StringVar()
        self.class_group_var = tk.StringVar()
        self.identify_var = tk.BooleanVar(value=True)
        self.classification_type_var = tk.StringVar(
            value=self.CLASSIFICATION_LABELS[CLASSIFICATION_SAMPLE]
        )
        self.range_minimum_vars = [tk.StringVar() for _ in range(3)]
        self.range_maximum_vars = [tk.StringVar() for _ in range(3)]
        self.minimum_size_var = tk.StringVar()
        self.maximum_size_var = tk.StringVar()
        self.threshold_var = tk.StringVar(value=str(self.DEFAULT_FLOOD_FILL_TOLERANCE))
        self.editor_title_var = tk.StringVar(value="Class")
        self.tolerance_label_var = tk.StringVar(value="Color tolerance")
        self.contrast_var = tk.StringVar(value="RGB contrast: —")
        self.item_count_var = tk.StringVar(value="Items: 0")
        self.preview_enabled_var = tk.BooleanVar(value=False)
        self.preview_time_var = tk.StringVar(value="Preview time: —")
        self.status_var = tk.StringVar(
            value="Create or load a profile from the Scan menu."
        )

        self.frame = self._build_panel()
        self.frame.place_forget()
        self.status_var.trace_add("write", self._schedule_footer_resize)
        self.class_name_var.trace_add("write", self._on_item_name_changed)
        self.app.img_label.bind("<Button-1>", self._on_image_click, add="+")

    @staticmethod
    def _heading(parent, text=None, textvariable=None):
        return Label(
            parent,
            text=text,
            textvariable=textvariable,
            bg="white",
            fg="black",
            font=("TkDefaultFont", 10, "bold"),
        )

    def _build_panel(self):
        panel = Frame(self.parent, bg="#f0f0f0", width=254)
        background = Frame(panel, bg="white", padx=8)
        background.pack(fill="both", expand=True, padx=(2, 2))

        style = ttk.Style()
        style.configure(
            "Normal.TButton",
            font="TkDefaultFont",
            background="white",
            relief="flat",
        )
        style.configure("Profile.TCheckbutton", background="white")

        Label(
            background,
            text="Scan Search Profile",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 13),
        ).pack(pady=(8, 7))

        self._heading(background, "Profile").pack(anchor="w")
        profile_name_row = Frame(background, bg="white")
        profile_name_row.pack(fill="x", pady=(4, 0))
        Label(profile_name_row, text="Name", bg="white", fg="black").pack(side="left")
        self.profile_name_entry = ttk.Entry(
            profile_name_row,
            textvariable=self.profile_name_var,
        )
        self.profile_name_entry.pack(side="left", fill="x", expand=True, padx=(6, 0))

        profile_size_row = Frame(background, bg="white")
        profile_size_row.pack(fill="x", pady=(5, 0))
        Label(profile_size_row, text="Size (um)", bg="white", fg="black").pack(side="left")
        Label(profile_size_row, text="Min", bg="white", fg="black").pack(
            side="left", padx=(8, 3)
        )
        self.profile_minimum_size_entry = ttk.Entry(
            profile_size_row,
            textvariable=self.profile_minimum_size_var,
            width=6,
        )
        self.profile_minimum_size_entry.pack(side="left", fill="x", expand=True)
        Label(profile_size_row, text="Max", bg="white", fg="black").pack(
            side="left", padx=(8, 3)
        )
        self.profile_maximum_size_entry = ttk.Entry(
            profile_size_row,
            textvariable=self.profile_maximum_size_var,
            width=6,
        )
        self.profile_maximum_size_entry.pack(side="left", fill="x", expand=True)

        self.item_count_label = Label(
            background,
            textvariable=self.item_count_var,
            bg="white",
            fg="black",
        )
        self.item_count_label.pack(anchor="w", pady=(9, 2))

        list_frame = Frame(background, bg="white", height=82)
        list_frame.pack(fill="x")
        list_frame.pack_propagate(False)
        self.class_list = tk.Listbox(
            list_frame,
            height=4,
            width=30,
            exportselection=False,
        )
        item_scrollbar = ttk.Scrollbar(
            list_frame,
            orient="vertical",
            command=self.class_list.yview,
        )
        self.class_list.configure(yscrollcommand=item_scrollbar.set)
        self.class_list.pack(side="left", fill="both", expand=True)
        item_scrollbar.pack(side="right", fill="y")
        self.class_list.bind("<<ListboxSelect>>", self._on_item_selected)
        self.class_list.bind("<ButtonPress-1>", self._start_class_drag)
        self.class_list.bind("<B1-Motion>", self._drag_class)
        self.class_list.bind("<ButtonRelease-1>", self._end_class_drag)

        item_buttons = Frame(background, bg="white")
        item_buttons.pack(fill="x", pady=(5, 0))
        item_buttons.columnconfigure(0, weight=1)
        item_buttons.columnconfigure(1, weight=1)
        self.new_class_button = ttk.Button(
            item_buttons,
            text="New class",
            style="Normal.TButton",
            command=self.new_class,
        )
        self.new_class_button.grid(row=0, column=0, sticky="ew", padx=(0, 2))
        self.new_class_button.grid_configure(columnspan=2, padx=0)
        self.edit_item_button = ttk.Button(
            item_buttons,
            text="Edit selected",
            style="Normal.TButton",
            command=self.edit_selected_item,
        )
        self.edit_item_button.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=(0, 2),
            pady=(3, 0),
        )
        self.copy_item_button = ttk.Button(
            item_buttons,
            text="Copy selected",
            style="Normal.TButton",
            command=self.copy_selected_item,
        )
        self.copy_item_button.grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(2, 0),
            pady=(3, 0),
        )
        self.extend_profile_button = ttk.Button(
            item_buttons,
            text="Extend profile",
            style="Normal.TButton",
            command=self.extend_profile,
        )
        self.extend_profile_button.grid(
            row=2,
            column=0,
            sticky="ew",
            padx=(0, 2),
            pady=(3, 0),
        )
        self.remove_item_button = ttk.Button(
            item_buttons,
            text="Remove selected",
            style="Normal.TButton",
            command=self.remove_selected_item,
        )
        self.remove_item_button.grid(
            row=2,
            column=1,
            sticky="ew",
            padx=(2, 0),
            pady=(3, 0),
        )

        self.editor_heading = self._heading(background, textvariable=self.editor_title_var)
        self.editor_heading.pack(anchor="w", pady=(9, 0))
        self.editor = Frame(background, bg="white")
        self.editor.pack(fill="x", pady=(4, 0))
        self.editor.columnconfigure(1, weight=1)

        Label(self.editor, text="Label", bg="white", fg="black").grid(
            row=0,
            column=0,
            sticky="w",
        )
        self.class_name_entry = ttk.Entry(
            self.editor,
            textvariable=self.class_name_var,
        )
        self.class_name_entry.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        self.dynamic_editor = Frame(self.editor, bg="white")
        self.dynamic_editor.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.dynamic_editor.columnconfigure(0, weight=1)

        self.class_options = Frame(self.dynamic_editor, bg="white")
        group_row = Frame(self.class_options, bg="white")
        group_row.pack(fill="x", pady=(5, 0))
        Label(group_row, text="Group", bg="white", fg="black").pack(side="left")
        self.class_group_combo = ttk.Combobox(
            group_row,
            textvariable=self.class_group_var,
        )
        self.class_group_combo.pack(side="left", fill="x", expand=True, padx=(6, 0))

        classification_row = Frame(self.class_options, bg="white")
        classification_row.pack(fill="x", pady=(5, 0))
        Label(
            classification_row,
            text="Classification",
            bg="white",
            fg="black",
        ).pack(side="left")
        self.classification_type_combo = ttk.Combobox(
            classification_row,
            textvariable=self.classification_type_var,
            values=tuple(self.CLASSIFICATION_VALUES),
            state="readonly",
            width=15,
        )
        self.classification_type_combo.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(6, 0),
        )
        self.classification_type_combo.bind(
            "<<ComboboxSelected>>",
            self._on_classification_type_changed,
        )
        self.identify_check = ttk.Checkbutton(
            self.class_options,
            text="Identify",
            variable=self.identify_var,
            style="Profile.TCheckbutton",
        )
        self.identify_check.pack(anchor="w", pady=(4, 0))

        self.region_options = Frame(self.dynamic_editor, bg="white")
        region_row = Frame(self.region_options, bg="white")
        region_row.pack(fill="x", pady=(5, 0))
        self.load_image_button = ttk.Button(
            region_row,
            text="Load image",
            style="Normal.TButton",
            command=self.load_item_image,
        )
        self.load_image_button.pack(side="left")
        Label(
            region_row,
            textvariable=self.tolerance_label_var,
            bg="white",
            fg="black",
        ).pack(side="left", padx=(12, 4))
        self.threshold_spinbox = ttk.Spinbox(
            region_row,
            from_=0,
            to=255,
            textvariable=self.threshold_var,
            width=5,
        )
        self.threshold_spinbox.pack(side="left")

        self.range_options = Frame(self.dynamic_editor, bg="white")
        Label(
            self.range_options,
            text="Signed RGB contrast range",
            bg="white",
            fg="black",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(5, 1))
        Label(self.range_options, text="Min", bg="white", fg="black").grid(
            row=1,
            column=1,
            padx=(7, 2),
        )
        Label(self.range_options, text="Max", bg="white", fg="black").grid(
            row=1,
            column=2,
            padx=(3, 0),
        )
        self.range_entries = []
        for row, channel in enumerate(("Red", "Green", "Blue"), start=2):
            Label(
                self.range_options,
                text=channel,
                bg="white",
                fg="black",
            ).grid(row=row, column=0, sticky="w")
            minimum_entry = ttk.Spinbox(
                self.range_options,
                from_=-255,
                to=255,
                textvariable=self.range_minimum_vars[row - 2],
                width=7,
            )
            maximum_entry = ttk.Spinbox(
                self.range_options,
                from_=-255,
                to=255,
                textvariable=self.range_maximum_vars[row - 2],
                width=7,
            )
            minimum_entry.grid(row=row, column=1, sticky="ew", padx=(7, 2), pady=1)
            maximum_entry.grid(row=row, column=2, sticky="ew", padx=(3, 0), pady=1)
            self.range_entries.extend((minimum_entry, maximum_entry))
        self.range_options.columnconfigure(1, weight=1)
        self.range_options.columnconfigure(2, weight=1)

        self.class_size_options = Frame(self.dynamic_editor, bg="white")
        Label(
            self.class_size_options,
            text="Class size (um)",
            bg="white",
            fg="black",
        ).pack(side="left")
        Label(self.class_size_options, text="Min", bg="white", fg="black").pack(
            side="left", padx=(7, 3)
        )
        self.minimum_size_entry = ttk.Entry(
            self.class_size_options,
            textvariable=self.minimum_size_var,
            width=6,
        )
        self.minimum_size_entry.pack(side="left", fill="x", expand=True)
        Label(self.class_size_options, text="Max", bg="white", fg="black").pack(
            side="left", padx=(7, 3)
        )
        self.maximum_size_entry = ttk.Entry(
            self.class_size_options,
            textvariable=self.maximum_size_var,
            width=6,
        )
        self.maximum_size_entry.pack(side="left", fill="x", expand=True)

        self.contrast_label = Label(
            self.dynamic_editor,
            textvariable=self.contrast_var,
            bg="white",
            fg="black",
        )
        self.confirm_item_button = ttk.Button(
            self.dynamic_editor,
            text="Confirm class",
            style="Normal.TButton",
            command=self.confirm_item,
        )

        self.preview_heading = self._heading(background, "Profile Preview")
        self.preview_heading.pack(anchor="w", pady=(9, 0))
        preview_row = Frame(background, bg="white")
        preview_row.pack(fill="x", pady=(4, 0))
        preview_row.columnconfigure(0, weight=1)
        preview_row.columnconfigure(1, weight=1)
        self.load_preview_button = ttk.Button(
            preview_row,
            text="Load preview image",
            style="Normal.TButton",
            command=self.load_preview_image,
        )
        self.load_preview_button.grid(row=0, column=0, sticky="ew", padx=(0, 2))
        self.run_preview_button = ttk.Button(
            preview_row,
            text="Run profile",
            style="Normal.TButton",
            command=self.run_profile_preview,
        )
        self.run_preview_button.grid(row=0, column=1, sticky="ew", padx=(2, 0))
        self.preview_toggle = ttk.Checkbutton(
            background,
            text="Show profile result",
            variable=self.preview_enabled_var,
            command=self.toggle_profile_preview,
            style="Profile.TCheckbutton",
        )
        self.preview_toggle.pack(anchor="w", pady=(4, 0))
        self.preview_toggle.state(["disabled"])
        Label(
            background,
            textvariable=self.preview_time_var,
            bg="white",
            fg="#333333",
        ).pack(anchor="w", pady=(2, 0))

        self.status_group = Frame(background, bg="white")
        self.status_group.pack(fill="x", pady=(7, 0))
        self.status_label = Label(
            self.status_group,
            textvariable=self.status_var,
            bg="white",
            fg="#333333",
            justify="center",
            wraplength=234,
        )
        self.status_label.pack(fill="x")
        self.save_profile_button = ttk.Button(
            self.status_group,
            text="Save profile",
            style="Normal.TButton",
            command=self.save_profile,
        )
        self.save_profile_button.pack(pady=(10, 0))
        self.bottom_spacing = Frame(background, bg="white", height=15)
        self.bottom_spacing.pack(fill="x")
        self.bottom_spacing.pack_propagate(False)

        self.panel_frame = panel
        self.panel_background = background
        self.profile_edit_widgets = (
            self.profile_name_entry,
            self.profile_minimum_size_entry,
            self.profile_maximum_size_entry,
            self.new_class_button,
            self.edit_item_button,
            self.copy_item_button,
            self.extend_profile_button,
            self.remove_item_button,
            self.load_preview_button,
            self.run_preview_button,
            self.save_profile_button,
        )
        self.item_edit_widgets = (
            self.class_name_entry,
            self.class_group_combo,
            self.classification_type_combo,
            self.identify_check,
            self.load_image_button,
            self.threshold_spinbox,
            *self.range_entries,
            self.minimum_size_entry,
            self.maximum_size_entry,
            self.confirm_item_button,
        )
        self._show_editor_options(None)
        self._resize_panel_to_footer()
        return panel

    def start_create(self):
        self.mode = "create"
        self.scan_profile.clear()
        self.profile_name_var.set("")
        self.profile_minimum_size_var.set("")
        self.profile_maximum_size_var.set("")
        self._reset_editor()
        self._reset_preview()
        self._set_editing_enabled(True)
        self._refresh_item_list()
        self.status_var.set("Name the profile, then create a class.")
        self.app.set_view("Create Search Profile")
        self.show()
        self.app.display_image_message("Create a class or load a preview image.")
        self.profile_name_entry.focus_set()

    def choose_and_load_profile(self):
        self._reset_editor()
        self._reset_preview()
        self._set_item_editing_enabled(False)
        self.app.set_view("Load Search Profile")
        self.show()
        self.app.display_image_message("Open a profile, then select a class.")
        selected = self._choose_profile_path("Load Scan Search Profile")
        if not selected:
            return
        try:
            profile = self.scan_profile.load_profile(selected)
        except (OSError, ScanProfileError) as exc:
            messagebox.showerror("Invalid Scan Search Profile", str(exc))
            return
        self.show_loaded_profile(profile)
        self.app.set_active_scan_profile(profile)
        messagebox.showinfo(
            "Profile Loaded",
            f"Loaded '{profile.name}' with {len(profile.classes)} class(es).",
        )

    def show_loaded_profile(self, profile: ScanProfile):
        self.mode = "edit"
        self.scan_profile = profile
        self.profile_name_var.set(profile.name)
        self.profile_minimum_size_var.set(
            "" if profile.minimum_size_um is None else f"{profile.minimum_size_um:g}"
        )
        self.profile_maximum_size_var.set(
            "" if profile.maximum_size_um is None else f"{profile.maximum_size_um:g}"
        )
        self._reset_editor()
        self._reset_preview()
        self._set_editing_enabled(True)
        self._refresh_item_list()
        self.status_var.set(
            f"Editing {profile.name}. Select an item or run a preview."
        )
        self.app.set_view("Load Search Profile")
        self.show()
        self.app.display_image_message("Select a class or load a preview image.")

    def new_class(self):
        if self.class_edit_active:
            self._show_confirm_current_message()
            return
        self._clear_list_selection()
        self.editor_kind = "class"
        self.pending_new_class = True
        self.editing_class_index = None
        self.selected_item = None
        self.class_name_var.set(self._next_default_class_name())
        self.class_group_var.set("")
        self.identify_var.set(True)
        self.classification_type_var.set(
            self.CLASSIFICATION_LABELS[CLASSIFICATION_SAMPLE]
        )
        self._set_range_values(None, None)
        self.minimum_size_var.set("")
        self.maximum_size_var.set("")
        self.threshold_var.set(str(self.DEFAULT_FLOOD_FILL_TOLERANCE))
        source_available = self.current_image_bgr is not None
        self._clear_region_selection()
        self._show_editor_options("class")
        self._set_item_editing_enabled(True)
        self._refresh_item_list(select_pending=True)
        if source_available:
            self._display_bgr(self.current_image_bgr)
            self.status_var.set("Click a region in the current image or load another image.")
        else:
            self.status_var.set("Load an image for the new class.")
            self.app.display_image_message("Open an image to create this class.")
        self.class_name_entry.focus_set()

    def edit_selected_item(self):
        selected = self._selected_list_item()
        if selected is None:
            messagebox.showinfo("Edit Class", "Select a confirmed class first.")
            return
        _kind, index = selected
        self.editor_kind = "class"
        self.editing_class_index = index
        self.pending_new_class = False
        self._show_editor_options("class")
        self._set_item_editing_enabled(True)
        self.status_var.set(f"Editing {self.class_name_var.get()}.")
        self.class_name_entry.focus_set()

    def copy_selected_item(self):
        if self.class_edit_active:
            self._show_confirm_current_message()
            return
        selected = self._selected_list_item()
        if selected is None:
            messagebox.showinfo("Copy Class", "Select a confirmed class first.")
            return
        _kind, index = selected
        copied = self.scan_profile.copy_class(index)
        new_index = len(self.scan_profile.classes) - 1
        self._refresh_item_list(select_item=("class", new_index))
        self._on_item_selected()
        self.editor_kind = "class"
        self.editing_class_index = new_index
        self._show_editor_options("class")
        self._set_item_editing_enabled(True)
        self.status_var.set(f"Copied as {copied['name']}. Modify it, then confirm.")

    def extend_profile(self):
        if self.class_edit_active:
            self._show_confirm_current_message()
            return
        selected = self._choose_profile_path("Extend From Scan Search Profile")
        if not selected:
            return
        try:
            source_profile = ScanProfile(self.scan_profile.profiles_dir).load_profile(selected)
            added_classes = self.scan_profile.extend_from_profile(source_profile)
        except (OSError, ScanProfileError) as exc:
            messagebox.showerror("Profile Extend Error", str(exc))
            return
        self._refresh_item_list()
        self.status_var.set(
            f"Added {len(added_classes)} class(es) from {source_profile.name}."
        )

    def remove_selected_item(self):
        if self.pending_new_class:
            self._reset_editor()
            self._refresh_item_list()
            self.status_var.set("Cancelled the unconfirmed item.")
            return
        selected = self._selected_list_item()
        if selected is None:
            return
        _kind, index = selected
        removed = self.scan_profile.remove_class(index)
        self._reset_editor()
        self._clear_pending_image()
        self._refresh_item_list()
        self.status_var.set(f"Removed {removed['name']}.")
        self.app.display_image_message("Select an item or load a preview image.")

    def load_item_image(self):
        if not self.class_edit_active or self.editor_kind != "class":
            self._show_create_or_edit_item_prompt()
            return
        if self._classification_type() == CLASSIFICATION_RANGE:
            messagebox.showinfo(
                "Contrast Range",
                "Contrast range classes use the six RGB minimum and maximum values "
                "instead of a reference image.",
            )
            return
        selected = filedialog.askopenfilename(
            title="Select an Image Containing the Region",
            filetypes=self.CLASS_IMAGE_TYPES,
        )
        if not selected:
            return
        source_path = Path(selected)
        if not image_metadata.is_vignette_corrected(source_path):
            messagebox.showwarning(
                "Vignette Correction Required",
                "Only PNG images saved with vignette correction can define classes.",
            )
            return
        image_bgr = self._read_bgr_image(source_path)
        if image_bgr is None:
            messagebox.showerror("Image Error", f"Could not read image:\n{source_path}")
            return
        self.current_source_path = source_path
        self.current_image_bgr = image_bgr
        self._clear_region_selection()
        self.app.set_view(
            "Load Search Profile" if self.mode == "edit" else "Create Search Profile"
        )
        self.root.update_idletasks()
        self._display_bgr(image_bgr)
        self.status_var.set("Click a point inside the reference region.")

    def confirm_item(self):
        if not self.class_edit_active:
            self._show_create_or_edit_item_prompt()
            return
        self._confirm_class()

    def _confirm_class(self):
        classification_type = self._classification_type()
        if (
            classification_type == CLASSIFICATION_SAMPLE
            and not self._region_is_ready()
        ):
            return
        class_name = self.class_name_var.get().strip() or self._next_default_class_name()
        self.class_name_var.set(class_name)
        try:
            minimum_size, maximum_size = self._get_class_size_requirement()
            if classification_type == CLASSIFICATION_RANGE:
                minimum_rgb, maximum_rgb = self._get_contrast_range()
                threshold = None
            else:
                threshold = self._get_threshold()
                minimum_rgb = maximum_rgb = None
        except ScanProfileError as exc:
            messagebox.showwarning("Invalid Class", str(exc))
            return
        if (
            classification_type == CLASSIFICATION_SAMPLE
            and threshold != self.current_threshold
        ):
            self._show_select_region_again()
            return
        existing_index = self.scan_profile.find_class(class_name)
        target_index = self.editing_class_index
        if target_index is not None:
            if existing_index is not None and existing_index != target_index:
                messagebox.showwarning(
                    "Duplicate Class Label",
                    f"A class labeled '{class_name}' already exists.",
                )
                return
        elif existing_index is not None:
            messagebox.showwarning(
                "Duplicate Class Label",
                f"A class labeled '{class_name}' already exists.",
            )
            return
        if classification_type == CLASSIFICATION_SAMPLE:
            similar_match = self._find_similar_class(
                self.current_contrast_rgb,
                exclude_index=target_index,
            )
            if similar_match is not None:
                similar_class, distance = similar_match
                if not messagebox.askyesno(
                    "Add Similar Class?",
                    f"'{class_name}' closely resembles '{similar_class['name']}'.\n\n"
                    f"Contrast distance: {distance:.1f}.\n\nContinue?",
                ):
                    return
        try:
            common_arguments = {
                "name": class_name,
                "minimum_size_um": minimum_size,
                "maximum_size_um": maximum_size,
                "group": self.class_group_var.get(),
                "identify": bool(self.identify_var.get()),
                "index": target_index,
            }
            if classification_type == CLASSIFICATION_RANGE:
                profile_class = self.scan_profile.set_range_class(
                    minimum_contrast_rgb=minimum_rgb,
                    maximum_contrast_rgb=maximum_rgb,
                    **common_arguments,
                )
            else:
                profile_class = self.scan_profile.set_class(
                    source_path=self.current_source_path,
                    image_bgr=self.current_image_bgr.copy(),
                    region_mask=self.current_region_mask.copy(),
                    seed_point=self.current_seed_point,
                    threshold=self.current_threshold,
                    **common_arguments,
                )
        except ScanProfileError as exc:
            messagebox.showwarning("Invalid Class", str(exc))
            return
        if classification_type == CLASSIFICATION_RANGE:
            status = (
                f"Confirmed {profile_class['name']}. RGB range: "
                f"{profile_class['minimum_contrast_rgb']} to "
                f"{profile_class['maximum_contrast_rgb']}."
            )
        else:
            status = (
                f"Confirmed {profile_class['name']}. "
                f"RGB contrast: {profile_class['contrast_rgb']}."
            )
        self._finish_item_confirmation(profile_class, status)

    def _finish_item_confirmation(self, item, status):
        if self.current_image_bgr is not None:
            self._display_bgr(self.current_image_bgr)
        self._reset_editor(clear_image=False)
        self._refresh_item_list()
        self.status_var.set(status)

    def load_preview_image(self):
        selected = filedialog.askopenfilename(
            title="Select a Profile Preview Image",
            filetypes=self.IMAGE_TYPES,
        )
        if not selected:
            return
        path = Path(selected)
        image_bgr = self._read_bgr_image(path)
        if image_bgr is None:
            messagebox.showerror("Preview Image Error", f"Could not read image:\n{path}")
            return
        self.preview_source_path = path
        self.preview_image_bgr = image_bgr
        self.preview_result_rgb = None
        self.preview_enabled_var.set(False)
        self.preview_time_var.set("Preview time: —")
        self.preview_toggle.state(["disabled"])
        self.app.set_view(
            "Load Search Profile" if self.mode == "edit" else "Create Search Profile"
        )
        self._display_bgr(image_bgr)
        self.status_var.set("Preview image loaded. Click Run profile to test it.")

    def run_profile_preview(self):
        if self.class_edit_active:
            self._show_confirm_current_message()
            return
        if self.preview_image_bgr is None:
            messagebox.showinfo("Preview Image", "Load a preview image first.")
            return
        if not self.scan_profile.classes:
            messagebox.showinfo("Empty Profile", "Add at least one class first.")
            return
        try:
            self.status_var.set("Running profile preview...")
            self.preview_time_var.set("Preview time: running...")
            self.root.update_idletasks()
            self.scan_profile.set_size_requirement(*self._get_profile_size_requirement())
            pixel_size = self._preview_pixel_size()
            started_at = perf_counter()
            result_rgb, _, details = contour_finder.find_flakes(
                self.preview_image_bgr,
                return_details=True,
                profile_configuration=self.scan_profile.matching_configuration(),
                pixel_size_um=pixel_size,
                color_seed=0,
                legend_position=self.PREVIEW_LEGEND_POSITION,
            )
            elapsed_seconds = perf_counter() - started_at
        except (OSError, ScanProfileError, ValueError, cv2.error) as exc:
            self.preview_time_var.set("Preview time: failed")
            messagebox.showerror("Profile Preview Error", str(exc))
            return
        self.preview_result_rgb = result_rgb.copy()
        self.preview_time_var.set(f"Preview time: {elapsed_seconds:.3f}s")
        self.preview_enabled_var.set(True)
        self.preview_toggle.state(["!disabled"])
        self.toggle_profile_preview()
        matched = sum(item["matched_class"] is not None for item in details["region_results"])
        self.status_var.set(
            f"Preview complete in {elapsed_seconds:.3f}s: "
            f"{matched} identified, "
            f"{len(details['region_results'])} total regions."
        )

    def toggle_profile_preview(self):
        if self.preview_image_bgr is None:
            return
        if self.preview_enabled_var.get() and self.preview_result_rgb is not None:
            self.app.display_image(self.preview_result_rgb)
        else:
            self._display_bgr(self.preview_image_bgr)

    def save_profile(self):
        if self.class_edit_active:
            self._show_confirm_current_message()
            return
        try:
            self.scan_profile.set_size_requirement(*self._get_profile_size_requirement())
            profile = self.scan_profile.save_profile(self.profile_name_var.get())
        except FileExistsError as exc:
            if not messagebox.askyesno(
                "Replace Profile",
                f"A profile already exists at:\n{exc.args[0]}\n\nReplace it?",
            ):
                return
            try:
                profile = self.scan_profile.save_profile(
                    self.profile_name_var.get(),
                    overwrite=True,
                )
            except (OSError, ScanProfileError) as save_exc:
                messagebox.showerror("Profile Save Error", str(save_exc))
                return
        except (OSError, ScanProfileError) as exc:
            messagebox.showerror("Profile Save Error", str(exc))
            return
        self.app.set_active_scan_profile(profile)
        self.show_loaded_profile(profile)
        messagebox.showinfo(
            "Profile Saved",
            f"Saved and activated '{profile.name}' at:\n{profile.path}",
        )

    def _on_image_click(self, event):
        if self.mode not in ("create", "edit"):
            return
        if self.app.get_view() not in ("Create Search Profile", "Load Search Profile"):
            return
        if not self.class_edit_active:
            self._show_create_or_edit_item_prompt()
            return
        if self._classification_type() == CLASSIFICATION_RANGE:
            self.status_var.set(
                "Contrast range classes are defined by the RGB min/max fields."
            )
            return
        if self.current_image_bgr is None:
            messagebox.showinfo("Load Image", "Load an image before selecting a region.")
            return
        seed_point = self.app.display_to_image_point(event.x, event.y)
        if seed_point is None:
            self.status_var.set("Click inside the displayed image, not the gray border.")
            return
        try:
            threshold = self._get_threshold()
            _, _, region_mask, contour = get_region_from_point(
                image_bgr=self.current_image_bgr,
                seed_point=seed_point,
                threshold=threshold,
                connectivity=8,
            )
        except (cv2.error, ScanProfileError, ValueError) as exc:
            messagebox.showerror("Region Selection Error", str(exc))
            return
        if contour is None or not np.any(region_mask > 0):
            self.status_var.set(
                "No region was found. Try another point or color tolerance."
            )
            return
        try:
            contrast = region_contrast_rgb(self.current_image_bgr, region_mask)
        except ValueError as exc:
            messagebox.showerror("Contrast Error", str(exc))
            return
        self.current_seed_point = seed_point
        self.current_region_mask = region_mask
        self.current_threshold = threshold
        self.current_contrast_rgb = contrast
        self._set_contrast_display(contrast)
        self._display_bgr(
            build_region_overlay(self.current_image_bgr, region_mask, seed_point)
        )
        self.status_var.set("Region selected. Confirm or click again.")

    def _on_item_selected(self, event=None):
        if self._updating_item_list:
            return
        selected = self._selected_list_item()
        if selected is None:
            return
        self.pending_new_class = False
        kind, index = selected
        self.selected_item = selected
        self.editor_kind = kind
        self.editing_class_index = None
        self._set_item_editing_enabled(False)
        self._load_class_into_editor(index)
        self._show_editor_options(kind)
        self.status_var.set(
            f"Selected {self.class_name_var.get()}. Click Edit selected to modify it."
        )

    def _load_class_into_editor(self, index):
        item = self.scan_profile.get_class(index)
        self.class_name_var.set(item["name"])
        self.class_group_var.set(item["group"])
        self.identify_var.set(item["identify"])
        classification_type = item.get(
            "classification_type",
            CLASSIFICATION_SAMPLE,
        )
        self.classification_type_var.set(
            self.CLASSIFICATION_LABELS[classification_type]
        )
        self.minimum_size_var.set(
            "" if item["minimum_size_um"] is None else f"{item['minimum_size_um']:g}"
        )
        self.maximum_size_var.set(
            "" if item["maximum_size_um"] is None else f"{item['maximum_size_um']:g}"
        )
        if classification_type == CLASSIFICATION_RANGE:
            self._set_range_values(
                item["minimum_contrast_rgb"],
                item["maximum_contrast_rgb"],
            )
            self._clear_pending_image()
            self._set_contrast_display(None)
            self.app.display_image_message(
                "This class matches its signed RGB contrast range."
            )
        else:
            self.threshold_var.set(str(item["threshold"]))
            self._set_range_values(None, None)
            self._load_region_reference(item)

    def _load_region_reference(self, item):
        self.current_source_path = item["source_path"]
        self.current_image_bgr = item["image_bgr"].copy()
        self.current_region_mask = item["region_mask"].copy()
        self.current_seed_point = item["seed_point"]
        self.current_threshold = item["threshold"]
        self.current_contrast_rgb = item["contrast_rgb"]
        self._set_contrast_display(item["contrast_rgb"])
        self._display_bgr(
            build_region_overlay(
                self.current_image_bgr,
                self.current_region_mask,
                self.current_seed_point,
            )
        )

    def _show_editor_options(self, kind):
        self.class_options.pack_forget()
        self.region_options.pack_forget()
        self.range_options.pack_forget()
        self.class_size_options.pack_forget()
        self.contrast_label.pack_forget()
        self.confirm_item_button.pack_forget()
        if kind is None:
            self.editor_heading.pack_forget()
            self.editor.pack_forget()
            self._schedule_footer_resize()
            return
        if not self.editor_heading.winfo_manager():
            self.editor_heading.pack(anchor="w", pady=(9, 0), before=self.preview_heading)
            self.editor.pack(fill="x", pady=(4, 0), before=self.preview_heading)
        self.editor_title_var.set("Class")
        self.class_options.pack(fill="x")
        if self._classification_type() == CLASSIFICATION_RANGE:
            self.range_options.pack(fill="x")
        else:
            self.region_options.pack(fill="x")
        self.class_size_options.pack(fill="x", pady=(5, 0))
        if self._classification_type() == CLASSIFICATION_SAMPLE:
            self.contrast_label.pack(anchor="w", pady=(5, 0))
        self.confirm_item_button.configure(text="Confirm class")
        self.confirm_item_button.pack(pady=(5, 0))
        self._schedule_footer_resize()

    def _set_editing_enabled(self, enabled):
        for widget in self.profile_edit_widgets:
            widget.state(["!disabled"] if enabled else ["disabled"])
        self._set_item_editing_enabled(False)

    def _set_item_editing_enabled(self, enabled):
        self.class_edit_active = bool(enabled)
        for widget in self.item_edit_widgets:
            widget.state(["!disabled"] if enabled else ["disabled"])

    def _refresh_item_list(
        self,
        select_item: tuple[str, int] | None = None,
        *,
        select_pending=False,
    ):
        self._updating_item_list = True
        try:
            self.class_list.delete(0, tk.END)
            self._display_items = []
            for index, profile_class in enumerate(self.scan_profile.classes):
                self.class_list.insert(tk.END, self._class_list_text(profile_class))
                self._display_items.append(("class", index))
            if self.pending_new_class:
                tags = self._class_tags(
                    self._classification_type(),
                    bool(self.identify_var.get()),
                )
                self.class_list.insert(
                    tk.END,
                    f"{tags} {self.class_name_var.get()}",
                )
            class_count = len(self.scan_profile.classes)
            pending_count = int(self.pending_new_class)
            self.item_count_var.set(f"Classes: {class_count + pending_count}")
            self.class_group_combo.configure(values=self.scan_profile.groups)
            if select_item is not None:
                try:
                    list_index = self._display_items.index(select_item)
                except ValueError:
                    pass
                else:
                    self.class_list.selection_set(list_index)
                    self.class_list.activate(list_index)
            elif select_pending and self.pending_new_class:
                list_index = len(self._display_items)
                self.class_list.selection_set(list_index)
                self.class_list.activate(list_index)
        finally:
            self._updating_item_list = False

    @classmethod
    def _class_list_text(cls, profile_class):
        tags = cls._class_tags(
            profile_class.get("classification_type", CLASSIFICATION_SAMPLE),
            profile_class["identify"],
        )
        group = f"{profile_class['group']} / " if profile_class["group"] else ""
        return f"{tags} {group}{profile_class['name']}"

    @staticmethod
    def _class_tags(classification_type, identify):
        type_tag = "[R]" if classification_type == CLASSIFICATION_RANGE else "[S]"
        return f"{type_tag}[I]" if identify else type_tag

    def _selected_list_item(self):
        selection = self.class_list.curselection()
        if not selection:
            return None
        list_index = int(selection[0])
        if not 0 <= list_index < len(self._display_items):
            return None
        return self._display_items[list_index]

    def _on_item_name_changed(self, *_):
        if self._updating_item_list or not self.class_edit_active:
            return
        if self.pending_new_class:
            self._refresh_item_list(select_pending=True)

    def _start_class_drag(self, event):
        self.drag_class_index = None
        if self.class_edit_active or self.pending_new_class:
            return
        list_index = self.class_list.nearest(event.y)
        if not 0 <= list_index < len(self._display_items):
            return
        kind, class_index = self._display_items[list_index]
        item_bounds = self.class_list.bbox(list_index)
        if (
            kind != "class"
            or item_bounds is None
            or not item_bounds[1] <= event.y <= item_bounds[1] + item_bounds[3]
        ):
            return
        self.drag_class_index = class_index

    def _drag_class(self, event):
        if self.drag_class_index is None or not self.scan_profile.classes:
            return
        list_index = self.class_list.nearest(event.y)
        target_index = max(0, min(list_index, len(self.scan_profile.classes) - 1))
        if target_index == self.drag_class_index:
            return "break"
        profile_class = self.scan_profile.move_class(self.drag_class_index, target_index)
        self.drag_class_index = target_index
        self._refresh_item_list(select_item=("class", target_index))
        self.status_var.set(f"Moved {profile_class['name']} to position {target_index + 1}.")
        return "break"

    def _end_class_drag(self, event):
        self.drag_class_index = None

    def _get_threshold(self):
        try:
            threshold = int(self.threshold_var.get())
        except ValueError as exc:
            raise ScanProfileError(
                "Color tolerance must be a whole number from 0 to 255."
            ) from exc
        if not 0 <= threshold <= 255:
            raise ScanProfileError("Color tolerance must be between 0 and 255.")
        return threshold

    def _classification_type(self):
        return self.CLASSIFICATION_VALUES.get(
            self.classification_type_var.get(),
            CLASSIFICATION_SAMPLE,
        )

    def _on_classification_type_changed(self, event=None):
        if not self.class_edit_active:
            return
        classification_type = self._classification_type()
        if (
            classification_type == CLASSIFICATION_RANGE
            and self.current_contrast_rgb is not None
            and all(not value.get().strip() for value in self.range_minimum_vars)
            and all(not value.get().strip() for value in self.range_maximum_vars)
        ):
            try:
                tolerance = self._get_threshold()
            except ScanProfileError:
                tolerance = self.DEFAULT_FLOOD_FILL_TOLERANCE
            minimum_rgb = tuple(
                max(-255, value - tolerance)
                for value in self.current_contrast_rgb
            )
            maximum_rgb = tuple(
                min(255, value + tolerance)
                for value in self.current_contrast_rgb
            )
            self._set_range_values(minimum_rgb, maximum_rgb)
        self._show_editor_options("class")
        if self.pending_new_class:
            self._refresh_item_list(select_pending=True)
        if classification_type == CLASSIFICATION_RANGE:
            self.status_var.set(
                "Enter inclusive signed RGB contrast minima and maxima."
            )
        elif self.current_image_bgr is None:
            self.status_var.set(
                "Load an image and click a reference region for this sampled class."
            )
        else:
            self.status_var.set("Click a reference region for this sampled class.")

    def _get_contrast_range(self):
        values = []
        for bound, variables in (
            ("minimum", self.range_minimum_vars),
            ("maximum", self.range_maximum_vars),
        ):
            channels = []
            for channel, variable in zip(("Red", "Green", "Blue"), variables):
                raw_value = variable.get().strip()
                try:
                    value = int(raw_value)
                except ValueError as exc:
                    raise ScanProfileError(
                        f"{channel} {bound} contrast must be a whole number."
                    ) from exc
                if not -255 <= value <= 255:
                    raise ScanProfileError(
                        f"{channel} {bound} contrast must be between -255 and 255."
                    )
                channels.append(value)
            values.append(tuple(channels))
        minimum_rgb, maximum_rgb = values
        for channel, minimum, maximum in zip(
            ("Red", "Green", "Blue"),
            minimum_rgb,
            maximum_rgb,
        ):
            if minimum > maximum:
                raise ScanProfileError(
                    f"{channel} minimum contrast cannot exceed its maximum."
                )
        return minimum_rgb, maximum_rgb

    def _set_range_values(self, minimum_rgb, maximum_rgb):
        for variables, values in (
            (self.range_minimum_vars, minimum_rgb),
            (self.range_maximum_vars, maximum_rgb),
        ):
            for index, variable in enumerate(variables):
                variable.set("" if values is None else str(values[index]))

    def _get_class_size_requirement(self):
        return self._parse_size_requirement(
            self.minimum_size_var.get(),
            self.maximum_size_var.get(),
            "Class",
        )

    def _get_profile_size_requirement(self):
        return self._parse_size_requirement(
            self.profile_minimum_size_var.get(),
            self.profile_maximum_size_var.get(),
            "Profile",
        )

    def _parse_size_requirement(self, minimum_text, maximum_text, scope):
        values = []
        for label, raw_value in (
            (f"{scope} minimum size", minimum_text.strip()),
            (f"{scope} maximum size", maximum_text.strip()),
        ):
            if not raw_value:
                values.append(None)
                continue
            try:
                values.append(float(raw_value))
            except ValueError as exc:
                raise ScanProfileError(f"{label} must be a number of micrometers.") from exc
        return self.scan_profile.validate_size_requirement(*values)

    def _preview_pixel_size(self):
        magnification = str(self.app.get_magnification()).upper()
        resolution = self.app.get_resolution()
        try:
            return float(PIXEL_SIZE[magnification][resolution])
        except (KeyError, TypeError, ValueError) as exc:
            has_size_limits = (
                self.scan_profile.minimum_size_um is not None
                or self.scan_profile.maximum_size_um is not None
                or any(
                    item["minimum_size_um"] is not None
                    or item["maximum_size_um"] is not None
                    for item in self.scan_profile.classes
                )
            )
            if has_size_limits:
                raise ScanProfileError(
                    "The current magnification and resolution do not provide a preview pixel size."
                ) from exc
            return None

    def _region_is_ready(self):
        if self.current_image_bgr is None or self.current_source_path is None:
            messagebox.showwarning("No Image", "Load a reference image first.")
            return False
        if self.current_region_mask is None or self.current_seed_point is None:
            messagebox.showwarning("No Region", "Click inside the reference region first.")
            return False
        return True

    def _show_select_region_again(self):
        messagebox.showwarning(
            "Select Region Again",
            "The color tolerance changed after the preview. Click inside the region again.",
        )

    def _find_similar_class(self, contrast_rgb, exclude_index=None):
        if contrast_rgb is None:
            return None
        candidate = np.asarray(contrast_rgb, dtype=np.float64)
        closest = None
        for index, profile_class in enumerate(self.scan_profile.classes):
            if index == exclude_index:
                continue
            if profile_class.get(
                "classification_type",
                CLASSIFICATION_SAMPLE,
            ) != CLASSIFICATION_SAMPLE:
                continue
            distance = float(np.linalg.norm(
                candidate - np.asarray(profile_class["contrast_rgb"], dtype=np.float64)
            ))
            if distance <= self.SIMILAR_CLASS_CONTRAST_DISTANCE and (
                closest is None or distance < closest[1]
            ):
                closest = profile_class, distance
        return closest

    def _next_default_class_name(self):
        existing = {item["name"].casefold() for item in self.scan_profile.classes}
        number = len(self.scan_profile.classes) + 1
        while f"class {number}".casefold() in existing:
            number += 1
        return f"Class {number}"

    def _reset_editor(self, clear_image=True):
        self.class_edit_active = False
        self.editor_kind = None
        self.selected_item = None
        self.editing_class_index = None
        self.pending_new_class = False
        self.class_name_var.set("")
        self.class_group_var.set("")
        self.identify_var.set(True)
        self.classification_type_var.set(
            self.CLASSIFICATION_LABELS[CLASSIFICATION_SAMPLE]
        )
        self._set_range_values(None, None)
        self.minimum_size_var.set("")
        self.maximum_size_var.set("")
        self.threshold_var.set(str(self.DEFAULT_FLOOD_FILL_TOLERANCE))
        self._set_item_editing_enabled(False)
        self._show_editor_options(None)
        if clear_image:
            self._clear_pending_image()

    def _reset_preview(self):
        self.preview_source_path = None
        self.preview_image_bgr = None
        self.preview_result_rgb = None
        self.preview_enabled_var.set(False)
        self.preview_time_var.set("Preview time: —")
        if hasattr(self, "preview_toggle"):
            self.preview_toggle.state(["disabled"])

    def _clear_list_selection(self):
        self.class_list.selection_clear(0, tk.END)

    def _clear_pending_image(self):
        self.current_source_path = None
        self.current_image_bgr = None
        self._clear_region_selection()

    def _clear_region_selection(self):
        self.current_region_mask = None
        self.current_seed_point = None
        self.current_threshold = None
        self.current_contrast_rgb = None
        self._set_contrast_display(None)

    def _set_contrast_display(self, contrast):
        if contrast is None:
            self.contrast_var.set("RGB contrast: —")
        else:
            red, green, blue = contrast
            self.contrast_var.set(f"RGB contrast: ({red}, {green}, {blue})")

    @staticmethod
    def _read_bgr_image(path):
        try:
            encoded = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
            return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
        except (OSError, cv2.error):
            return None

    def _display_bgr(self, image_bgr):
        self.app.display_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    def _choose_profile_path(self, title):
        self.scan_profile.profiles_dir.mkdir(parents=True, exist_ok=True)
        load_folder = messagebox.askyesnocancel(
            title,
            "Use a profile folder?\n\n"
            "Yes: select the folder containing profile.json\n"
            "No: select profile.json directly",
        )
        if load_folder is None:
            return None
        if load_folder:
            return filedialog.askdirectory(
                title=title,
                initialdir=self.scan_profile.profiles_dir,
            )
        return filedialog.askopenfilename(
            title=title,
            initialdir=self.scan_profile.profiles_dir,
            filetypes=[("Scan search profile", "profile.json"), ("JSON", "*.json")],
        )

    @staticmethod
    def _show_confirm_current_message():
        messagebox.showinfo("Confirm Class", "Confirm the current class first.")

    @staticmethod
    def _show_create_or_edit_item_prompt():
        messagebox.showinfo(
            "Create or Edit Class",
            "Create a new class or edit an existing one first.",
        )

    def _schedule_footer_resize(self, *_):
        self.root.after_idle(self._resize_panel_to_footer)

    def _resize_panel_to_footer(self):
        if not hasattr(self, "panel_background"):
            return
        self.panel_background.update_idletasks()
        self.panel_frame.configure(
            width=self.panel_background.winfo_reqwidth() + 4,
            height=self.panel_background.winfo_reqheight(),
        )

    def show(self):
        self.app.close_all_panels()
        self.frame.place(relx=0.0, rely=0.0, anchor="nw")

    def hide(self):
        self.frame.place_forget()
