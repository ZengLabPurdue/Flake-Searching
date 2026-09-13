"""Standalone scan-profile creator and loader.

This shell intentionally contains no microscope, stage, camera, map, or scan
controls. It reuses the main application's profile model and editor panel so
both applications always read and write the same profile format.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
APP_DIRECTORY = REPOSITORY_ROOT / "app"
if str(APP_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(APP_DIRECTORY))

from scanning.scan_profile import ScanProfile  # noqa: E402
from ui.panels.scan_profile_panel import ScanProfilePanel  # noqa: E402


WINDOW_TITLE = "Flake Search Profile Editor"
DEFAULT_MAGNIFICATION = "10X"
DEFAULT_RESOLUTION = "HIGH"
MAGNIFICATIONS = ("2X", "10X", "20X", "100X")
RESOLUTIONS = ("HIGH", "MED", "LOW")


def display_to_source_point(bounds, display_x, display_y):
    """Map a displayed-image click back to its source pixel."""
    if bounds is None:
        return None
    x_offset, y_offset, display_width, display_height, image_width, image_height = (
        bounds
    )
    relative_x = display_x - x_offset
    relative_y = display_y - y_offset
    if not (0 <= relative_x < display_width and 0 <= relative_y < display_height):
        return None
    image_x = min(image_width - 1, int(relative_x * image_width / display_width))
    image_y = min(image_height - 1, int(relative_y * image_height / display_height))
    return image_x, image_y


class ProfileEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry("1280x820")
        self.root.minsize(900, 620)

        self.view_mode = None
        self.active_scan_profile = None
        self._display_photo = None
        self._displayed_image_bounds = None
        self._current_image_rgb = None
        self._resize_job = None

        self.magnification_var = tk.StringVar(value=DEFAULT_MAGNIFICATION)
        self.resolution_var = tk.StringVar(value=DEFAULT_RESOLUTION)

        self._build_menu()

        self.main_frame = tk.Frame(root, bg="#f0f0f0")
        self.main_frame.pack(fill="both", expand=True)
        self.img_label = tk.Label(
            self.main_frame,
            bg="#e7e7e7",
            fg="#555555",
            font=("TkDefaultFont", 14),
            anchor="center",
        )
        self.img_label.pack(fill="both", expand=True)
        self.img_label.bind("<Configure>", self._on_image_area_resized, add="+")

        self.scan_profile = ScanProfile()
        self.scan_profile_panel = ScanProfilePanel(
            parent=self.main_frame,
            root=self.root,
            app=self,
            scan_profile=self.scan_profile,
        )

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after_idle(self.scan_profile_panel.start_create)

    def _build_menu(self):
        menu_bar = tk.Menu(self.root)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Create Profile", command=self._start_create)
        file_menu.add_command(label="Load Profile", command=self._load_profile)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.on_close)
        menu_bar.add_cascade(label="File", menu=file_menu)

        profile_menu = tk.Menu(menu_bar, tearoff=0)
        profile_menu.add_command(label="Create Profile", command=self._start_create)
        profile_menu.add_command(label="Load Profile", command=self._load_profile)
        menu_bar.add_cascade(label="Profile", menu=profile_menu)

        preview_menu = tk.Menu(menu_bar, tearoff=0)
        magnification_menu = tk.Menu(preview_menu, tearoff=0)
        for magnification in MAGNIFICATIONS:
            magnification_menu.add_radiobutton(
                label=magnification,
                variable=self.magnification_var,
                value=magnification,
                command=self._on_preview_calibration_changed,
            )
        preview_menu.add_cascade(
            label="Magnification",
            menu=magnification_menu,
        )
        resolution_menu = tk.Menu(preview_menu, tearoff=0)
        for resolution in RESOLUTIONS:
            resolution_menu.add_radiobutton(
                label=resolution,
                variable=self.resolution_var,
                value=resolution,
                command=self._on_preview_calibration_changed,
            )
        preview_menu.add_cascade(label="Resolution", menu=resolution_menu)
        menu_bar.add_cascade(label="Preview", menu=preview_menu)
        self.root.configure(menu=menu_bar)

    def _start_create(self):
        self.scan_profile_panel.start_create()

    def _load_profile(self):
        self.scan_profile_panel.choose_and_load_profile()

    def _on_preview_calibration_changed(self):
        if hasattr(self, "scan_profile_panel"):
            self.scan_profile_panel.preview_calibration_changed()

    def display_image(self, image, color_order="RGB"):
        color_order = color_order.upper()
        if color_order == "BGR":
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif color_order == "GRAY":
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif color_order == "RGB":
            image_rgb = np.ascontiguousarray(image)
        else:
            raise ValueError(f"Unsupported display color order: {color_order}")
        self._current_image_rgb = np.ascontiguousarray(image_rgb)
        self._render_current_image()

    def _render_current_image(self):
        image_rgb = self._current_image_rgb
        if image_rgb is None:
            return
        label_width = self.img_label.winfo_width()
        label_height = self.img_label.winfo_height()
        if label_width < 10 or label_height < 10:
            self._schedule_image_render()
            return

        image_height, image_width = image_rgb.shape[:2]
        scale = min(label_width / image_width, label_height / image_height)
        display_width = max(1, min(label_width, int(round(image_width * scale))))
        display_height = max(1, min(label_height, int(round(image_height * scale))))
        interpolation = (
            cv2.INTER_AREA
            if display_width < image_width or display_height < image_height
            else cv2.INTER_LINEAR
        )
        resized = cv2.resize(
            image_rgb,
            (display_width, display_height),
            interpolation=interpolation,
        )
        x_offset = (label_width - display_width) // 2
        y_offset = (label_height - display_height) // 2
        self._displayed_image_bounds = (
            x_offset,
            y_offset,
            display_width,
            display_height,
            image_width,
            image_height,
        )

        self._display_photo = ImageTk.PhotoImage(Image.fromarray(resized))
        self.img_label.configure(
            image=self._display_photo,
            text="",
            anchor="center",
        )
        self.img_label.image = self._display_photo

    def _on_image_area_resized(self, _event=None):
        if self._current_image_rgb is not None:
            self._schedule_image_render()

    def _schedule_image_render(self):
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(60, self._finish_scheduled_render)

    def _finish_scheduled_render(self):
        self._resize_job = None
        self._render_current_image()

    def display_image_message(self, message):
        self._current_image_rgb = None
        self._displayed_image_bounds = None
        self._display_photo = None
        self.img_label.configure(
            image="",
            text=message,
            fg="#555555",
            anchor="center",
        )
        self.img_label.image = None

    def display_to_image_point(self, display_x, display_y):
        return display_to_source_point(
            self._displayed_image_bounds,
            display_x,
            display_y,
        )

    def close_all_panels(self):
        if hasattr(self, "scan_profile_panel"):
            self.scan_profile_panel.hide()

    def set_view(self, mode, filter_status=None):
        del filter_status
        self.view_mode = mode
        self.img_label.pack(fill="both", expand=True)
        self.root.update_idletasks()

    def get_view(self):
        return self.view_mode

    def get_magnification(self):
        return self.magnification_var.get()

    def get_resolution(self):
        return self.resolution_var.get()

    def set_active_scan_profile(self, profile):
        self.active_scan_profile = profile
        self.root.title(f"{WINDOW_TITLE} — {profile.name}")

    def get_active_scan_profile(self):
        return self.active_scan_profile

    def on_close(self):
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
            self._resize_job = None
        self.root.destroy()


def main():
    root = tk.Tk()
    ProfileEditorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
