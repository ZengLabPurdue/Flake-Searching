"""Visualize the intermediate stages of the production flake detector.

This is a diagnostic front end only.  Detection and classification are run by
``App/Scanning/contour_finder.py`` so the displayed stages stay aligned with
the microscope application's current processing pipeline.

Run without arguments to choose an image and scan profile interactively::

    python "Flake Recognition/flake_detection_stages.py"

Each stage opens in its own full-size Matplotlib window. Close the current
window to open the next stage; the normal Matplotlib toolbar remains available
for saving. Pass ``--navigator`` for Previous/Next controls, ``--grid`` for the
complete overview, or ``--help`` for all other options.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Button


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
APP_DIRECTORY = REPOSITORY_ROOT / "App"
if str(APP_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(APP_DIRECTORY))

from Imaging import image_metadata  # noqa: E402
from Scanning import contour_finder as _app_contour_finder  # noqa: E402
from config import PIXEL_SIZE  # noqa: E402


APP_CONTOUR_IMPLEMENTATION_PATH = Path(_app_contour_finder.__file__).resolve()
PROFILES_DIRECTORY = APP_DIRECTORY / "Profiles"
DEFAULT_PIXEL_SIZE_UM = float(PIXEL_SIZE["10X"]["HIGH"])
DEFAULT_EDGE_THRESHOLD = 10
EXTERNAL_CONTOUR_RGB = (0, 255, 0)
INTERNAL_CONTOUR_RGB = (255, 0, 0)
EXTERNAL_CONTOUR_CMAP = ListedColormap(
    ((0, 0, 0, 1), (0, 1, 0, 1)),
)
INTERNAL_CONTOUR_CMAP = ListedColormap(
    ((0, 0, 0, 1), (1, 0, 0, 1)),
)
BACKGROUND_EXCLUSION_CMAP = ListedColormap(
    ((0, 0, 0, 0), (0, 0, 0, 1)),
)
PANEL_COLUMNS = 4
VIEW_SEQUENCE = "sequence"
VIEW_NAVIGATOR = "navigator"
VIEW_GRID = "grid"
VIEW_MODES = {VIEW_SEQUENCE, VIEW_NAVIGATOR, VIEW_GRID}
SUPPORTED_IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

_REQUIRED_DETAIL_IMAGES = (
    "image_rgb",
    "closed_edges",
    "external_edge_mask",
    "internal_edge_mask",
    "contour_img",
    "all_external_mask",
    "unfiltered_external_mask",
    "external_mask",
    "all_internal_mask",
    "internal_mask",
    "matched_region_mask",
    "classified_image_without_legend",
)


def read_image_bgr(path: str | Path) -> np.ndarray:
    """Read a color image while supporting spaces and non-ASCII paths."""
    path = Path(path)
    try:
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    except (OSError, cv2.error) as exc:
        raise OSError(f"Could not read image: {path}") from exc
    if image is None:
        raise OSError(f"Could not read image: {path}")
    return image


def _count_title(label: str, count: int | None) -> str:
    return f"{label} ({count:,})" if count is not None else label


def build_stage_panels(details, external_contour_count=None):
    """Return ordered panel specifications for current production stages.

    The external and internal contour panels intentionally use
    ``external_edge_mask`` and ``internal_edge_mask``.  The similarly named
    ``internal_mask`` is a later classification result, not a contour mask.
    """
    missing = [name for name in _REQUIRED_DETAIL_IMAGES if name not in details]
    if missing:
        raise KeyError(
            "Production contour details are missing: " + ", ".join(missing)
        )

    internal_contours = details.get("internal_contours", ())
    internal_contour_count = len(internal_contours)
    external_contours_title = _count_title(
        "External Contours - Green",
        external_contour_count,
    )
    internal_contours_title = _count_title(
        "Internal Contours - Red",
        internal_contour_count,
    )
    return [
        {
            "key": "original",
            "title": "Original",
            "image": details["image_rgb"],
            "cmap": None,
        },
        {
            "key": "closed_edges",
            "title": "Closed Detection Edges",
            "image": details["closed_edges"],
            "cmap": "gray",
        },
        {
            "key": "external_contours",
            "title": external_contours_title,
            "image": details["external_edge_mask"],
            "cmap": EXTERNAL_CONTOUR_CMAP,
        },
        {
            "key": "internal_contours",
            "title": internal_contours_title,
            "image": details["internal_edge_mask"],
            "cmap": INTERNAL_CONTOUR_CMAP,
        },
        {
            "key": "contour_overlay",
            "title": "Contours on Original\nExternal Green / Internal Red",
            "image": details["contour_img"],
            "cmap": None,
        },
        {
            "key": "masked_background",
            "title": "Masked Background\nPixels Used for Color Reference",
            "image": details["image_rgb"],
            "cmap": None,
            "overlay_mask": details["all_external_mask"],
        },
        {
            "key": "external_candidates_before_filters",
            "title": "External Candidates\nBefore Profile Filters",
            "image": details["unfiltered_external_mask"],
            "cmap": "gray",
        },
        {
            "key": "external_candidates_after_filters",
            "title": (
                "Filtered External Candidates\n"
                "White Preview Base Stays Unfiltered"
            ),
            "image": details["external_mask"],
            "cmap": "gray",
        },
        {
            "key": "internal_matches_before_filters",
            "title": "Matched Internal Regions\nBefore Profile Filters",
            "image": details["all_internal_mask"],
            "cmap": "gray",
        },
        {
            "key": "accepted_internal_regions",
            "title": "Accepted Internal Regions\nAfter Profile Filters",
            "image": details["internal_mask"],
            "cmap": "gray",
        },
        {
            "key": "accepted_matched_regions",
            "title": "Accepted Matched Regions\nExternal + Internal",
            "image": details["matched_region_mask"],
            "cmap": "gray",
        },
        {
            "key": "pixel_classification",
            "title": "Final Pixel Classification",
            "image": details["classified_image_without_legend"],
            "cmap": None,
        },
    ]


def _draw_stage_panel(axis, panel, *, title_fontsize=10):
    """Draw one stage specification on an existing Matplotlib axis."""
    axis.clear()
    if panel["cmap"] is None:
        axis.imshow(panel["image"], interpolation="nearest")
    else:
        axis.imshow(
            panel["image"],
            cmap=panel["cmap"],
            vmin=0,
            vmax=255,
            interpolation="nearest",
        )
    overlay_mask = panel.get("overlay_mask")
    if overlay_mask is not None:
        axis.imshow(
            overlay_mask,
            cmap=BACKGROUND_EXCLUSION_CMAP,
            vmin=0,
            vmax=255,
            interpolation="nearest",
        )
    axis.set_title(panel["title"], fontsize=title_fontsize)
    axis.axis("off")


def _stage_figure_title(
    *,
    image_name=None,
    profile_name=None,
    pixel_size_um=None,
    pixel_size_source=None,
):
    title_parts = ["Current Flake Detection Stages"]
    if image_name:
        title_parts.append(str(image_name))
    if profile_name:
        title_parts.append(f"Profile: {profile_name}")
    else:
        title_parts.append("No profile (contours only)")
    if pixel_size_um is not None:
        pixel_size_label = f"{float(pixel_size_um):g} um/pixel"
        if pixel_size_source:
            pixel_size_label += f" ({pixel_size_source})"
        title_parts.append(pixel_size_label)
    return " | ".join(title_parts)


def _single_line_panel_title(panel):
    return panel["title"].replace("\n", " - ")


def create_single_stage_figure(
    panel,
    stage_index,
    stage_count,
    *,
    image_name=None,
    profile_name=None,
    pixel_size_um=None,
    pixel_size_source=None,
):
    """Build one full-size stage figure with the normal save toolbar."""
    height, width = np.asarray(panel["image"]).shape[:2]
    image_aspect = width / max(1, height)
    maximum_width_inches = 13
    maximum_height_inches = 9
    if image_aspect >= maximum_width_inches / maximum_height_inches:
        figure_size = (maximum_width_inches, maximum_width_inches / image_aspect)
    else:
        figure_size = (maximum_height_inches * image_aspect, maximum_height_inches)
    figure = plt.figure(figsize=figure_size)
    axis = figure.add_axes((0, 0, 1, 1))
    _draw_stage_panel(axis, panel)
    axis.set_title("")
    axis.margins(0)

    # The toolbar calls Figure.savefig directly. Crop there so maximizing the
    # GUI cannot add screen-shaped letterboxing to the saved stage image.
    original_savefig = figure.savefig

    def save_image_only(*args, **kwargs):
        kwargs.setdefault("bbox_inches", "tight")
        kwargs.setdefault("pad_inches", 0)
        return original_savefig(*args, **kwargs)

    figure.savefig = save_image_only
    manager = getattr(figure.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        image_label = Path(str(image_name)).stem if image_name else "flake_detection"
        manager.set_window_title(
            f"{image_label} - {stage_index + 1:02d} - "
            f"{_single_line_panel_title(panel)}"
        )
    return figure


def _maximize_figure_window(figure):
    """Best-effort window maximization while preserving the save toolbar."""
    manager = getattr(figure.canvas, "manager", None)
    window = getattr(manager, "window", None)
    if window is None:
        return
    try:
        if hasattr(window, "showMaximized"):
            window.showMaximized()
        elif hasattr(window, "state"):
            window.state("zoomed")
        elif hasattr(window, "Maximize"):
            window.Maximize(True)
    except Exception:
        # Backends differ; the figure still uses nearly all available canvas.
        return


def show_stage_figures_sequentially(
    details,
    *,
    external_contour_count=None,
    image_name=None,
    profile_name=None,
    pixel_size_um=None,
    pixel_size_source=None,
    first_figure=None,
):
    """Show every stage in order, advancing only when its window closes."""
    panels = build_stage_panels(details, external_contour_count)
    figure_arguments = {
        "image_name": image_name,
        "profile_name": profile_name,
        "pixel_size_um": pixel_size_um,
        "pixel_size_source": pixel_size_source,
    }
    for index, panel in enumerate(panels):
        figure = (
            first_figure
            if index == 0 and first_figure is not None
            else create_single_stage_figure(
                panel,
                index,
                len(panels),
                **figure_arguments,
            )
        )
        print(
            f"Showing stage {index + 1}/{len(panels)}: "
            f"{_single_line_panel_title(panel)}. "
            "Close the window to continue."
        )
        _maximize_figure_window(figure)
        plt.show(block=True)
        plt.close(figure)


class StageNavigator:
    """Display one detection stage at a time with button and key navigation."""

    NEXT_KEYS = {
        "right",
        "down",
        "pagedown",
        "enter",
        "n",
        "d",
        " ",
        "space",
    }
    PREVIOUS_KEYS = {"left", "up", "pageup", "backspace", "p", "a"}

    def __init__(self, figure, axis, panels):
        if not panels:
            raise ValueError("At least one detection stage is required.")
        self.figure = figure
        self.axis = axis
        self.panels = list(panels)
        self.index = 0

        previous_axis = figure.add_axes((0.35, 0.025, 0.13, 0.05))
        next_axis = figure.add_axes((0.52, 0.025, 0.13, 0.05))
        self.previous_button = Button(previous_axis, "Previous")
        self.next_button = Button(next_axis, "Next")
        self.previous_button.on_clicked(self.previous)
        self.next_button.on_clicked(self.next)
        self.status_text = figure.text(
            0.5,
            0.105,
            "",
            ha="center",
            va="center",
            fontsize=10,
        )
        self.key_connection = figure.canvas.mpl_connect(
            "key_press_event",
            self.on_key_press,
        )
        self.scroll_connection = figure.canvas.mpl_connect(
            "scroll_event",
            self.on_scroll,
        )
        self.show_stage(0)

    @property
    def current_panel(self):
        return self.panels[self.index]

    def show_stage(self, index):
        self.index = max(0, min(int(index), len(self.panels) - 1))
        _draw_stage_panel(
            self.axis,
            self.current_panel,
            title_fontsize=15,
        )
        self.axis.set_title(
            f"Stage {self.index + 1} of {len(self.panels)}\n"
            f"{self.current_panel['title']}",
            fontsize=15,
        )
        self.status_text.set_text(
            f"Stage {self.index + 1} of {len(self.panels)}  |  "
            "Previous/Next or Left/Right arrows  |  Home/End  |  Q closes"
        )
        self.figure.canvas.draw_idle()

    def previous(self, _event=None):
        self.show_stage(self.index - 1)

    def next(self, _event=None):
        self.show_stage(self.index + 1)

    def on_key_press(self, event):
        if event.key in self.NEXT_KEYS:
            self.next()
        elif event.key in self.PREVIOUS_KEYS:
            self.previous()
        elif event.key == "home":
            self.show_stage(0)
        elif event.key == "end":
            self.show_stage(len(self.panels) - 1)
        elif event.key in {"q", "escape"}:
            plt.close(self.figure)

    def on_scroll(self, event):
        if event.button == "up":
            self.previous()
        elif event.button == "down":
            self.next()


def create_stage_navigator_figure(
    details,
    *,
    external_contour_count=None,
    image_name=None,
    profile_name=None,
    pixel_size_um=None,
    pixel_size_source=None,
):
    """Build a full-size, one-stage-at-a-time interactive figure."""
    panels = build_stage_panels(details, external_contour_count)
    figure, axis = plt.subplots(figsize=(13, 9))
    figure.subplots_adjust(left=0.03, right=0.97, bottom=0.15, top=0.88)
    figure.suptitle(
        _stage_figure_title(
            image_name=image_name,
            profile_name=profile_name,
            pixel_size_um=pixel_size_um,
            pixel_size_source=pixel_size_source,
        ),
        fontsize=13,
    )
    manager = getattr(figure.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title("Flake Detection Stages")
    controller = StageNavigator(figure, axis, panels)
    # Keep callbacks alive and make navigation state inspectable in tests.
    figure._stage_navigator = controller
    return figure


# Compatibility aliases for the initial public version of this diagnostic.
StageSequenceController = StageNavigator
create_stage_sequence_figure = create_stage_navigator_figure


def create_stage_figure(
    details,
    *,
    external_contour_count=None,
    image_name=None,
    profile_name=None,
    pixel_size_um=None,
    pixel_size_source=None,
):
    """Build and return the complete stage-grid figure without showing it."""
    panels = build_stage_panels(details, external_contour_count)
    rows = math.ceil(len(panels) / PANEL_COLUMNS)
    figure, axes = plt.subplots(
        rows,
        PANEL_COLUMNS,
        figsize=(4.6 * PANEL_COLUMNS, 4.0 * rows),
        squeeze=False,
    )

    for axis, panel in zip(axes.ravel(), panels):
        _draw_stage_panel(axis, panel)

    for axis in axes.ravel()[len(panels):]:
        axis.axis("off")

    figure.suptitle(
        _stage_figure_title(
            image_name=image_name,
            profile_name=profile_name,
            pixel_size_um=pixel_size_um,
            pixel_size_source=pixel_size_source,
        ),
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.0, w_pad=1.0)
    return figure


def _profile_display_name(profile_path):
    if profile_path is None:
        return None
    profile_path = Path(profile_path)
    return (
        profile_path.name
        if profile_path.is_dir()
        else (profile_path.parent.name or profile_path.stem)
    )


def visualize_detection_stages(
    image_bgr,
    *,
    profile_path=None,
    edge_threshold=DEFAULT_EDGE_THRESHOLD,
    area_threshold=_app_contour_finder.DEFAULT_AREA_THRESHOLD,
    contrast_threshold=_app_contour_finder.DEFAULT_CONTRAST_MATCH_THRESHOLD,
    region_flood_fill_threshold=(
        _app_contour_finder.DEFAULT_REGION_FLOOD_FILL_THRESHOLD
    ),
    pixel_size_um=None,
    pixel_size_source=None,
    image_name=None,
    display=True,
    save_path=None,
    benchmark=False,
    view_mode=VIEW_SEQUENCE,
):
    """Run the production detector and visualize its intermediate outputs."""
    if view_mode not in VIEW_MODES:
        raise ValueError(
            f"View mode must be one of: {', '.join(sorted(VIEW_MODES))}."
        )
    profile_path = Path(profile_path) if profile_path is not None else None
    _, external_contours, details = _app_contour_finder.find_flakes(
        image_bgr,
        edge_threshold=edge_threshold,
        area_threshold=area_threshold,
        display=False,
        return_details=True,
        profile_path=profile_path,
        contrast_threshold=contrast_threshold,
        region_flood_fill_threshold=region_flood_fill_threshold,
        color_seed=0,
        draw_legend=False,
        benchmark=benchmark,
        pixel_size_um=pixel_size_um,
    )
    details["external_contours"] = external_contours
    profile_name = _profile_display_name(profile_path)
    figure_arguments = {
        "external_contour_count": len(external_contours),
        "image_name": image_name,
        "profile_name": profile_name,
        "pixel_size_um": pixel_size_um,
        "pixel_size_source": pixel_size_source,
    }
    if view_mode == VIEW_GRID:
        figure = create_stage_figure(details, **figure_arguments)
    elif view_mode == VIEW_NAVIGATOR:
        figure = create_stage_navigator_figure(details, **figure_arguments)
    else:
        panels = build_stage_panels(details, len(external_contours))
        figure = create_single_stage_figure(
            panels[0],
            0,
            len(panels),
            image_name=image_name,
            profile_name=profile_name,
            pixel_size_um=pixel_size_um,
            pixel_size_source=pixel_size_source,
        )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        export_figure = (
            figure
            if view_mode == VIEW_GRID
            else create_stage_figure(details, **figure_arguments)
        )
        export_figure.savefig(save_path, dpi=180, bbox_inches="tight")
        if export_figure is not figure:
            plt.close(export_figure)
        print(f"Saved stage plot to: {save_path.resolve()}")
    if display:
        if view_mode == VIEW_SEQUENCE:
            show_stage_figures_sequentially(
                details,
                **figure_arguments,
                first_figure=figure,
            )
        else:
            plt.show()
    return figure, details


def _choose_image_path() -> Path | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        selected = filedialog.askopenfilename(
            parent=root,
            title="Choose an Image for Flake Detection",
            filetypes=[
                ("Images", "*.bmp *.jpeg *.jpg *.png *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
    finally:
        root.destroy()
    return Path(selected) if selected else None


def _discover_profile_paths(profiles_directory=PROFILES_DIRECTORY) -> list[Path]:
    profiles_directory = Path(profiles_directory)
    if not profiles_directory.is_dir():
        return []
    return sorted(
        (
            profile_path
            for profile_path in profiles_directory.glob("*/profile.json")
            if profile_path.is_file()
        ),
        key=lambda profile_path: profile_path.parent.name.casefold(),
    )


def _choose_profile_path() -> Path | None:
    """Choose a discovered scan profile, a custom JSON, or contour-only mode."""
    import tkinter as tk
    from tkinter import filedialog, ttk

    profile_paths = _discover_profile_paths()
    labels_to_paths = {
        profile_path.parent.name: profile_path
        for profile_path in profile_paths
    }
    default_label = next(iter(labels_to_paths), "")

    root = tk.Tk()
    root.title("Choose a Scan Profile")
    root.resizable(False, False)
    selected_label = tk.StringVar(value=default_label)
    result = {"path": None}

    frame = ttk.Frame(root, padding=14)
    frame.grid(row=0, column=0, sticky="nsew")
    ttk.Label(
        frame,
        text="Profile used by the current flake-classification process:",
    ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
    profile_combo = ttk.Combobox(
        frame,
        textvariable=selected_label,
        values=list(labels_to_paths),
        state="readonly",
        width=38,
    )
    profile_combo.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 12))

    def finish(path):
        result["path"] = path
        root.destroy()

    def use_selected_profile():
        selected_path = labels_to_paths.get(selected_label.get())
        if selected_path is not None:
            finish(selected_path)

    def browse_for_profile():
        selected_path = filedialog.askopenfilename(
            parent=root,
            title="Choose a Scan Profile",
            initialdir=PROFILES_DIRECTORY,
            filetypes=[
                ("Scan profile", "profile.json"),
                ("JSON files", "*.json"),
                ("All files", "*.*"),
            ],
        )
        if selected_path:
            finish(Path(selected_path))

    use_button = ttk.Button(
        frame,
        text="Use Profile",
        command=use_selected_profile,
    )
    use_button.grid(row=2, column=0, padx=(0, 8))
    ttk.Button(
        frame,
        text="Browse...",
        command=browse_for_profile,
    ).grid(row=2, column=1, padx=(0, 8))
    ttk.Button(
        frame,
        text="No Profile",
        command=lambda: finish(None),
    ).grid(row=2, column=2)

    if not labels_to_paths:
        profile_combo.configure(state="disabled")
        use_button.configure(state="disabled")
    root.bind("<Return>", lambda _event: use_selected_profile())
    root.bind("<Escape>", lambda _event: finish(None))
    root.protocol("WM_DELETE_WINDOW", lambda: finish(None))
    profile_combo.focus_set()
    root.mainloop()
    return result["path"]


def _pixel_size_for_image(image_path: str | Path, override=None):
    if override is not None:
        return float(override), "command-line override"
    metadata = image_metadata.read_png_metadata(image_path)
    magnification = metadata.get("magnification", "").strip().upper()
    resolution = metadata.get("resolution", "").strip().upper()
    try:
        return float(PIXEL_SIZE[magnification][resolution]), "PNG metadata"
    except KeyError:
        return DEFAULT_PIXEL_SIZE_UM, "10X HIGH fallback"


def _resolve_profile_path(profile_path: str | Path) -> Path:
    profile_path = Path(profile_path).resolve()
    resolved_json = profile_path / "profile.json" if profile_path.is_dir() else profile_path
    if not resolved_json.is_file():
        raise FileNotFoundError(f"Profile was not found: {resolved_json}")
    return profile_path


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Display current production flake-detection stages, including "
            "separate external and internal contours."
        )
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        help="Image to analyze. A file picker opens when omitted.",
    )
    profile_group = parser.add_mutually_exclusive_group()
    profile_group.add_argument(
        "--profile",
        type=Path,
        help="Scan profile directory or profile.json (a picker opens when omitted).",
    )
    profile_group.add_argument(
        "--no-profile",
        action="store_true",
        help="Run contour stages without class matching.",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=DEFAULT_EDGE_THRESHOLD,
        help=f"Production edge threshold (default: {DEFAULT_EDGE_THRESHOLD:g}).",
    )
    parser.add_argument(
        "--area-threshold",
        type=float,
        default=_app_contour_finder.DEFAULT_AREA_THRESHOLD,
        help=(
            "Minimum external contour area in pixels "
            f"(default: {_app_contour_finder.DEFAULT_AREA_THRESHOLD:g})."
        ),
    )
    parser.add_argument(
        "--contrast-threshold",
        type=float,
        default=_app_contour_finder.DEFAULT_CONTRAST_MATCH_THRESHOLD,
        help=(
            "Class-match tolerance used by current scans "
            f"(default: {_app_contour_finder.DEFAULT_CONTRAST_MATCH_THRESHOLD:g})."
        ),
    )
    parser.add_argument(
        "--flood-threshold",
        type=float,
        default=_app_contour_finder.DEFAULT_REGION_FLOOD_FILL_THRESHOLD,
        help=(
            "Region flood-fill color threshold "
            f"(default: {_app_contour_finder.DEFAULT_REGION_FLOOD_FILL_THRESHOLD:g})."
        ),
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="Micrometers per pixel; otherwise use PNG metadata or 10X HIGH fallback.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save the complete stage grid to this image path.",
    )
    view_group = parser.add_mutually_exclusive_group()
    view_group.add_argument(
        "--navigator",
        action="store_true",
        help="Use one window with Previous/Next controls instead of closing each stage.",
    )
    view_group.add_argument(
        "--grid",
        action="store_true",
        help="Show the complete overview grid instead of separate stage windows.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive Matplotlib window.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Print production pipeline timing details.",
    )
    return parser


def _print_summary(details, external_contour_count):
    region_results = list(details.get("region_results", ()))
    internal_count = len(details.get("internal_contours", ()))
    internal_regions = sum(
        result.get("region_type") == "internal"
        for result in region_results
    )
    matched_regions = sum(
        result.get("matched_class") is not None
        for result in region_results
    )
    accepted_regions = len(details.get("combined_region_results", ()))
    print(f"Production implementation: {APP_CONTOUR_IMPLEMENTATION_PATH}")
    print(f"External contours       : {external_contour_count:,}")
    print(f"Internal contours       : {internal_count:,}")
    print(f"Internal regions tested : {internal_regions:,}")
    print(f"Color-matched regions   : {matched_regions:,}")
    print(f"Accepted regions        : {accepted_regions:,}")


def main(argv=None) -> int:
    arguments = _argument_parser().parse_args(argv)
    image_path = arguments.image or _choose_image_path()
    if image_path is None:
        print("No image selected.")
        return 1
    image_path = image_path.resolve()
    if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported image type: {image_path.suffix or '(none)'}")

    if arguments.profile is not None:
        profile_path = _resolve_profile_path(arguments.profile)
    elif arguments.no_profile or arguments.no_show:
        profile_path = None
    else:
        selected_profile = _choose_profile_path()
        profile_path = (
            _resolve_profile_path(selected_profile)
            if selected_profile is not None
            else None
        )

    image_bgr = read_image_bgr(image_path)
    pixel_size_um, pixel_size_source = _pixel_size_for_image(
        image_path,
        arguments.pixel_size,
    )
    if pixel_size_source == "10X HIGH fallback":
        print(
            "Warning: image calibration metadata was not found; using the "
            f"10X HIGH fallback ({pixel_size_um:g} um/pixel). "
            "Use --pixel-size when the image uses another calibration."
        )
    view_mode = (
        VIEW_GRID
        if arguments.grid
        else VIEW_NAVIGATOR
        if arguments.navigator
        else VIEW_SEQUENCE
    )
    figure, details = visualize_detection_stages(
        image_bgr,
        profile_path=profile_path,
        edge_threshold=arguments.edge_threshold,
        area_threshold=arguments.area_threshold,
        contrast_threshold=arguments.contrast_threshold,
        region_flood_fill_threshold=arguments.flood_threshold,
        pixel_size_um=pixel_size_um,
        pixel_size_source=pixel_size_source,
        image_name=image_path.name,
        display=False,
        save_path=arguments.save,
        benchmark=arguments.benchmark,
        view_mode=view_mode,
    )
    external_contour_count = len(details["external_contours"])
    _print_summary(details, external_contour_count)
    if not arguments.no_show:
        if view_mode == VIEW_SEQUENCE:
            show_stage_figures_sequentially(
                details,
                external_contour_count=external_contour_count,
                image_name=image_path.name,
                profile_name=_profile_display_name(profile_path),
                pixel_size_um=pixel_size_um,
                pixel_size_source=pixel_size_source,
                first_figure=figure,
            )
        else:
            plt.show()
    else:
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
