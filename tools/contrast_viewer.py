"""Standalone benchmark and 3D contrast viewer for the App contour pipeline.

The contour detection and region classification functions exposed by this file
come directly from ``app/scanning``. The code below is only a test harness and
visualizer, which prevents this standalone copy from drifting away from the
implementation used during a real scan.

Run with an image path::

    python tools/contrast_viewer.py sample.png

Run without a path to choose an image interactively. Use ``--help`` for profile,
threshold, plot-export, and non-interactive options.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
from time import perf_counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
APP_DIRECTORY = REPOSITORY_ROOT / "app"
if str(APP_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(APP_DIRECTORY))

from scanning import contour_finder as _app_contour_finder  # noqa: E402
from scanning import region_classifier as _app_region_classifier  # noqa: E402
from config import PIXEL_SIZE  # noqa: E402
from imaging import image_metadata  # noqa: E402


APP_CONTOUR_IMPLEMENTATION_PATH = Path(_app_contour_finder.__file__).resolve()
APP_CLASSIFIER_IMPLEMENTATION_PATH = Path(_app_region_classifier.__file__).resolve()
GR_BASE_PROFILE_PATH = (
    APP_DIRECTORY / "Profiles" / "Gr_Base_Profile" / "profile.json"
)
PROFILES_DIRECTORY = APP_DIRECTORY / "Profiles"
DEFAULT_GR_PIXEL_SIZE_UM = float(PIXEL_SIZE["10X"]["HIGH"])
RANGE_BOX_ALPHA = 0.20
RANGE_BOX_EDGE_ALPHA = 0.70
REGION_POINT_EDGE_WIDTH = 0.15
REFERENCE_VOLUME_LINE_WIDTH = 0.5
REFERENCE_SPHERE_ALPHA = 0.18
REFERENCE_SPHERE_EDGE_ALPHA = 0.32
REFERENCE_SPHERE_MIN_RADIUS = 0.5
#REFERENCE_SPHERE_RADIUS = float(
#   _app_region_classifier.DEFAULT_CONTRAST_MATCH_THRESHOLD
#)
REFERENCE_SPHERE_RADIUS = 15.0
REFERENCE_LABEL_GAP_FRACTION = 0.20
REFERENCE_LABEL_MIN_GAP = 8.0
REFERENCE_CONNECTOR_WIDTH = 1.0
REFERENCE_LABEL_DIRECTIONS = np.asarray(
    (
        (1, 1, 1),
        (1, -1, 1),
        (-1, 1, 1),
        (-1, -1, 1),
        (1, 1, -1),
        (1, -1, -1),
        (-1, 1, -1),
        (-1, -1, -1),
        (1, 0, 1),
        (-1, 0, 1),
        (0, 1, -1),
        (0, -1, -1),
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ),
    dtype=np.float64,
)
SUPPORTED_IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

# Preserve the former import surface for other experiments in this directory.
# Every production function now resolves to the App implementation.
_APP_EXPORTS = [
    name
    for name in dir(_app_contour_finder)
    if not name.startswith("__")
]
globals().update({
    name: getattr(_app_contour_finder, name)
    for name in _APP_EXPORTS
})


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


def _actual_point_colors(region_results) -> np.ndarray:
    colors = np.asarray(
        [region["average_color_rgb"] for region in region_results],
        dtype=np.float64,
    )
    return np.clip(colors / 255.0, 0.0, 1.0)


def _region_category(region) -> str:
    if region.get("matched_class") is not None:
        return "identified"
    return "unmatched"


def _contrast_center_radius(contrast_sets) -> tuple[np.ndarray, float]:
    nonempty = [values for values in contrast_sets if len(values)]
    if not nonempty:
        return np.zeros(3, dtype=np.float64), 5.0
    values = np.concatenate(nonempty, axis=0)
    minimum = np.min(values, axis=0)
    maximum = np.max(values, axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(5.0, float(np.max(maximum - minimum)) / 2.0) * 1.08
    return center, radius


def _set_equal_contrast_limits(axis, contrast_sets) -> None:
    center, radius = _contrast_center_radius(contrast_sets)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def _profile_class_plot_color(profile_class, generated_colors):
    color = profile_class.get("display_color_rgb")
    if color is None:
        color = generated_colors[profile_class["name"]]
    color = np.asarray(color, dtype=np.float64)
    if np.max(color) > 1.0:
        color /= 255.0
    return np.clip(color, 0.0, 1.0)


def _plot_contrast_range_box(axis, minimum, maximum, color) -> np.ndarray:
    """Draw a translucent inclusive RGB contrast volume and return its corners."""
    corners = np.asarray([
        (red, green, blue)
        for red in (minimum[0], maximum[0])
        for green in (minimum[1], maximum[1])
        for blue in (minimum[2], maximum[2])
    ], dtype=np.float64)
    faces = [
        corners[[0, 1, 3, 2]],
        corners[[4, 5, 7, 6]],
        corners[[0, 1, 5, 4]],
        corners[[2, 3, 7, 6]],
        corners[[0, 2, 6, 4]],
        corners[[1, 3, 7, 5]],
    ]
    volume = Poly3DCollection(
        faces,
        facecolors=[(*color, RANGE_BOX_ALPHA)],
        edgecolors=[(*color, RANGE_BOX_EDGE_ALPHA)],
        linewidths=REFERENCE_VOLUME_LINE_WIDTH,
    )
    axis.add_collection3d(volume)
    return corners


def _plot_contrast_sphere(axis, center, radius, color):
    """Draw a translucent class-colored sphere around a sampled reference."""
    radius = max(float(radius), REFERENCE_SPHERE_MIN_RADIUS)
    longitude = np.linspace(0.0, 2.0 * np.pi, 32)
    latitude = np.linspace(0.0, np.pi, 18)
    x = center[0] + radius * np.outer(np.cos(longitude), np.sin(latitude))
    y = center[1] + radius * np.outer(np.sin(longitude), np.sin(latitude))
    z = center[2] + radius * np.outer(np.ones_like(longitude), np.cos(latitude))
    surface = axis.plot_surface(
        x,
        y,
        z,
        color=(*color, REFERENCE_SPHERE_ALPHA),
        edgecolor=(*color, REFERENCE_SPHERE_EDGE_ALPHA),
        linewidth=REFERENCE_VOLUME_LINE_WIDTH,
        antialiased=True,
        shade=False,
    )
    return surface, radius


def _reference_label_placement(
    index,
    center,
    boundary_radius,
    all_annotations,
    occupied_label_positions,
    plot_center,
    label_gap,
):
    """Choose a separated 3D label position and its surface connector point."""
    radial = center - plot_center
    radial_length = np.linalg.norm(radial)
    radial = radial / radial_length if radial_length else radial
    best_placement = None
    best_score = -np.inf
    direction_count = len(REFERENCE_LABEL_DIRECTIONS)
    for direction_offset in range(direction_count):
        direction = REFERENCE_LABEL_DIRECTIONS[
            (index + direction_offset) % direction_count
        ].copy()
        direction /= np.linalg.norm(direction)
        line_start = center + direction * boundary_radius
        label_position = line_start + direction * label_gap
        other_clearances = [
            np.linalg.norm(label_position - other_center) - other_radius
            for other_index, (other_center, other_radius, _, _) in enumerate(
                all_annotations
            )
            if other_index != index
        ]
        sphere_clearance = min(other_clearances, default=label_gap)
        label_clearance = min(
            (
                np.linalg.norm(label_position - occupied_position)
                for occupied_position in occupied_label_positions
            ),
            default=label_gap * 2.0,
        )
        outward_alignment = (
            float(np.dot(direction, radial))
            if radial_length
            else 0.0
        )
        score = (
            sphere_clearance
            + 0.35 * label_clearance
            + 0.15 * label_gap * outward_alignment
        )
        if score > best_score:
            best_score = score
            best_placement = (direction, line_start, label_position)
    return best_placement


def plot_contour_contrasts_3d(
    region_results,
    profile_configuration=None,
    *,
    display=True,
    save_path=None,
    annotate_references=True,
    benchmark_seconds=None,
    reference_sphere_radius=REFERENCE_SPHERE_RADIUS,
):
    """Plot all classified regions in signed RGB-contrast space.

    Every region is a circle using its measured average RGB color, regardless
    of whether it matched a class. Translucent class-colored spheres surround
    sampled class references, while translucent boxes show contrast ranges.
    Every sampled reference uses the same contrast-match radius.
    """
    regions = list(region_results)
    configuration = profile_configuration or {"classes": []}
    contrasts = (
        np.asarray([region["contrast_rgb"] for region in regions], dtype=np.float64)
        if regions
        else np.empty((0, 3), dtype=np.float64)
    )
    point_colors = (
        _actual_point_colors(regions)
        if regions
        else np.empty((0, 3), dtype=np.float64)
    )
    categories = np.asarray([_region_category(region) for region in regions])

    figure = plt.figure(figsize=(11, 8.5))
    axis = figure.add_subplot(111, projection="3d")
    if regions:
        axis.scatter(
            contrasts[:, 0],
            contrasts[:, 1],
            contrasts[:, 2],
            c=point_colors,
            marker="o",
            s=24,
            edgecolors="black",
            linewidths=REGION_POINT_EDGE_WIDTH,
            depthshade=False,
            label="Contour regions",
        )

    profile_classes = list(configuration.get("classes", ()))
    generated_colors = _app_region_classifier.generate_class_colors(
        profile_classes,
        color_seed=0,
    )
    reference_contrasts = []
    reference_annotations = []
    for profile_class in profile_classes:
        if not profile_class.get("identify", True):
            continue
        class_color = _profile_class_plot_color(profile_class, generated_colors)
        classification_type = profile_class.get("classification_type", "sample")
        if classification_type == "range":
            minimum = np.asarray(
                profile_class["minimum_contrast_rgb"],
                dtype=np.float64,
            )
            maximum = np.asarray(
                profile_class["maximum_contrast_rgb"],
                dtype=np.float64,
            )
            corners = _plot_contrast_range_box(
                axis,
                minimum,
                maximum,
                class_color,
            )
            reference_contrasts.extend(corners)
            contrast = (minimum + maximum) / 2.0
            annotation_radius = float(np.linalg.norm((maximum - minimum) / 2.0))
        else:
            contrast = np.asarray(profile_class["contrast_rgb"], dtype=np.float64)
            reference_contrasts.append(contrast)
            _, annotation_radius = _plot_contrast_sphere(
                axis,
                contrast,
                reference_sphere_radius,
                class_color,
            )
        if annotate_references:
            group = profile_class.get("group")
            label = (
                f"{group} / {profile_class['name']}"
                if group
                else profile_class["name"]
            )
            reference_annotations.append(
                (contrast, annotation_radius, label, class_color)
            )

    axis.scatter(
        [0],
        [0],
        [0],
        c="black",
        marker="+",
        s=60,
        linewidths=1.2,
        depthshade=False,
    )
    reference_array = (
        np.asarray(reference_contrasts, dtype=np.float64).reshape(-1, 3)
        if reference_contrasts
        else np.empty((0, 3), dtype=np.float64)
    )
    base_contrast_sets = (contrasts, reference_array, np.zeros((1, 3)))
    plot_center, plot_radius = _contrast_center_radius(base_contrast_sets)
    label_gap = max(
        REFERENCE_LABEL_MIN_GAP,
        plot_radius * REFERENCE_LABEL_GAP_FRACTION,
    )
    label_positions = []
    for index, (center, boundary_radius, label, class_color) in enumerate(
        reference_annotations
    ):
        direction, line_start, label_position = _reference_label_placement(
            index,
            center,
            boundary_radius,
            reference_annotations,
            label_positions,
            plot_center,
            label_gap,
        )
        axis.plot(
            [line_start[0], label_position[0]],
            [line_start[1], label_position[1]],
            [line_start[2], label_position[2]],
            color=class_color,
            linewidth=REFERENCE_CONNECTOR_WIDTH,
            alpha=0.9,
        )
        axis.text(
            label_position[0],
            label_position[1],
            label_position[2],
            label,
            color=class_color,
            fontsize=9,
            fontweight="bold",
            horizontalalignment="left" if direction[0] >= 0 else "right",
            verticalalignment="center",
        )
        label_positions.append(label_position)
    label_array = (
        np.asarray(label_positions, dtype=np.float64).reshape(-1, 3)
        if label_positions
        else np.empty((0, 3), dtype=np.float64)
    )
    _set_equal_contrast_limits(axis, (*base_contrast_sets, label_array))

    counts = Counter(categories.tolist())
    axis.set_xlabel("Red contrast (region - background)")
    axis.set_ylabel("Green contrast (region - background)")
    axis.set_zlabel("Blue contrast (region - background)")
    timing = (
        f" | processing: {benchmark_seconds:.3f} s"
        if benchmark_seconds is not None
        else ""
    )
    axis.set_title(
        "Contour-region RGB contrasts\n"
        f"{len(regions):,} total | {counts['identified']:,} identified | "
        f"{counts['unmatched']:,} unmatched"
        f"{timing}"
    )
    figure.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Saved 3D contrast plot to: {save_path.resolve()}")
    if display:
        plt.show()
    return figure


def print_region_summary(region_results) -> None:
    regions = list(region_results)
    counts = Counter(_region_category(region) for region in regions)
    classes = Counter(
        region["matched_class"]
        for region in regions
        if region.get("matched_class") is not None
    )
    print("\nRegion summary")
    print("-" * 48)
    print(f"Total regions : {len(regions):,}")
    print(f"Identified    : {counts['identified']:,}")
    print(f"Unmatched     : {counts['unmatched']:,}")
    for name, count in sorted(classes.items()):
        print(f"  {name}: {count:,}")


def print_region_points(region_results) -> None:
    """Print every plotted point in a compact, copyable text format."""
    print("\nindex  state       class                      contrast RGB       actual RGB")
    print("-" * 88)
    for index, region in enumerate(region_results, start=1):
        state = _region_category(region)
        label = region.get("matched_class") or "-"
        contrast = tuple(region["contrast_rgb"])
        actual = tuple(region["average_color_rgb"])
        print(
            f"{index:>5}  {state:<11} {label:<26.26} "
            f"{str(contrast):<20} {actual}"
        )


def benchmark_and_plot(
    image_bgr,
    profile_path=None,
    *,
    edge_threshold=10,
    area_threshold=DEFAULT_AREA_THRESHOLD,
    contrast_threshold=None,
    region_flood_fill_threshold=DEFAULT_REGION_FLOOD_FILL_THRESHOLD,
    pixel_size_um=None,
    legend_position=DEFAULT_LEGEND_POSITION,
    display=True,
    save_plot=None,
    annotate_references=True,
    print_points=False,
):
    """Benchmark the production pipeline and plot every resulting contrast."""
    classified_image, contours, details = _app_contour_finder.find_flakes(
        image_bgr,
        edge_threshold=edge_threshold,
        area_threshold=area_threshold,
        return_details=True,
        profile_path=profile_path,
        contrast_threshold=contrast_threshold,
        region_flood_fill_threshold=region_flood_fill_threshold,
        color_seed=0,
        benchmark=True,
        pixel_size_um=pixel_size_um,
        legend_position=legend_position,
    )
    configuration = (
        _app_region_classifier.load_profile_configuration(
            profile_path,
            contrast_threshold,
        )
        if profile_path is not None
        else {"classes": []}
    )
    print_region_summary(details["region_results"])
    if print_points:
        print_region_points(details["region_results"])
    figure = plot_contour_contrasts_3d(
        details["region_results"],
        configuration,
        display=display,
        save_path=save_plot,
        annotate_references=annotate_references,
        benchmark_seconds=details["benchmark"]["wall_time_seconds"],
        reference_sphere_radius=(
            contrast_threshold
            if contrast_threshold is not None
            else REFERENCE_SPHERE_RADIUS
        ),
    )
    return classified_image, contours, details, figure


def _choose_image_path() -> Path | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        selected = filedialog.askopenfilename(
            title="Choose an Image for the 3D Contrast Plot",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")],
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


def _choose_profile_path(default_profile=None) -> Path | None:
    import tkinter as tk
    from tkinter import filedialog, ttk

    profile_paths = _discover_profile_paths()
    labels_to_paths = {
        profile_path.parent.name: profile_path
        for profile_path in profile_paths
    }
    default_profile = Path(default_profile).resolve() if default_profile else None
    default_label = next(
        (
            label
            for label, profile_path in labels_to_paths.items()
            if profile_path.resolve() == default_profile
        ),
        next(iter(labels_to_paths), ""),
    )

    root = tk.Tk()
    root.title("Choose a Scan Profile")
    root.resizable(False, False)
    selected_label = tk.StringVar(value=default_label)
    result = {"path": None}

    frame = ttk.Frame(root, padding=14)
    frame.grid(row=0, column=0, sticky="nsew")
    ttk.Label(
        frame,
        text="Profile used for classification and the 3D range display:",
    ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
    profile_combo = ttk.Combobox(
        frame,
        textvariable=selected_label,
        values=list(labels_to_paths),
        state="readonly",
        width=38,
    )
    profile_combo.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 12))

    def use_selected_profile():
        selected_path = labels_to_paths.get(selected_label.get())
        if selected_path is not None:
            result["path"] = selected_path
            root.destroy()

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
            result["path"] = Path(selected_path)
            root.destroy()

    def use_without_profile():
        result["path"] = None
        root.destroy()

    use_button = ttk.Button(
        frame,
        text="Use Profile",
        command=use_selected_profile,
    )
    use_button.grid(row=2, column=0, padx=(0, 8))
    ttk.Button(
        frame,
        text="Browse…",
        command=browse_for_profile,
    ).grid(row=2, column=1, padx=(0, 8))
    ttk.Button(
        frame,
        text="No Profile",
        command=use_without_profile,
    ).grid(row=2, column=2)

    if not labels_to_paths:
        profile_combo.configure(state="disabled")
        use_button.configure(state="disabled")
    root.bind("<Return>", lambda _event: use_selected_profile())
    root.bind("<Escape>", lambda _event: use_without_profile())
    root.protocol("WM_DELETE_WINDOW", use_without_profile)
    profile_combo.focus_set()
    root.mainloop()
    return result["path"]


def _pixel_size_for_image(image_path, override=None) -> float:
    if override is not None:
        return float(override)
    metadata = image_metadata.read_png_metadata(image_path)
    magnification = metadata.get("magnification", "").strip().upper()
    resolution = metadata.get("resolution", "").strip().upper()
    try:
        return float(PIXEL_SIZE[magnification][resolution])
    except KeyError:
        return DEFAULT_GR_PIXEL_SIZE_UM


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the App contour classifier and plot every region in "
            "signed RGB-contrast space using its measured color."
        )
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        help="Image to analyze. A file picker opens when omitted.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help=(
            "Profile JSON or profile directory "
            "(interactive runs prompt; otherwise the default is "
            "app/Profiles/Gr_Base_Profile)."
        ),
    )
    parser.add_argument("--edge-threshold", type=float, default=10)
    parser.add_argument("--area-threshold", type=float, default=DEFAULT_AREA_THRESHOLD)
    parser.add_argument(
        "--contrast-threshold",
        type=float,
        default=None,
        help="Override every class tolerance instead of using the profile values.",
    )
    parser.add_argument(
        "--flood-threshold",
        type=float,
        default=DEFAULT_REGION_FLOOD_FILL_THRESHOLD,
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help=(
            "Micrometers per pixel. When omitted, use PNG camera metadata or "
            f"the Gr profile's 10X HIGH fallback ({DEFAULT_GR_PIXEL_SIZE_UM:g})."
        ),
    )
    parser.add_argument(
        "--legend-position",
        choices=sorted(LEGEND_POSITIONS),
        default=DEFAULT_LEGEND_POSITION,
    )
    parser.add_argument("--save-plot", type=Path, default=None)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive Matplotlib window.",
    )
    parser.add_argument(
        "--no-reference-labels",
        action="store_true",
        help="Hide class names beside their reference markers.",
    )
    parser.add_argument(
        "--print-points",
        action="store_true",
        help="Also print every plotted contrast and actual RGB value.",
    )
    return parser


def main(argv=None) -> int:
    arguments = _argument_parser().parse_args(argv)
    image_path = arguments.image or _choose_image_path()
    if image_path is None:
        print("No image selected.")
        return 1
    image_path = image_path.resolve()
    profile_path = arguments.profile
    if profile_path is None:
        default_profile = (
            GR_BASE_PROFILE_PATH
            if GR_BASE_PROFILE_PATH.is_file()
            else None
        )
        profile_path = (
            _choose_profile_path(default_profile)
            if not arguments.no_show
            else default_profile
        )
    if profile_path is not None:
        profile_path = profile_path.resolve()
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile was not found: {profile_path}")

    pixel_size_um = _pixel_size_for_image(image_path, arguments.pixel_size)
    print(f"Contour implementation   : {APP_CONTOUR_IMPLEMENTATION_PATH}")
    print(f"Classifier implementation: {APP_CLASSIFIER_IMPLEMENTATION_PATH}")
    print(f"Image                    : {image_path}")
    print(f"Profile                  : {profile_path if profile_path else 'none'}")
    print(f"Pixel size               : {pixel_size_um:g} um/pixel")
    image_bgr = read_image_bgr(image_path)
    print(f"Image dimensions         : {image_bgr.shape[1]} x {image_bgr.shape[0]}")

    benchmark_and_plot(
        image_bgr,
        profile_path,
        edge_threshold=arguments.edge_threshold,
        area_threshold=arguments.area_threshold,
        contrast_threshold=arguments.contrast_threshold,
        region_flood_fill_threshold=arguments.flood_threshold,
        pixel_size_um=pixel_size_um,
        legend_position=arguments.legend_position,
        display=not arguments.no_show,
        save_plot=arguments.save_plot,
        annotate_references=not arguments.no_reference_labels,
        print_points=arguments.print_points,
    )
    return 0


def _choose_directory_path() -> Path | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        selected = filedialog.askdirectory(
            title="Choose an Image Directory to Benchmark"
        )
    finally:
        root.destroy()
    return Path(selected) if selected else None


def _directory_argument_parser() -> argparse.ArgumentParser:
    default_profile = GR_BASE_PROFILE_PATH
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the production contour classifier over every supported "
            "image in a directory without plotting or saving output images."
        )
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        help="Directory to benchmark. A directory picker opens when omitted.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=default_profile if default_profile.is_file() else None,
        help=(
            "Profile JSON or profile directory "
            "(default: app/Profiles/Gr_Base_Profile)."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Include images in nested directories.",
    )
    parser.add_argument("--edge-threshold", type=float, default=10)
    parser.add_argument(
        "--area-threshold",
        type=float,
        default=DEFAULT_AREA_THRESHOLD,
    )
    parser.add_argument(
        "--contrast-threshold",
        type=float,
        default=None,
        help="Override every class tolerance instead of using the profile values.",
    )
    parser.add_argument(
        "--flood-threshold",
        type=float,
        default=DEFAULT_REGION_FLOOD_FILL_THRESHOLD,
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=DEFAULT_GR_PIXEL_SIZE_UM,
        help="Micrometers per pixel (default: Gr profile 10X HIGH calibration).",
    )
    return parser


def _discover_images(directory: Path, recursive: bool) -> list[Path]:
    candidates = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        (
            path
            for path in candidates
            if path.is_file() and path.suffix.casefold() in SUPPORTED_IMAGE_SUFFIXES
        ),
        key=lambda path: str(path.relative_to(directory)).casefold(),
    )


def _merge_benchmark(aggregate, benchmark) -> None:
    for name, measurement in benchmark["functions"].items():
        destination = aggregate.setdefault(
            name,
            {"calls": 0, "total_seconds": 0.0, "max_seconds": 0.0},
        )
        destination["calls"] += measurement["calls"]
        destination["total_seconds"] += measurement["total_seconds"]
        destination["max_seconds"] = max(
            destination["max_seconds"],
            measurement["max_seconds"],
        )


def _print_aggregate_benchmark(aggregate, pipeline_seconds) -> None:
    print("\nDirectory benchmark (inclusive timings across successful images)")
    print(
        f"{'Function':<30} {'Calls':>9} {'Total ms':>12} "
        f"{'Average ms':>12} {'Max ms':>12}"
    )
    print("-" * 79)
    for name, measurement in sorted(
        aggregate.items(),
        key=lambda item: item[1]["total_seconds"],
        reverse=True,
    ):
        calls = measurement["calls"]
        print(
            f"{name:<30} {calls:>9,d} "
            f"{measurement['total_seconds'] * 1000:>12.3f} "
            f"{measurement['total_seconds'] / calls * 1000:>12.3f} "
            f"{measurement['max_seconds'] * 1000:>12.3f}"
        )
    print("-" * 79)
    print(f"{'Combined pipeline time':<40} {pipeline_seconds * 1000:>12.3f}")


def benchmark_directory(
    directory,
    profile_path=None,
    *,
    recursive=False,
    edge_threshold=10,
    area_threshold=DEFAULT_AREA_THRESHOLD,
    contrast_threshold=None,
    region_flood_fill_threshold=DEFAULT_REGION_FLOOD_FILL_THRESHOLD,
    pixel_size_um=None,
) -> int:
    """Benchmark all images in a directory without displaying or saving them."""
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Image directory was not found: {directory}")
    profile_path = Path(profile_path).resolve() if profile_path is not None else None
    if profile_path is not None and not profile_path.exists():
        raise FileNotFoundError(f"Profile was not found: {profile_path}")

    image_paths = _discover_images(directory, recursive)
    if not image_paths:
        print(f"No supported images found in: {directory}")
        return 1

    print(f"Contour implementation  : {APP_CONTOUR_IMPLEMENTATION_PATH}")
    print(f"Classifier implementation: {APP_CLASSIFIER_IMPLEMENTATION_PATH}")
    print(f"Image directory         : {directory}")
    print(f"Profile                 : {profile_path if profile_path else 'none'}")
    print(f"Images discovered       : {len(image_paths):,}")

    aggregate = {}
    combined_pipeline_seconds = 0.0
    successful = 0
    failed = 0
    total_regions = 0
    total_identified = 0
    batch_started_at = perf_counter()

    for index, image_path in enumerate(image_paths, start=1):
        relative_path = image_path.relative_to(directory)
        prefix = f"[{index:>{len(str(len(image_paths)))}}/{len(image_paths)}]"
        print(f"{prefix} Processing {relative_path} ...", flush=True)
        try:
            image_bgr = read_image_bgr(image_path)
            # find_flakes prints an individual benchmark table. Suppress it so
            # directory mode can emit one status line and one aggregate table.
            with redirect_stdout(io.StringIO()):
                _, _, details = _app_contour_finder.find_flakes(
                    image_bgr,
                    edge_threshold=edge_threshold,
                    area_threshold=area_threshold,
                    return_details=True,
                    profile_path=profile_path,
                    contrast_threshold=contrast_threshold,
                    region_flood_fill_threshold=region_flood_fill_threshold,
                    color_seed=0,
                    draw_legend=False,
                    benchmark=True,
                    pixel_size_um=pixel_size_um,
                )
            benchmark = details["benchmark"]
            regions = details["region_results"]
            identified = sum(
                region["matched_class"] is not None for region in regions
            )
            pipeline_seconds = benchmark["wall_time_seconds"]
            _merge_benchmark(aggregate, benchmark)
            combined_pipeline_seconds += pipeline_seconds
            successful += 1
            total_regions += len(regions)
            total_identified += identified
            print(
                f"{prefix} Completed {relative_path} | "
                f"{pipeline_seconds:.3f} s | {len(regions):,} regions | "
                f"{identified:,} identified",
                flush=True,
            )
        except Exception as exc:  # Continue benchmarking the remaining images.
            failed += 1
            print(
                f"{prefix} FAILED {relative_path} | "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    batch_seconds = perf_counter() - batch_started_at
    if aggregate:
        _print_aggregate_benchmark(aggregate, combined_pipeline_seconds)
    print("\nDirectory summary")
    print(f"Successful images : {successful:,}")
    print(f"Failed images     : {failed:,}")
    print(f"Total regions     : {total_regions:,}")
    print(f"Identified regions: {total_identified:,}")
    print(f"Batch elapsed time: {batch_seconds:.3f} s")
    return 0 if successful and not failed else 1


def directory_main(argv=None) -> int:
    arguments = _directory_argument_parser().parse_args(argv)
    directory = arguments.directory or _choose_directory_path()
    if directory is None:
        print("No directory selected.")
        return 1
    return benchmark_directory(
        directory,
        arguments.profile,
        recursive=arguments.recursive,
        edge_threshold=arguments.edge_threshold,
        area_threshold=arguments.area_threshold,
        contrast_threshold=arguments.contrast_threshold,
        region_flood_fill_threshold=arguments.flood_threshold,
        pixel_size_um=arguments.pixel_size,
    )


__all__ = _APP_EXPORTS + [
    "APP_CONTOUR_IMPLEMENTATION_PATH",
    "APP_CLASSIFIER_IMPLEMENTATION_PATH",
    "GR_BASE_PROFILE_PATH",
    "DEFAULT_GR_PIXEL_SIZE_UM",
    "read_image_bgr",
    "plot_contour_contrasts_3d",
    "print_region_summary",
    "print_region_points",
    "benchmark_and_plot",
    "benchmark_directory",
    "directory_main",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
