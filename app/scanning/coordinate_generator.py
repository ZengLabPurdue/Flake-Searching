import math

import cv2
import numpy as np

from config import CROP_RATIO, PIXEL_SIZE, RELATIVE_XY, RESOLUTION_DIM


def fit_map_to_stage_affine(map_points, stage_points):
    """Fit the 2D affine transform from stitched-map pixels to stage microns."""
    map_points = np.asarray(map_points, dtype=np.float64)
    stage_points = np.asarray(stage_points, dtype=np.float64)
    if (
        map_points.ndim != 2
        or stage_points.ndim != 2
        or map_points.shape[1:] != (2,)
        or stage_points.shape != map_points.shape
        or len(map_points) < 3
    ):
        raise ValueError(
            "At least three paired 2D map and stage points are required."
        )

    design = np.column_stack((map_points, np.ones(len(map_points))))
    coefficients, _, rank, _ = np.linalg.lstsq(
        design,
        stage_points,
        rcond=None,
    )
    if rank < 3 or not np.all(np.isfinite(coefficients)):
        raise ValueError("Map control points do not define a 2D affine transform.")
    # Return the conventional 2 x 3 matrix so stage = affine @ [x, y, 1].
    return coefficients.T


def map_point_to_stage(map_to_stage_affine, map_x, map_y):
    affine = np.asarray(map_to_stage_affine, dtype=np.float64)
    if affine.shape != (2, 3) or not np.all(np.isfinite(affine)):
        raise ValueError(
            "The map-to-stage affine transform must be a finite 2 x 3 matrix."
        )
    stage_x, stage_y = affine @ np.array(
        [float(map_x), float(map_y), 1.0],
        dtype=np.float64,
    )
    return float(stage_x), float(stage_y)


def _objective_aligned_center(
    map_x,
    map_y,
    target_magnification,
    scan_center_x,
    scan_center_y,
    scale,
    true_map,
    camera_size,
    resolution,
    map_to_stage_affine,
):
    """Convert a point in the stitched 2X map into a target-objective position."""
    if map_to_stage_affine is not None:
        center_x, center_y = map_point_to_stage(
            map_to_stage_affine,
            map_x,
            map_y,
        )
    else:
        camera_width, camera_height = camera_size
        # Mapper.place_frame_on_map uses a negative stage-to-map direction on
        # both axes. This nominal fallback is for scans without saved 2X tile
        # controls; normal full scans use the fitted affine transform above.
        center_x = -(
            map_x - true_map.shape[1] / 2
        ) * (
            PIXEL_SIZE["2X"][resolution]
            * RESOLUTION_DIM[resolution]["x"]
            * CROP_RATIO["2X"]["x"]
        ) / (
            camera_width / scale * CROP_RATIO["2X"]["x"]
        ) + scan_center_x
        center_y = -(
            map_y - true_map.shape[0] / 2
        ) * (
            PIXEL_SIZE["2X"][resolution]
            * RESOLUTION_DIM[resolution]["y"]
            * CROP_RATIO["2X"]["y"]
        ) / (
            camera_height / scale * CROP_RATIO["2X"]["y"]
        ) + scan_center_y

    source_offset = RELATIVE_XY["2X"]
    target_offset = RELATIVE_XY[target_magnification]
    return (
        center_x + target_offset["X"] - source_offset["X"],
        center_y + target_offset["Y"] - source_offset["Y"],
    )

def generate_rect_coords(x, y, div2=False):
    rect_coords = []
    total_frames = x * y

    # Integer division biases every even-sized grid by half a field. Center
    # even axes between their two middle positions instead.
    center_x = (x - 1) / 2
    center_y = (y - 1) / 2
    spacing = 0.5 if div2 else 1.0

    for i in range(x):
        if i % 2 == 0:
            y_range = range(y)
        else:
            y_range = range(y - 1, -1, -1)

        for j in y_range:
            rect_coords.append(
                (
                    (i - center_x) * spacing,
                    (j - center_y) * spacing,
                )
            )

    return rect_coords, total_frames

def generate_spiral_coords(length):
    spiral_coords = []
    total_frames = length ** 2

    dx, dy = 0, 0
    step = 1
    direction = 0

    while len(spiral_coords) < total_frames:
        for _ in range(2):
            for _ in range(step):
                if len(spiral_coords) >= total_frames:
                    break

                spiral_coords.append((dx, dy))

                if direction == 0:
                    dx += 1
                elif direction == 1:
                    dy += 1
                elif direction == 2:
                    dx -= 1
                else:
                    dy -= 1

            direction = (direction + 1) % 4

        step += 1

    return spiral_coords, total_frames

def generate_10x_scan_coordinates(
    app,
    chips,
    scan_center_x,
    scan_center_y,
    scale,
    true_map,
    camera_size,
    map_to_stage_affine=None,
):
    camera_width, camera_height = camera_size

    resolution = app.get_resolution()
    window_w = int(camera_width / scale / (PIXEL_SIZE["2X"][resolution] / PIXEL_SIZE["10X"][resolution]) * CROP_RATIO["10X"]["x"])
    window_h = int(camera_height / scale / (PIXEL_SIZE["2X"][resolution] / PIXEL_SIZE["10X"][resolution]) * CROP_RATIO["10X"]["y"])

    scan_coordinates_10x = []

    for chip in chips:
        x, y, w, h = chip

        num_windows_x = math.ceil(w / window_w)
        num_windows_y = math.ceil(h / window_h)

        grid_w = num_windows_x * window_w
        grid_h = num_windows_y * window_h

        chip_center_x = x + w // 2
        chip_center_y = y + h // 2

        start_x = max(0, chip_center_x - grid_w // 2)
        start_y = max(0, chip_center_y - grid_h // 2)

        start_pos_x, start_pos_y = _objective_aligned_center(
            chip_center_x,
            chip_center_y,
            "10X",
            scan_center_x,
            scan_center_y,
            scale,
            true_map,
            camera_size,
            resolution,
            map_to_stage_affine,
        )

        scan_coordinates_10x.append([
            int(round(start_pos_x)),
            int(round(start_pos_y)),
            num_windows_x,
            num_windows_y,
        ])

        print(f"Scan Start: ({scan_center_x}, {scan_center_y})")
        print(f"Chip Center: ({start_pos_x}, {start_pos_y})")
        print(f"Windows: ({num_windows_x}, {num_windows_y})")

        for i in range(num_windows_x):
            for j in range(num_windows_y):
                wx = start_x + i * window_w
                wy = start_y + j * window_h

                cv2.rectangle(true_map, (wx, wy), (wx + window_w, wy + window_h), (0, 255, 0), 5, cv2.LINE_AA)

        cv2.circle(true_map, (chip_center_x, chip_center_y), 8, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(true_map, (int(true_map.shape[1] / 2), int(true_map.shape[0] / 2)), 8, (255, 0, 0), -1, cv2.LINE_AA)

    return scan_coordinates_10x
    
def generate_20x_scan_coordinates(
    app,
    chips,
    scan_center_x,
    scan_center_y,
    scale,
    true_map,
    camera_size,
    map_to_stage_affine=None,
):
    camera_width, camera_height = camera_size

    resolution = app.get_resolution()
    window_w = int(camera_width / scale / (PIXEL_SIZE["2X"][resolution] / PIXEL_SIZE["20X"][resolution]) * CROP_RATIO["20X"]["x"])
    window_h = int(camera_height / scale / (PIXEL_SIZE["2X"][resolution] / PIXEL_SIZE["20X"][resolution]) * CROP_RATIO["20X"]["y"])

    scan_coordinates_20x = []

    for chip in chips:
        x, y, w, h = chip

        num_windows_x = math.ceil(w / window_w)
        num_windows_y = math.ceil(h / window_h)

        grid_w = num_windows_x * window_w
        grid_h = num_windows_y * window_h

        chip_center_x = x + w // 2
        chip_center_y = y + h // 2

        start_x = max(0, chip_center_x - grid_w // 2)
        start_y = max(0, chip_center_y - grid_h // 2)

        start_pos_x, start_pos_y = _objective_aligned_center(
            chip_center_x,
            chip_center_y,
            "20X",
            scan_center_x,
            scan_center_y,
            scale,
            true_map,
            camera_size,
            resolution,
            map_to_stage_affine,
        )

        scan_coordinates_20x.append([
            int(round(start_pos_x)),
            int(round(start_pos_y)),
            num_windows_x,
            num_windows_y,
        ])

        print(f"Scan Start: ({scan_center_x}, {scan_center_y})")
        print(f"Chip Center: ({start_pos_x}, {start_pos_y})")
        print(f"Windows: ({num_windows_x}, {num_windows_y})")

        for i in range(num_windows_x):
            for j in range(num_windows_y):
                wx = start_x + i * window_w
                wy = start_y + j * window_h

                cv2.rectangle(true_map, (wx, wy), (wx + window_w, wy + window_h), (0, 255, 0), 5, cv2.LINE_AA)

        cv2.circle(true_map, (chip_center_x, chip_center_y), 8, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(true_map, (int(true_map.shape[1] / 2), int(true_map.shape[0] / 2)), 8, (255, 0, 0), -1, cv2.LINE_AA)

    return scan_coordinates_20x
