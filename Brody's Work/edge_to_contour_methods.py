"""
Methods to turn edges (with small gaps) into contours.
"""
import cv2
import numpy as np
from typing import List, Tuple


def _ensure_binary(edges: np.ndarray) -> np.ndarray:
    if edges.max() <= 1:
        return (edges * 255).astype(np.uint8)
    _, out = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
    return out


# -----------------------------------------------------------------------------
# 1. Morphological closing
# -----------------------------------------------------------------------------
def close_then_contours(
    edges: np.ndarray,
    kernel_size: int = 5,
    kernel_shape: str = "rect",
    min_area: int = 0,
) -> List[np.ndarray]:
    """Bridge gaps by closing (dilate + erode), then find outer contours."""
    binary = _ensure_binary(edges)
    if kernel_size > 0:
        k = (kernel_size, kernel_size)
        kernel = (
            cv2.getStructuringElement(cv2.MORPH_RECT, k)
            if kernel_shape == "rect"
            else cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if min_area > 0:
        contours = [c for c in contours if cv2.contourArea(c) > min_area]
    return contours


# -----------------------------------------------------------------------------
# 2. Iterative small closing
# -----------------------------------------------------------------------------
def iterative_close_then_contours(
    edges: np.ndarray,
    iterations: int = 3,
    kernel_size: int = 3,
    min_area: int = 0,
) -> List[np.ndarray]:
    """Apply small closing multiple times."""
    binary = _ensure_binary(edges)
    k = (kernel_size, kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k)
    for _ in range(iterations):
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if min_area > 0:
        contours = [c for c in contours if cv2.contourArea(c) > min_area]
    return contours


# -----------------------------------------------------------------------------
# 3. Dilate-only then contour
# -----------------------------------------------------------------------------
def dilate_then_contours(
    edges: np.ndarray,
    dilate_size: int = 2,
    min_area: int = 0,
) -> List[np.ndarray]:
    """Dilate edges (no erode) so they grow and touch across small gaps."""
    binary = _ensure_binary(edges)
    if dilate_size > 0:
        k = 2 * dilate_size + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.dilate(binary, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if min_area > 0:
        contours = [c for c in contours if cv2.contourArea(c) > min_area]
    return contours


# -----------------------------------------------------------------------------
# 4. Line kernel closing
# -----------------------------------------------------------------------------
def line_close_then_contours(
    edges: np.ndarray,
    line_length: int = 5,
    min_area: int = 0,
) -> List[np.ndarray]:
    """Close using a line kernel at 0°, 45°, 90°, 135°."""
    binary = _ensure_binary(edges)
    if line_length > 0:
        size = 2 * line_length + 1
        c = line_length
        for angle in [0, 45, 90, 135]:
            kernel = np.zeros((size, size), dtype=np.uint8)
            if angle == 0:
                kernel[c, :] = 1
            elif angle == 90:
                kernel[:, c] = 1
            elif angle == 45:
                for i in range(-line_length, line_length + 1):
                    r, col = c + i, c + i
                    if 0 <= r < size and 0 <= col < size:
                        kernel[r, col] = 1
            else:  # 135
                for i in range(-line_length, line_length + 1):
                    r, col = c + i, c - i
                    if 0 <= r < size and 0 <= col < size:
                        kernel[r, col] = 1
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if min_area > 0:
        contours = [c for c in contours if cv2.contourArea(c) > min_area]
    return contours


# -----------------------------------------------------------------------------
# 5. Endpoint connection utilities
# -----------------------------------------------------------------------------
def _is_endpoint(edges: np.ndarray, y: int, x: int) -> bool:
    """Pixel is an endpoint if it has exactly one 8-neighbor that is on."""
    h, w = edges.shape
    if edges[y, x] < 127:
        return False
    count = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and edges[ny, nx] > 127:
                count += 1
    return count == 1


def _get_endpoint_direction(binary: np.ndarray, y: int, x: int) -> Tuple[float, float]:
    """Direction 'out' of the edge at this endpoint."""
    h, w = binary.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] > 127:
                u = float(y - ny)
                v = float(x - nx)
                n = np.hypot(u, v)
                if n > 0:
                    return (u / n, v / n)
                return (0.0, 0.0)
    return (0.0, 0.0)


def _find_endpoints(edges: np.ndarray) -> List[Tuple[int, int]]:
    out = []
    h, w = edges.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if _is_endpoint(edges, y, x):
                out.append((y, x))
    return out


def _nearest_edge_point_in_other_component(
    binary: np.ndarray,
    labels: np.ndarray,
    y: int,
    x: int,
    my_id: int,
    max_gap: int,
) -> Tuple[int, int, float]:
    """Find nearest white pixel that belongs to a different component."""
    h, w = binary.shape
    best_d = float("inf")
    best_p = (-1, -1)
    r = max_gap
    for py in range(max(0, y - r), min(h, y + r + 1)):
        for px in range(max(0, x - r), min(w, x + r + 1)):
            if binary[py, px] < 127:
                continue
            if labels[py, px] == my_id:
                continue
            d = np.hypot(py - y, px - x)
            if d < best_d and d > 0.5:
                best_d = d
                best_p = (py, px)
    return (best_p[0], best_p[1], best_d)


def smart_connect_endpoints_bridged(
    edges: np.ndarray,
    max_gap: int = 12,
    line_thickness: int = 2,
    direction_weight: float = 2.0,
    extend_pixels: int = 1,
) -> np.ndarray:
    """Smarter endpoint connection with direction cues."""
    binary = _ensure_binary(edges).copy()
    h, w = binary.shape
    num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
    endpoints = _find_endpoints(binary)
    used = [False] * len(endpoints)

    def score_candidate(ey: int, ex: int, ty: int, tx: int, dir_out: Tuple[float, float]) -> float:
        d = np.hypot(ty - ey, tx - ex)
        if d <= 0:
            return float("inf")
        u = (ty - ey) / d
        v = (tx - ex) / d
        dot = dir_out[0] * u + dir_out[1] * v
        return d - direction_weight * max(0.0, dot)

    for i, (y1, x1) in enumerate(endpoints):
        if used[i]:
            continue
        id1 = labels[y1, x1]
        dir1 = _get_endpoint_direction(binary, y1, x1)

        best_target = None
        best_score = float("inf")

        for j, (y2, x2) in enumerate(endpoints):
            if i == j or used[j]:
                continue
            if labels[y2, x2] == id1:
                continue
            d = np.hypot(y2 - y1, x2 - x1)
            if d > max_gap:
                continue
            sc = score_candidate(y1, x1, y2, x2, dir1)
            if sc < best_score:
                best_score = sc
                best_target = ("endpoint", y2, x2, j)

        py, px, nd = _nearest_edge_point_in_other_component(binary, labels, y1, x1, id1, max_gap)
        if nd <= max_gap and nd < float("inf"):
            sc = score_candidate(y1, x1, py, px, dir1)
            if sc < best_score:
                best_score = sc
                best_target = ("edge", py, px, None)

        if best_target is None:
            continue
        kind, ty, tx, j_idx = best_target

        if extend_pixels > 0 and (dir1[0] != 0 or dir1[1] != 0):
            sy = int(round(y1 + extend_pixels * dir1[0]))
            sx = int(round(x1 + extend_pixels * dir1[1]))
            sy = max(0, min(h - 1, sy))
            sx = max(0, min(w - 1, sx))
        else:
            sy, sx = y1, x1
        cv2.line(binary, (sx, sy), (tx, ty), 255, line_thickness)
        used[i] = True
        if kind == "endpoint" and j_idx is not None:
            used[j_idx] = True
    return binary


def connect_endpoints_bridged(
    edges: np.ndarray,
    max_gap: int = 12,
    line_thickness: int = 2,
) -> np.ndarray:
    """Return binary image with endpoint-to-endpoint bridges drawn."""
    binary = _ensure_binary(edges).copy()
    endpoints = _find_endpoints(binary)
    used = [False] * len(endpoints)
    for i, (y1, x1) in enumerate(endpoints):
        if used[i]:
            continue
        best_j, best_d = -1, max_gap + 1
        for j, (y2, x2) in enumerate(endpoints):
            if i == j or used[j]:
                continue
            d = int(np.hypot(y2 - y1, x2 - x1))
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0:
            y2, x2 = endpoints[best_j]
            cv2.line(binary, (x1, y1), (x2, y2), 255, line_thickness)
            used[i] = used[best_j] = True
    return binary


def connect_endpoints_then_contours(
    edges: np.ndarray,
    max_gap: int = 8,
    min_area: int = 0,
    line_thickness: int = 2,
) -> List[np.ndarray]:
    """Find endpoints, connect pairs, then find contours."""
    binary = connect_endpoints_bridged(edges, max_gap, line_thickness)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if min_area > 0:
        contours = [c for c in contours if cv2.contourArea(c) > min_area]
    return contours


def smart_connect_endpoints_then_contours(
    edges: np.ndarray,
    max_gap: int = 12,
    min_area: int = 0,
    line_thickness: int = 2,
    direction_weight: float = 2.0,
    extend_pixels: int = 1,
) -> List[np.ndarray]:
    """Smarter endpoint connection then find contours."""
    binary = smart_connect_endpoints_bridged(
        edges, max_gap, line_thickness, direction_weight, extend_pixels
    )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if min_area > 0:
        contours = [c for c in contours if cv2.contourArea(c) > min_area]
    return contours


# -----------------------------------------------------------------------------
# 5a. Fast gap closing (spatial grid - O(edge_pixels) vs O(search_area))
# -----------------------------------------------------------------------------
def _build_edge_grid(
    binary: np.ndarray,
    labels: np.ndarray,
    cell_size: int,
) -> dict:
    """Grid of (cy, cx) -> [(y, x, component_id), ...] for edge pixels."""
    h, w = binary.shape
    grid: dict = {}
    for y in range(h):
        for x in range(w):
            if binary[y, x] < 127:
                continue
            cy, cx = y // cell_size, x // cell_size
            if (cy, cx) not in grid:
                grid[(cy, cx)] = []
            grid[(cy, cx)].append((y, x, int(labels[y, x])))
    return grid


def _fast_nearest_in_other_component(
    y1: int,
    x1: int,
    my_id: int,
    max_gap: float,
    grid: dict,
    cell_size: int,
) -> Tuple[int, int, float]:
    """Find nearest edge pixel in different component using spatial grid."""
    cy1, cx1 = y1 // cell_size, x1 // cell_size
    r_cells = max(1, int(np.ceil(max_gap / cell_size)))
    best_d = float("inf")
    best_p = (-1, -1)
    for dcy in range(-r_cells, r_cells + 1):
        for dcx in range(-r_cells, r_cells + 1):
            key = (cy1 + dcy, cx1 + dcx)
            if key not in grid:
                continue
            for py, px, comp_id in grid[key]:
                if comp_id == my_id or comp_id == 0:
                    continue
                d = np.hypot(py - y1, px - x1)
                if 0.5 < d <= max_gap and d < best_d:
                    best_d = d
                    best_p = (py, px)
    return best_p[0], best_p[1], best_d


def fast_close_gaps(
    binary: np.ndarray,
    max_gap: int,
    line_thickness: int = 2,
    max_iterations: int = 200,
) -> np.ndarray:
    """
    Fast endpoint connection using spatial grid. Same behavior as
    smart_connect + force_close, but O(edge_pixels) instead of O(search_area²).
    """
    out = _ensure_binary(binary).copy()
    h, w = out.shape
    cell_size = max(10, max_gap // 4)

    for _ in range(max_iterations):
        endpoints = _find_endpoints(out)
        if not endpoints:
            break
        num_labels, labels = cv2.connectedComponents(out, connectivity=8)
        grid = _build_edge_grid(out, labels, cell_size)
        used = {ep: False for ep in endpoints}
        ep_set = set(endpoints)
        did_connect = False

        for y1, x1 in endpoints:
            if used[(y1, x1)]:
                continue
            my_id = labels[y1, x1]
            if my_id == 0:
                continue
            py, px, nd = _fast_nearest_in_other_component(
                y1, x1, my_id, float(max_gap), grid, cell_size
            )
            if nd < float("inf"):
                cv2.line(out, (x1, y1), (px, py), 255, line_thickness)
                used[(y1, x1)] = True
                if (py, px) in ep_set:
                    used[(py, px)] = True
                did_connect = True
        if not did_connect:
            break
    return out


# -----------------------------------------------------------------------------
# 5b. Force-close every open chain (guarantee closed contours)
# -----------------------------------------------------------------------------
def force_close_open_chains(
    binary: np.ndarray,
    line_thickness: int = 2,
    max_iterations: int = 100,
    max_gap: int = None,
) -> np.ndarray:
    """
    Connect ALL endpoints iteratively until none remain.
    """
    out = _ensure_binary(binary).copy()
    h, w = out.shape
    if max_gap is None:
        max_gap = int(np.hypot(h, w))  # Image diagonal
    
    for iteration in range(max_iterations):
        endpoints = _find_endpoints(out)
        if not endpoints:
            break  # No endpoints left - all closed!
        
        num_labels, labels = cv2.connectedComponents(out, connectivity=8)
        used = [False] * len(endpoints)
        did_connect = False
        
        for i, (y1, x1) in enumerate(endpoints):
            if used[i]:
                continue
            
            if y1 >= h or x1 >= w:
                continue
            
            # Find nearest edge point (endpoint or any edge pixel)
            best_target = None
            best_dist = float("inf")
            
            # Search in a square around the endpoint
            search_radius = min(max_gap, max(h, w))
            y_min = max(0, y1 - search_radius)
            y_max = min(h, y1 + search_radius + 1)
            x_min = max(0, x1 - search_radius)
            x_max = min(w, x1 + search_radius + 1)
            
            for py in range(y_min, y_max):
                for px in range(x_min, x_max):
                    if out[py, px] < 127:
                        continue
                    # Don't connect to self
                    if py == y1 and px == x1:
                        continue
                    d = np.hypot(py - y1, px - x1)
                    if d <= max_gap and d < best_dist and d > 0.5:
                        # Check if this is another endpoint (prefer endpoint connections)
                        is_endpoint = False
                        j_idx = None
                        for j, (ey, ex) in enumerate(endpoints):
                            if ey == py and ex == px and not used[j]:
                                is_endpoint = True
                                j_idx = j
                                break
                        
                        best_dist = d
                        best_target = (py, px, is_endpoint, j_idx)
            
            # Connect if we found a target
            if best_target is not None:
                ty, tx, is_ep, j_idx = best_target
                cv2.line(out, (x1, y1), (tx, ty), 255, line_thickness)
                used[i] = True
                if is_ep and j_idx is not None:
                    used[j_idx] = True
                did_connect = True
        
        if not did_connect:
            # No connections made - might be stuck, try connecting remaining endpoints
            remaining = [ep for i, ep in enumerate(endpoints) if not used[i]]
            if remaining:
                for y1, x1 in remaining[:10]:  # Limit to avoid infinite loops
                    # Find ANY edge point (even far away)
                    best_d = float("inf")
                    best_p = None
                    for py in range(h):
                        for px in range(w):
                            if out[py, px] < 127:
                                continue
                            d = np.hypot(py - y1, px - x1)
                            if d < best_d and d > 0.5:
                                best_d = d
                                best_p = (py, px)
                    if best_p:
                        cv2.line(out, (x1, y1), (best_p[1], best_p[0]), 255, line_thickness)
                        did_connect = True
            if not did_connect:
                break  # Truly stuck
    
    return out


def contours_guaranteed_closed(
    edges: np.ndarray,
    bridge_first: bool = True,
    bridge_max_gap: int = 12,
    min_area: int = 0,
    line_thickness: int = 2,
    force_close_max_gap: int = None,
) -> List[np.ndarray]:
    """
    All edges end up as closed contours.
    """
    binary = _ensure_binary(edges).copy()
    if bridge_first:
        binary = smart_connect_endpoints_bridged(
            binary, bridge_max_gap, line_thickness, direction_weight=2.0, extend_pixels=1
        )
    binary = force_close_open_chains(binary, line_thickness=line_thickness, max_gap=force_close_max_gap)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if min_area > 0:
        contours = [c for c in contours if cv2.contourArea(c) > min_area]
    # Explicitly close each contour by ensuring first point == last point
    closed_contours = []
    for c in contours:
        if len(c) > 0:
            first = c[0][0]
            last = c[-1][0]
            if not np.array_equal(first, last):
                closed = np.vstack([c, c[0].reshape(1, 1, 2)])
                closed_contours.append(closed)
            else:
                closed_contours.append(c)
    return closed_contours


# -----------------------------------------------------------------------------
# 6. Flood-fill "holes" (enclosed regions) after closing
# -----------------------------------------------------------------------------
def close_then_region_contours(
    edges: np.ndarray,
    close_kernel: int = 9,
    min_area: int = 20,
) -> List[np.ndarray]:
    """Close edges, then treat "holes" as enclosed shapes."""
    binary = _ensure_binary(edges)
    if close_kernel > 0:
        k = (close_kernel, close_kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    inv = 255 - binary
    h, w = inv.shape
    mask = inv.copy()
    for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if mask[seed[1], seed[0]] == 255:
            cv2.floodFill(mask, None, seed, 0)
    for x in [0, w // 2, w - 1]:
        if mask[0, x] == 255:
            cv2.floodFill(mask, None, (x, 0), 0)
        if mask[h - 1, x] == 255:
            cv2.floodFill(mask, None, (x, h - 1), 0)
    for y in [0, h // 2, h - 1]:
        if mask[y, 0] == 255:
            cv2.floodFill(mask, None, (0, y), 0)
        if mask[y, w - 1] == 255:
            cv2.floodFill(mask, None, (w - 1, y), 0)
    holes = (mask == 255).astype(np.uint8) * 255
    if holes.max() == 0:
        return []
    contours, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if min_area > 0:
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    return contours


# -----------------------------------------------------------------------------
# 7. Chained methods
# -----------------------------------------------------------------------------
def connect_then_close_contours(
    edges: np.ndarray,
    connect_gap: int = 14,
    close_kernel: int = 5,
    kernel_shape: str = "ellipse",
    min_area: int = 0,
) -> List[np.ndarray]:
    """First bridge endpoint pairs, then morphological close."""
    binary = connect_endpoints_bridged(edges, connect_gap, line_thickness=2)
    return close_then_contours(binary, close_kernel, kernel_shape, min_area)


def close_then_connect_contours(
    edges: np.ndarray,
    close_kernel: int = 9,
    connect_gap: int = 10,
    min_area: int = 0,
) -> List[np.ndarray]:
    """First close edges, then connect any remaining endpoints."""
    binary = _ensure_binary(edges)
    if close_kernel > 0:
        k = (close_kernel, close_kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = connect_endpoints_bridged(binary, connect_gap, line_thickness=2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if min_area > 0:
        contours = [c for c in contours if cv2.contourArea(c) > min_area]
    return contours


# -----------------------------------------------------------------------------
# Registry for easy iteration
# -----------------------------------------------------------------------------
def line_close_5(edges: np.ndarray) -> List[np.ndarray]:
    return line_close_then_contours(edges, 5, 0)


ALL_METHODS = [
    ("close_rect_5", lambda e: close_then_contours(e, 5, "rect", 0)),
    ("close_rect_7", lambda e: close_then_contours(e, 7, "rect", 0)),
    ("close_rect_11", lambda e: close_then_contours(e, 11, "rect", 0)),
    ("close_ellipse_5", lambda e: close_then_contours(e, 5, "ellipse", 0)),
    ("close_ellipse_9", lambda e: close_then_contours(e, 9, "ellipse", 0)),
    ("close_ellipse_15", lambda e: close_then_contours(e, 15, "ellipse", 0)),
    ("iterative_3x3_x3", lambda e: iterative_close_then_contours(e, 3, 3, 0)),
    ("iterative_5x5_x2", lambda e: iterative_close_then_contours(e, 2, 5, 0)),
    ("dilate_2", lambda e: dilate_then_contours(e, 2, 0)),
    ("dilate_3", lambda e: dilate_then_contours(e, 3, 0)),
    ("dilate_4", lambda e: dilate_then_contours(e, 4, 0)),
    ("line_close_5", line_close_5),
    ("connect_ep_6", lambda e: connect_endpoints_then_contours(e, 6, 0)),
    ("connect_ep_10", lambda e: connect_endpoints_then_contours(e, 10, 0)),
    ("connect_ep_15", lambda e: connect_endpoints_then_contours(e, 15, 0)),
    ("smart_connect_12", lambda e: smart_connect_endpoints_then_contours(e, 12, 0, 2, 2.0, 1)),
    ("smart_connect_15", lambda e: smart_connect_endpoints_then_contours(e, 15, 0, 2, 2.0, 2)),
    ("connect_then_close", lambda e: connect_then_close_contours(e, 14, 5, "ellipse", 0)),
    ("close_then_connect", lambda e: close_then_connect_contours(e, 9, 10, 0)),
    ("region_close_9", lambda e: close_then_region_contours(e, 9, 20)),
    ("region_close_15", lambda e: close_then_region_contours(e, 15, 20)),
    ("guaranteed_closed", lambda e: contours_guaranteed_closed(e, True, 12, 0, 2, None)),
]
