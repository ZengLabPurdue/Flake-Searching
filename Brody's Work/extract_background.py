#!/usr/bin/env python3
"""
Extract background from image by:
1. Very sensitive edge detection
2. Convert edges to contours
3. Mask out shapes (keep only background)

Usage:
    python extract_background.py <image_path> [output_path]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from canny_filter_tuner import compute_filtered_canny, edges_to_region_contours
from edge_to_contour_methods import contours_guaranteed_closed, close_then_region_contours
from preprocess import clahe_then_nlmeans

PAD = 15  # Padding for "pad" method so contours can close at image border
FRAME_MARGIN = 5  # Margin so "outline" is a little bigger than image; open chains that hit frame in 2 spots get closed

# 8-neighbor offsets (col, row)
_NEIGHBORS_8 = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]


def _trace_edges_to_chains(edges: np.ndarray) -> list:
    """
    Trace binary edge image into ordered chains (polylines).
    Returns list of (points, is_closed) where points is list of (x, y).
    """
    h, w = edges.shape[:2]
    binary = (edges > 127).astype(np.uint8)
    if binary.sum() == 0:
        return []

    # Label connected components (8-connectivity)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    chains = []

    for label in range(1, num_labels):
        ys, xs = np.where(labels == label)
        points_set = set(zip(xs.tolist(), ys.tolist()))
        if len(points_set) < 2:
            continue

        # Build adjacency: for each point, list of 8-neighbors in the set
        def neighbors(px, py):
            out = []
            for dx, dy in _NEIGHBORS_8:
                nx, ny = px + dx, py + dy
                if 0 <= nx < w and 0 <= ny < h and (nx, ny) in points_set:
                    out.append((nx, ny))
            return out

        # Start from an endpoint (degree 1) if any, else any point
        start = None
        for (px, py) in points_set:
            if len(neighbors(px, py)) <= 1:
                start = (px, py)
                break
        if start is None:
            start = next(iter(points_set))

        # Walk the path
        path = [start]
        remaining = points_set - {start}
        current = start
        while remaining:
            next_pts = [n for n in neighbors(current[0], current[1]) if n in remaining]
            if not next_pts:
                break
            # Prefer continuing in same direction if possible
            if len(path) >= 2:
                prev = path[-2]
                dx0, dy0 = current[0] - prev[0], current[1] - prev[1]
                same_dir = [(n[0], n[1]) for n in next_pts if (n[0] - current[0], n[1] - current[1]) == (dx0, dy0)]
                if same_dir:
                    next_pts = same_dir
            nxt = next_pts[0]
            path.append(nxt)
            remaining.discard(nxt)
            current = nxt

        if not path:
            continue
        # Check if closed: first and last are 8-neighbors
        first, last = path[0], path[-1]
        is_closed = (abs(first[0] - last[0]) <= 1 and abs(first[1] - last[1]) <= 1) and len(path) > 2
        chains.append((path, is_closed))

    return chains


def _on_frame(x: int, y: int, w: int, h: int, margin: int) -> bool:
    """True if (x,y) is on the outer frame (boundary of the padded image)."""
    return x <= margin or x >= w - 1 - margin or y <= margin or y >= h - 1 - margin


def _frame_segment(p1: tuple, p2: tuple, w: int, h: int) -> list:
    """
    Return list of (x,y) points along the frame rectangle (0,0)-(w-1,h-1)
    from p1 to p2 along the shorter arc. p1 and p2 must lie on the frame.
    """
    # Frame corners and edges (clockwise from top-left)
    # Top: (0,0) -> (w-1, 0); Right: (w-1,0) -> (w-1,h-1); Bottom: (w-1,h-1) -> (0,h-1); Left: (0,h-1) -> (0,0)
    def param(px, py):
        if py == 0 and px < w - 1:
            return px
        if px == w - 1 and py < h - 1:
            return (w - 1) + py
        if py == h - 1 and px > 0:
            return (w - 1) + (h - 1) + (w - 1 - px)
        return (w - 1) + (h - 1) + (w - 1) + (h - 1 - py)

    def point_at(t):
        total = 2 * (w + h - 2)
        t = t % total
        if t < w - 1:
            return (t, 0)
        t -= (w - 1)
        if t < h - 1:
            return (w - 1, t)
        t -= (h - 1)
        if t < w - 1:
            return (w - 1 - t, h - 1)
        t -= (w - 1)
        return (0, h - 1 - t)

    t1, t2 = param(p1[0], p1[1]), param(p2[0], p2[1])
    total = 2 * (w + h - 2)
    dt = (t2 - t1) % total
    if dt > total // 2:
        dt -= total
    out = []
    step = 1 if dt >= 0 else -1
    t = int(t1)
    end = int(t2)
    while True:
        out.append(point_at(t))
        if t == end:
            break
        t = (t + step) % total
    return out


def _edges_to_contours_with_frame(edges: np.ndarray, margin: int, w: int, h: int):
    """
    Get closed contours from edges. Use a frame (outline) that is the boundary of the
    (w+2*margin)x(h+2*margin) image. Open chains that touch the frame at both ends
    are closed by adding the frame segment between the two connection points.
    Returns list of contours (each Nx1x2, int32). Excludes the frame contour itself.
    """
    chains = _trace_edges_to_chains(edges)
    W, H = w + 2 * margin, h + 2 * margin  # padded size
    frame_rect = (0, 0, W - 1, H - 1)
    contours = []

    for path, is_closed in chains:
        if is_closed:
            if len(path) < 3:
                continue
            cnt = np.array(path, dtype=np.int32).reshape(-1, 1, 2)
            contours.append(cnt)
            continue

        # Open chain: check if both endpoints on frame
        if len(path) < 2:
            continue
        p1, p2 = path[0], path[-1]
        if not (_on_frame(p1[0], p1[1], W, H, 0) and _on_frame(p2[0], p2[1], W, H, 0)):
            continue
        # Close with frame segment (shorter arc from p2 back to p1)
        seg = _frame_segment(p2, p1, W, H)
        if not seg:
            continue
        # path is p1 -> ... -> p2; seg is p2 -> ... -> p1; full = path + seg (skip duplicate p2/p1)
        full = path + seg[1:-1]
        if len(full) < 3:
            continue
        cnt = np.array(full, dtype=np.int32).reshape(-1, 1, 2)
        contours.append(cnt)

    # Exclude the frame contour (the big rectangle)
    frame_area = (W - 1) * (H - 1)
    contours = [c for c in contours if cv2.contourArea(c) < 0.99 * frame_area]
    return contours


def _get_frame_closed_contours_only(edges: np.ndarray, margin: int, w: int, h: int) -> list:
    """Return only contours that were open chains closed by the frame (not already-closed chains)."""
    chains = _trace_edges_to_chains(edges)
    W, H = w + 2 * margin, h + 2 * margin
    contours = []
    for path, is_closed in chains:
        if is_closed:
            continue
        if len(path) < 2:
            continue
        p1, p2 = path[0], path[-1]
        if not (_on_frame(p1[0], p1[1], W, H, 0) and _on_frame(p2[0], p2[1], W, H, 0)):
            continue
        seg = _frame_segment(p2, p1, W, H)
        if not seg:
            continue
        full = path + seg[1:-1]
        if len(full) < 3:
            continue
        cnt = np.array(full, dtype=np.int32).reshape(-1, 1, 2)
        contours.append(cnt)
    return contours


def _contour_touches_frame(contour: np.ndarray, W: int, H: int) -> bool:
    """True if any point of the contour lies on the outer frame (x=0, x=W-1, y=0, y=H-1)."""
    for i in range(len(contour)):
        x, y = int(contour[i, 0, 0]), int(contour[i, 0, 1])
        if x <= 0 or x >= W - 1 or y <= 0 or y >= H - 1:
            return True
    return False


def bin_image(arr: np.ndarray, bin_factor: int = 4) -> np.ndarray:
    """
    Bin a 2D array by bin_factor (bin_factor x bin_factor -> 1 pixel) using mean.
    If dimensions are not divisible by bin_factor, trim to nearest multiple.
    """
    h, w = arr.shape[:2]
    # Trim to dimensions divisible by bin_factor
    h_trim = h - (h % bin_factor)
    w_trim = w - (w % bin_factor)
    arr = arr[:h_trim, :w_trim]

    if arr.ndim == 2:
        # Shape (H, W) -> (H/bin_factor, bin_factor, W/bin_factor, bin_factor) -> mean over (1, 3)
        binned = arr.reshape(h_trim // bin_factor, bin_factor, w_trim // bin_factor, bin_factor).mean(axis=(1, 3))
        return binned
    else:
        # Multi-channel (e.g. RGB): bin each channel
        binned = np.stack(
            [bin_image(arr[:, :, c], bin_factor) for c in range(arr.shape[2])], axis=-1
        )
        return binned


def extract_background(
    image_path: Path,
    output_path: Path = None,
    bin_factor: int = 4,
    canny_low: int = 5,
    canny_high: int = 25,
    min_area: int = 5,
    bridge_max_gap: int = 45,
    force_close_max_gap: int = 55,
    edge_method: str = "default",
):
    """
    Extract background from image by masking out detected shapes.
    edge_method: "default" (contours from edges), "pad" (pad image so edge-touching contours close),
                 "region" (flood-fill from border to get enclosed region contours).
    """
    print(f"Loading image: {image_path}")
    
    # Load image
    original_img = np.array(Image.open(image_path))
    if original_img.ndim == 2:
        original_img = np.stack([original_img, original_img, original_img], axis=-1)
    if original_img.shape[2] == 4:
        original_img = original_img[:, :, :3]
    
    orig_h, orig_w = original_img.shape[:2]
    print(f"  Original size: {orig_w}x{orig_h}")
    
    # Step 1: Optionally bin the image to reduce noise
    if bin_factor > 1:
        print(f"\n1. Binning by {bin_factor}x{bin_factor}...")
        binned_img = bin_image(original_img, bin_factor)
        bin_h, bin_w = binned_img.shape[:2]
        print(f"  Binned size: {bin_w}x{bin_h}")
        
        # Ensure uint8
        if binned_img.dtype != np.uint8:
            binned_img = np.clip(binned_img, 0, 255).astype(np.uint8)
        
        work_img = binned_img
        scale_x = orig_w / bin_w
        scale_y = orig_h / bin_h
    else:
        work_img = original_img
        scale_x = 1.0
        scale_y = 1.0
    
    # Step 2: Preprocess with CLAHE + NL-means (like filtered sensitive)
    print(f"\n2. Preprocessing (CLAHE + NL-means)...")
    preprocessed = clahe_then_nlmeans(work_img)
    work_img = preprocessed.astype(np.uint8)
    
    # Step 2b: Optional padding (default = frame outline; pad = constant border)
    pad_offset_x = 0
    pad_offset_y = 0
    pre_pad_w, pre_pad_h = work_img.shape[1], work_img.shape[0]
    if edge_method == "default":
        # Frame method: pad a little so "outline" is bigger than image; open chains that hit frame in 2 spots get closed
        work_img = cv2.copyMakeBorder(
            work_img, FRAME_MARGIN, FRAME_MARGIN, FRAME_MARGIN, FRAME_MARGIN,
            borderType=cv2.BORDER_REPLICATE,
        )
        pad_offset_x = FRAME_MARGIN
        pad_offset_y = FRAME_MARGIN
        print(f"  Frame outline: padded by {FRAME_MARGIN} px (outline a little bigger than image)")
    elif edge_method == "pad":
        work_h, work_w = work_img.shape[:2]
        work_img = cv2.copyMakeBorder(
            work_img, PAD, PAD, PAD, PAD,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )
        pad_offset_x = PAD
        pad_offset_y = PAD
        pre_pad_w, pre_pad_h = work_w, work_h
        print(f"  Padded by {PAD} px on all sides ({work_w}x{work_h} -> {work_img.shape[1]}x{work_img.shape[0]})")

    # Step 3: Very sensitive edge detection
    print(f"\n3. Edge detection (low={canny_low}, high={canny_high}, min_area={min_area})...")
    edges, n_white, n_removed = compute_filtered_canny(
        work_img, blur_sigma=0.6, canny_low=canny_low, canny_high=canny_high, min_area=min_area
    )
    print(f"  Edge pixels: {n_white:,} (removed {n_removed:,} from small components)")
    
    # For default: use closed edges only for internal contours (bridges blurry gaps); use original edges for wall/frame closing
    edges_for_internal = edges
    if edge_method == "default":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_for_internal = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Step 4: Get contours (method-dependent)
    if edge_method == "region":
        # Flood-fill from border; contours = enclosed regions (holes)
        print(f"\n4. Region contours (close edges, flood-fill from border, contours = enclosed regions)...")
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        contours = edges_to_region_contours(edges_closed, min_area=min_area, close_kernel_size=0, dilate_before_close=0)
        if not contours:
            contours = close_then_region_contours(edges, close_kernel=9, min_area=min_area)
        print(f"  Found {len(contours)} region contours")
    elif edge_method == "default":
        # Hybrid: internal from closed edges (bridges blurry gaps); wall shape from original edges + frame
        print(f"\n4. Contours (internal from findContours + wall shape from frame closing)...")
        all_closed = contours_guaranteed_closed(
            edges_for_internal,
            bridge_first=True,
            bridge_max_gap=bridge_max_gap,
            min_area=0,
            line_thickness=2,
            force_close_max_gap=force_close_max_gap,
        )
        H_edges, W_edges = edges.shape[:2]
        internal = [c for c in all_closed if not _contour_touches_frame(c, W_edges, H_edges)]
        frame_closed = _get_frame_closed_contours_only(edges, FRAME_MARGIN, pre_pad_w, pre_pad_h)
        contours = internal + frame_closed
        print(f"  Found {len(contours)} contours ({len(internal)} internal + {len(frame_closed)} closed at frame)")
    else:
        # Pad method: contours from edge boundaries (guaranteed closed), then exclude frame
        print(f"\n4. Converting edges to contours (bridge_max_gap={bridge_max_gap}, force_close_max_gap={force_close_max_gap})...")
        contours = contours_guaranteed_closed(
            edges,
            bridge_first=True,
            bridge_max_gap=bridge_max_gap,
            min_area=0,
            line_thickness=2,
            force_close_max_gap=force_close_max_gap
        )
        h_edges, w_edges = edges.shape[:2]
        full_pad_area = w_edges * h_edges
        contours = [c for c in contours if cv2.contourArea(c) < 0.99 * full_pad_area]
        print(f"  Found {len(contours)} contours (excluded pad-boundary frame)")
    
    # Step 5: Scale contours to original image size (subtract pad if used)
    print(f"\n5. Scaling contours to original size (scale: {scale_x:.2f}x{scale_y:.2f})...")
    scaled_contours = []
    for c in contours:
        scaled = c.copy().astype(np.float32)
        scaled[:, 0, 0] = (scaled[:, 0, 0] - pad_offset_x) * scale_x  # x
        scaled[:, 0, 1] = (scaled[:, 0, 1] - pad_offset_y) * scale_y  # y
        scaled = scaled.astype(np.int32)
        scaled_contours.append(scaled)
    
    # Step 6: Create mask from contours (shapes = white, background = black)
    print(f"\n6. Creating mask from contours...")
    if edge_method == "pad":
        # Pad method: the remaining contour(s) enclose the BACKGROUND (hole touching the frame).
        # Fill contour interior with black (background), leave exterior white (shapes).
        mask = np.ones((orig_h, orig_w), dtype=np.uint8) * 255
        cv2.drawContours(mask, scaled_contours, -1, 0, -1)  # Fill contour interior with black
    else:
        # Default/region: contours enclose shapes. Fill contours with white (shapes).
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        cv2.drawContours(mask, scaled_contours, -1, 255, -1)  # Fill contours
    
    # Fill any black outline regions (find black contours and fill their interiors)
    print(f"   Filling black outline regions...")
    # Find black regions (background) that form closed contours
    black_mask = (mask == 0).astype(np.uint8) * 255
    
    # Use RETR_TREE to find contour hierarchy - black contours inside white contours
    black_contours, hierarchy = cv2.findContours(black_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    n_filled = 0
    if hierarchy is not None:
        for i, contour in enumerate(black_contours):
            area = cv2.contourArea(contour)
            if area < 10:  # Skip tiny noise
                continue
            
            # Check if contour touches border
            touches_border = False
            for point in contour:
                x, y = point[0][0], point[0][1]  # OpenCV contour: (x=col, y=row)
                if x == 0 or x == orig_w - 1 or y == 0 or y == orig_h - 1:
                    touches_border = True
                    break
            
            # Fill black contours that:
            # 1. Don't touch border (enclosed inside white shapes), OR
            # 2. Touch border but have a parent contour (nested inside white), OR
            # 3. Touch border but are small (edge noise / small shape interior)
            has_parent = hierarchy[0][i][3] >= 0  # Parent contour index
            image_area = orig_w * orig_h
            area_threshold = image_area * 0.02  # 2% - small holes / edge regions

            should_fill = (not touches_border) or (
                touches_border and (has_parent or area < area_threshold)
            )
            
            if should_fill:
                cv2.drawContours(mask, [contour], -1, 255, -1)  # Fill with white (shape)
                n_filled += int(area)
    
    if n_filled > 0:
        print(f"   Filled {n_filled:,} pixels in black outline regions")
    
    # Step 7: Invert mask (background = white, shapes = black)
    mask_inv = 255 - mask
    
    # Step 7b: Fill white regions in mask_background that are surrounded by black
    print(f"\n7b. Checking for white regions surrounded by black...")
    white_mask = (mask_inv == 255).astype(np.uint8) * 255
    
    # Find white contours (background regions)
    white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"   Found {len(white_contours)} white regions to check")
    
    n_converted = 0
    for contour in white_contours:
        area = cv2.contourArea(contour)
        if area < 50:  # Skip very small regions
            continue
        
        # Never convert white regions that touch the image border - that's the real background
        touches_border = False
        for pt in contour:
            px, py = pt[0][0], pt[0][1]
            if px <= 0 or px >= orig_w - 1 or py <= 0 or py >= orig_h - 1:
                touches_border = True
                break
        if touches_border:
            continue
        
        # Create a mask for this specific contour
        contour_mask_full = np.zeros((orig_h, orig_w), dtype=np.uint8)
        cv2.drawContours(contour_mask_full, [contour], -1, 255, -1)
        
        # Dilate the contour to get pixels immediately around it
        kernel_size = max(10, int(min(cv2.boundingRect(contour)[2], cv2.boundingRect(contour)[3]) * 0.2))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated_contour = cv2.dilate(contour_mask_full, kernel, iterations=1)
        
        # Get the border pixels (dilated - original)
        border_pixels = dilated_contour - contour_mask_full
        
        # Count black pixels in the border
        border_black = np.sum((mask_inv == 0) & (border_pixels > 0))
        border_white = np.sum((mask_inv == 255) & (border_pixels > 0))
        border_total = border_black + border_white
        
        # Also check a larger surrounding area
        x, y, w, h = cv2.boundingRect(contour)
        expand_pixels = int(max(w, h) * 0.5)
        x_start = max(0, x - expand_pixels)
        y_start = max(0, y - expand_pixels)
        x_end = min(orig_w, x + w + expand_pixels)
        y_end = min(orig_h, y + h + expand_pixels)
        
        region = mask_inv[y_start:y_end, x_start:x_end]
        region_black = np.sum(region == 0)
        region_white = np.sum(region == 255)
        region_total = region_black + region_white
        
        # Convert if:
        # 1. Border has more black than white (immediate surrounding), OR
        # 2. Expanded region has more black than white, OR
        # 3. Border has at least 30% black pixels
        should_convert = False
        
        if border_total > 10:
            if border_black > border_white or border_black / border_total >= 0.3:
                should_convert = True
        
        if region_total > 0 and region_black > region_white:
            should_convert = True
        
        if should_convert:
            # Fill this white region with black (treat as shape, not background)
            cv2.drawContours(mask_inv, [contour], -1, 0, -1)
            n_converted += int(area)
            print(f"   Converted region: area={int(area)}, border_black={border_black}/{border_total} ({border_black/border_total*100:.1f}%), region_black={region_black}/{region_total} ({region_black/region_total*100:.1f}%)")
    
    if n_converted > 0:
        print(f"   Total converted: {n_converted:,} pixels from background to shape (surrounded by black)")
    else:
        print(f"   No white regions found that are sufficiently surrounded by black")
    
    # Step 8: Apply mask to original image (keep only background)
    print(f"\n7. Applying mask to extract background...")
    if original_img.ndim == 3:
        # For RGB images, apply mask to each channel
        background = original_img.copy()
        for c in range(3):
            background[:, :, c] = np.where(mask_inv > 127, original_img[:, :, c], 0)
    else:
        background = np.where(mask_inv > 127, original_img, 0)
    
    # Ensure uint8
    background = np.clip(background, 0, 255).astype(np.uint8)
    
    # Step 9: Save output — all outputs go in one directory (e.g. x20_images/<stem>_background/)
    if output_path is None:
        output_dir = image_path.parent / f"{image_path.stem}_background"
    else:
        output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    background_path = output_dir / "background.png"
    Image.fromarray(background).save(background_path)
    print(f"\n✓ Saved background image: {background_path}")

    # Save intermediate outputs in the same directory
    if bin_factor > 1:
        edges_scaled = cv2.resize(edges, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    else:
        edges_scaled = edges
    Image.fromarray(edges_scaled).save(output_dir / "edges.png")
    Image.fromarray(mask).save(output_dir / "mask_shapes.png")
    Image.fromarray(mask_inv).save(output_dir / "mask_background.png")
    overlay = original_img.copy()
    cv2.drawContours(overlay, scaled_contours, -1, (0, 255, 0), 2)
    Image.fromarray(overlay).save(output_dir / "contours_overlay.png")

    print(f"  All outputs saved to: {output_dir}/")
    
    return background, mask_inv, scaled_contours


def main():
    parser = argparse.ArgumentParser(
        description="Extract background from image using sensitive edge detection and contour masking."
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to input image file"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory for background.png and debug images (default: <image_dir>/<image_stem>_background/)"
    )
    parser.add_argument(
        "--bin",
        type=int,
        default=4,
        help="Binning factor to reduce noise (default: 4, use 0 for no binning)"
    )
    parser.add_argument(
        "--canny-low",
        type=int,
        default=5,
        help="Canny low threshold (default: 5)"
    )
    parser.add_argument(
        "--canny-high",
        type=int,
        default=25,
        help="Canny high threshold (default: 25)"
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=5,
        help="Minimum area for edge components in pixels (default: 5)"
    )
    parser.add_argument(
        "--bridge-gap",
        type=int,
        default=45,
        help="Maximum gap to bridge when connecting endpoints (default: 45)"
    )
    parser.add_argument(
        "--close-gap",
        type=int,
        default=55,
        help="Maximum gap when force-closing contours (default: 55)"
    )
    parser.add_argument(
        "--method",
        choices=["default", "pad", "region"],
        default="default",
        help="Edge-to-contour method: default (contours from edges), pad (pad image so edge-touching contours close), region (flood-fill from border for enclosed regions)"
    )
    
    args = parser.parse_args()
    
    if not args.image.exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    extract_background(
        args.image,
        args.output,
        bin_factor=args.bin,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        min_area=args.min_area,
        bridge_max_gap=args.bridge_gap,
        force_close_max_gap=args.close_gap,
        edge_method=args.method,
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python extract_background.py <image_path> [options]")
        print("\nExample:")
        print("  python extract_background.py images/x20_images/RGr_001_A_20X_A_C.png")
        print("  python extract_background.py image.png --bin 4 --canny-low 1 --canny-high 10")
        sys.exit(1)
    
    main()
