"""Build aligned RGB-and-mask images for binary region review and training."""

from pathlib import Path

import cv2
import numpy as np


REGION_SAMPLE_SIZE = 512
NEW_FLAKE_FILTER_DIR = (
    Path(__file__).resolve().parents[2]
    / "Labeled Data"
    / "New Flake Filter"
)


def build_region_training_image(
    image_rgb,
    detection,
    output_size=REGION_SAMPLE_SIZE,
):
    """Return one RGB-left/mask-right training image for a matched region."""
    if (
        not isinstance(image_rgb, np.ndarray)
        or image_rgb.ndim != 3
        or image_rgb.shape[2] != 3
    ):
        raise ValueError("The source image must be an RGB image.")
    bounding_box = detection.get("bounding_box")
    instance_mask = detection.get("instance_mask")
    if not bounding_box or not isinstance(instance_mask, np.ndarray):
        return None

    box_x = int(bounding_box["x"])
    box_y = int(bounding_box["y"])
    box_width = int(bounding_box["width"])
    box_height = int(bounding_box["height"])
    if (
        box_width < 1
        or box_height < 1
        or instance_mask.shape != (box_height, box_width)
        or not np.any(instance_mask > 0)
    ):
        return None

    # Normal regions retain a native 512-pixel field of view. Exceptionally
    # large regions use a larger source square so the complete mask survives,
    # then both halves are resized together to the standard output dimensions.
    source_size = max(
        int(output_size),
        int(np.ceil(max(box_width, box_height) * 1.5)),
    )
    center_x = box_x + box_width / 2
    center_y = box_y + box_height / 2
    crop_x = int(np.floor(center_x - source_size / 2))
    crop_y = int(np.floor(center_y - source_size / 2))

    rgb_crop = np.zeros((source_size, source_size, 3), dtype=np.uint8)
    mask_crop = np.zeros((source_size, source_size), dtype=np.uint8)

    image_height, image_width = image_rgb.shape[:2]
    source_x1 = max(0, crop_x)
    source_y1 = max(0, crop_y)
    source_x2 = min(image_width, crop_x + source_size)
    source_y2 = min(image_height, crop_y + source_size)
    if source_x2 > source_x1 and source_y2 > source_y1:
        destination_x1 = source_x1 - crop_x
        destination_y1 = source_y1 - crop_y
        destination_x2 = destination_x1 + source_x2 - source_x1
        destination_y2 = destination_y1 + source_y2 - source_y1
        rgb_crop[
            destination_y1:destination_y2,
            destination_x1:destination_x2,
        ] = image_rgb[source_y1:source_y2, source_x1:source_x2]

    mask_x1 = box_x - crop_x
    mask_y1 = box_y - crop_y
    destination_x1 = max(0, mask_x1)
    destination_y1 = max(0, mask_y1)
    destination_x2 = min(source_size, mask_x1 + box_width)
    destination_y2 = min(source_size, mask_y1 + box_height)
    if destination_x2 <= destination_x1 or destination_y2 <= destination_y1:
        return None
    source_mask_x1 = destination_x1 - mask_x1
    source_mask_y1 = destination_y1 - mask_y1
    source_mask_x2 = source_mask_x1 + destination_x2 - destination_x1
    source_mask_y2 = source_mask_y1 + destination_y2 - destination_y1
    mask_crop[
        destination_y1:destination_y2,
        destination_x1:destination_x2,
    ] = instance_mask[
        source_mask_y1:source_mask_y2,
        source_mask_x1:source_mask_x2,
    ]

    if source_size != output_size:
        rgb_crop = cv2.resize(
            rgb_crop,
            (output_size, output_size),
            interpolation=cv2.INTER_AREA,
        )
        mask_crop = cv2.resize(
            mask_crop,
            (output_size, output_size),
            interpolation=cv2.INTER_NEAREST,
        )
    mask_rgb = np.repeat(
        np.where(mask_crop > 0, 255, 0).astype(np.uint8)[:, :, np.newaxis],
        3,
        axis=2,
    )
    return np.concatenate((rgb_crop, mask_rgb), axis=1)


def save_region_training_images(
    image_rgb,
    detections,
    output_dir,
    source_name,
):
    """Save one 512x1024 PNG per matched candidate and return their paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_stem = Path(source_name).stem
    saved_paths = []
    for index, detection in enumerate(detections, start=1):
        training_image = build_region_training_image(image_rgb, detection)
        if training_image is None:
            continue
        output_path = output_dir / f"{source_stem}_region_{index:03d}.png"
        saved = cv2.imwrite(
            str(output_path),
            cv2.cvtColor(training_image, cv2.COLOR_RGB2BGR),
        )
        if not saved:
            raise OSError(f"Could not save region training image: {output_path}")
        saved_paths.append(output_path)
    return saved_paths
