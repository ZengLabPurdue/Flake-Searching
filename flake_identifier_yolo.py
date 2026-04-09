"""
Flake identification using YOLO instance segmentation (labeled_seg_best.pt).

Same public shape as flake_identifier.Flake_Identifier: RGB image in,
returns (scanned_image_rgb, flakes, save). Flakes are lists of
(contour, (x, y, w, h), component_rgb, background_rgb).

Classes match training: yellow, green, good, blue (see train_labeled_segmentor.py).
"""
import os
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

_HOME = Path(__file__).resolve().parent
_DEFAULT_WEIGHTS = _HOME / "labeled_seg_best.pt"

CLASS_NAMES = ["yellow", "green", "good", "blue"]
# BGR for cv2 drawing; image is processed in BGR then converted back to RGB
CLASS_COLORS_BGR = {
    "yellow": (0, 200, 255),
    "green": (0, 210, 80),
    "good": (0, 140, 255),
    "blue": (220, 60, 60),
}
CONF_THRESH = 0.25
BBoxPad = 1.2


class Flake_Identifier:
    def __init__(self, weights_path: Path | str | None = None):
        wpath = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS
        if not wpath.is_file():
            self.model = None
            print(f"Error: YOLO weights not found: {wpath}")
        else:
            self.model = YOLO(str(wpath))
            print("Flake identifier (YOLO-seg) initialized!")

    def identify_flakes(self, image: np.ndarray, output: bool = False):
        """
        Parameters
        ----------
        image : ndarray (H, W, 3) RGB uint8

        Returns
        -------
        scanned_image_rgb, flakes, save
        """
        start_time = time.time()
        save = False
        flakes = []

        if self.model is None or image is None or image.size == 0:
            return image.copy() if image is not None else np.array([]), flakes, save

        h_img, w_img = image.shape[:2]
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        scanned_bgr = img_bgr.copy()

        results = self.model.predict(
            img_bgr,
            conf=0.01,
            iou=0.45,
            retina_masks=True,
            verbose=False,
        )
        r0 = results[0]
        if r0.masks is None or r0.boxes is None:
            if output:
                print(f"Time taken: {time.time() - start_time:.2f}s")
            return cv2.cvtColor(scanned_bgr, cv2.COLOR_BGR2RGB), flakes, save

        masks = r0.masks.data.cpu().numpy()
        boxes = r0.boxes

        for i, box in enumerate(boxes):
            conf = float(box.conf[0])
            if conf < CONF_THRESH:
                continue
            cls_id = int(box.cls[0])
            if cls_id >= len(CLASS_NAMES):
                continue
            name = CLASS_NAMES[cls_id]
            color_bgr = CLASS_COLORS_BGR[name]

            if cls_id == 2:
                save = True

            mask_raw = masks[i]
            mask_img = cv2.resize(mask_raw, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
            binary = (mask_img > 0.5).astype(np.uint8) * 255

            cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) < 1:
                continue
            c_fixed = np.asarray(c, dtype=np.int32).reshape(-1, 1, 2)

            x, y, w, h = cv2.boundingRect(c_fixed)
            cx, cy = x + w / 2, y + h / 2
            new_w, new_h = w * BBoxPad, h * BBoxPad
            new_x = int(cx - new_w / 2)
            new_y = int(cy - new_h / 2)
            new_x2 = int(cx + new_w / 2)
            new_y2 = int(cy + new_h / 2)

            new_x = max(0, new_x)
            new_y = max(0, new_y)
            new_x2 = min(w_img, new_x2)
            new_y2 = min(h_img, new_y2)
            if new_x2 <= new_x or new_y2 <= new_y:
                continue

            contour_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            cv2.drawContours(contour_mask, [c_fixed], -1, 255, thickness=-1)
            contour_mask_crop = contour_mask[new_y:new_y2, new_x:new_x2]
            image_crop_bgr = img_bgr[new_y:new_y2, new_x:new_x2]

            flake_px = contour_mask_crop > 0
            back_px = contour_mask_crop == 0

            if not np.any(flake_px):
                continue

            if image_crop_bgr.ndim >= 3:
                comp_b = int(cv2.mean(image_crop_bgr[:, :, 0], mask=contour_mask_crop)[0])
                comp_g = int(cv2.mean(image_crop_bgr[:, :, 1], mask=contour_mask_crop)[0])
                comp_r = int(cv2.mean(image_crop_bgr[:, :, 2], mask=contour_mask_crop)[0])
            else:
                v = int(cv2.mean(image_crop_bgr, mask=contour_mask_crop)[0])
                comp_r = comp_g = comp_b = v

            # Background: mean in crop outside mask; store as RGB to match flake_identifier API
            if np.any(back_px):
                back_b = int(image_crop_bgr[:, :, 0][back_px].mean())
                back_g = int(image_crop_bgr[:, :, 1][back_px].mean())
                back_r = int(image_crop_bgr[:, :, 2][back_px].mean())
            else:
                back_b = back_g = back_r = 0

            flakes.append(
                (
                    c_fixed,
                    (new_x, new_y, int(new_w), int(new_h)),
                    (comp_r, comp_g, comp_b),
                    (back_r, back_g, back_b),
                )
            )

            cv2.drawContours(scanned_bgr, [c_fixed], -1, color=color_bgr, thickness=2)
            cv2.rectangle(
                scanned_bgr,
                (new_x, new_y),
                (new_x2, new_y2),
                color=color_bgr,
                thickness=2,
            )

        scanned_rgb = cv2.cvtColor(scanned_bgr, cv2.COLOR_BGR2RGB)

        if output:
            print(f"Time taken: {time.time() - start_time:.2f}s")
            plt.figure(figsize=(10, 8))
            plt.imshow(scanned_rgb)
            plt.title("YOLO-seg: contours and boxes by class")
            plt.axis("off")
            plt.show()

        return scanned_rgb, flakes, save


if __name__ == "__main__":
    from tkinter import filedialog

    path = filedialog.askopenfilename(
        filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")]
    )
    if not path:
        raise SystemExit(0)
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    fid = Flake_Identifier()
    fid.identify_flakes(rgb, output=True)
