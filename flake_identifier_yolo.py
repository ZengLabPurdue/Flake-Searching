"""
Flake identification using YOLO instance segmentation (labeled_seg_best.pt).

RGB image in → returns ``(scanned_image_rgb, flakes, save)``.

Each flake entry is a 6-tuple:
  ``(contour, (x, y, w, h), component_rgb, background_rgb, class_name, confidence)``

**Overlay colors** (contour + box + label; after BGR→RGB they look like):

  ========= ======================================================
  Class     Color on the image (approximate)
  ========= ======================================================
  yellow    golden / amber outline and label
  green     green
  good      orange (training class 'good' flake)
  blue      blue–magenta / reddish-blue tint on outline
  ========= ======================================================

Classes match ``train_labeled_segmentor.py``: yellow, green, good, blue (YOLO class ids 0–3).

CLI:  python flake_identifier_yolo.py path/to/image.jpg
      python flake_identifier_yolo.py   # pick file in a dialog
"""
import argparse
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


def _draw_label_bgr(
    img_bgr, cx: int, cy: int, text: str, color_bgr: tuple[int, int, int]
):
    """Draw ``text`` in ``color_bgr`` just above (cx, cy), with a dark backdrop."""
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    pad = 3
    x0, y0 = cx, cy - th - pad * 2
    x1, y1 = cx + tw + pad * 2, cy + pad
    h, w = img_bgr.shape[:2]
    if x1 > w:
        x0 = max(0, w - (x1 - x0))
        x1 = x0 + tw + pad * 2
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if y1 > h:
        y0 = max(0, h - (y1 - y0))
        y1 = h
    cv2.rectangle(img_bgr, (x0, y0), (x1, y1), (15, 15, 15), -1)
    cv2.putText(
        img_bgr, text, (x0 + pad, y1 - pad),
        font, scale, color_bgr, thick, cv2.LINE_AA,
    )


def _draw_legend_bgr(img_bgr, counts: dict[str, int]):
    lx, ly = 12, 20
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    header = "Class (count) = outline color"
    (tw, th), _ = cv2.getTextSize(header, font, scale, thick)
    cv2.rectangle(img_bgr, (lx - 3, ly - th - 3), (lx + tw + 6, ly + 5), (15, 15, 15), -1)
    cv2.putText(
        img_bgr, header, (lx, ly), font, scale, (210, 210, 210), thick, cv2.LINE_AA,
    )
    ly += th + 10
    for name in CLASS_NAMES:
        n = counts[name]
        label = f"  {name}: {n}"
        color_bgr = CLASS_COLORS_BGR[name]
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        cv2.rectangle(img_bgr, (lx - 3, ly - th - 3), (lx + tw + 6, ly + 5), (15, 15, 15), -1)
        cv2.putText(
            img_bgr, label, (lx, ly), font, scale, color_bgr, thick, cv2.LINE_AA,
        )
        ly += th + 10


def load_image_rgb(path: str | Path) -> np.ndarray:
    """Load an image file as RGB uint8. Raises FileNotFoundError / ValueError if missing or empty."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Image not found: {p}")
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {p}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


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

        Flakes include ``class_name`` (str) and ``confidence`` (float) per detection
        after ``background_rgb``.
        """
        start_time = time.time()
        save = False
        flakes = []
        class_counts = {n: 0 for n in CLASS_NAMES}

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
            class_counts[name] += 1

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
                    name,
                    conf,
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

            M = cv2.moments(c_fixed)
            if M["m00"] and M["m00"] > 1e-6:
                tcx = int(M["m10"] / M["m00"])
                tcy = int(M["m01"] / M["m00"])
            else:
                tcx, tcy = (new_x + new_x2) // 2, (new_y + new_y2) // 2
            _draw_label_bgr(scanned_bgr, tcx, tcy, f"{name} {conf:.2f}", color_bgr)

        _draw_legend_bgr(scanned_bgr, class_counts)

        scanned_rgb = cv2.cvtColor(scanned_bgr, cv2.COLOR_BGR2RGB)

        if output:
            print(f"Time taken: {time.time() - start_time:.2f}s")
            plt.figure(figsize=(10, 8))
            plt.imshow(scanned_rgb)
            plt.title("YOLO-seg: class + confidence (see legend)")
            plt.axis("off")
            plt.show()

        return scanned_rgb, flakes, save

    def identify_flakes_from_path(
        self,
        image_path: str | Path,
        *,
        output: bool = False,
    ):
        """Load ``image_path`` (RGB) and run :meth:`identify_flakes`."""
        rgb = load_image_rgb(image_path)
        return self.identify_flakes(rgb, output=output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run YOLO flake segmentation on an image.",
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Path to image (.png, .jpg, …). If omitted, a file picker opens.",
    )
    parser.add_argument(
        "-o",
        "--out",
        metavar="PATH",
        default=None,
        help="Save overlay image (RGB) to this path (e.g. overlay.jpg).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the matplotlib window.",
    )
    args = parser.parse_args()

    path = args.image
    if path is None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
        )
        if not path:
            raise SystemExit(0)

    fid = Flake_Identifier()
    out_rgb, flakes, save = fid.identify_flakes_from_path(
        path, output=not args.no_show
    )
    print(f"{path}: {len(flakes)} flake(s), save={save}")
    for _, _, _, _, fn, fc in flakes:
        print(f"  {fn}  conf={fc:.3f}")

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(outp), cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise SystemExit(f"Failed to write: {outp}")
