import cv2
from tkinter import filedialog
import sys
import time

def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"

def progress_bar(progress, total, start_time, bar_length=40):
    percent = progress / total
    filled = int(bar_length * percent)
    bar = "█" * filled + "-" * (bar_length - filled)

    elapsed = time.time() - start_time

    sys.stdout.write(
        f"\r|{bar}| {percent:6.2%} "
        f"{format_time(elapsed)}"
    )
    sys.stdout.flush()

    if progress == total:
        print()

def load_image():
    image_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return image

def save_image(image, conversion_type = cv2.COLOR_RGB2BGR):
    save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("Bitmap", "*.bmp")])

    if save_path:
        image_bgr = cv2.cvtColor(image, conversion_type)
        cv2.imwrite(save_path, image_bgr)