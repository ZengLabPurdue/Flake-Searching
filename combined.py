import os
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from tkinter import *
from tkinter import filedialog

home_dir = os.path.dirname(os.path.abspath(__file__))
brody_work_path = Path(home_dir) / "Brody's Work"

sys.path.insert(0, str(brody_work_path))

import single_frame_pipeline

image_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

print(type(image_rgb))

masked_image, contours = single_frame_pipeline.process_frame(image_rgb)

# Assuming `contours` is a list of ndarrays with shape (n_points, 2)
opencv_contours = [c.astype(np.int32).reshape(-1, 1, 2) for c in contours]

# Draw contours on the image
image_with_contours = image_rgb.copy()
cv2.drawContours(image_with_contours, opencv_contours, -1, color=(255, 0, 0), thickness=4)  # red contours

plt.imshow(masked_image)
plt.title("Original Image")
plt.axis("off")
plt.show()

# Show with matplotlib
plt.imshow(image_with_contours)
plt.title("Contours on Image")
plt.axis("off")
plt.show()

print("Run")
