import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tkinter import filedialog

image_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

threshold = 50

blue = image[:, :, 0] 

binary = (blue >= threshold).astype(np.uint8) * 255

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

binary = (binary > 0).astype(np.uint8) * 255
h, w = binary.shape[:2]

holes = binary.copy()
mask = np.zeros((h + 2, w + 2), np.uint8)
cv2.floodFill(holes, mask, (0, 0), 255)
holes_inv = cv2.bitwise_not(holes)
filled_white_holes = cv2.bitwise_or(binary, holes_inv)
filled_white_holes = (filled_white_holes > 0).astype(np.uint8) * 255

inverted = cv2.bitwise_not(filled_white_holes).copy()
mask2 = np.zeros((h + 2, w + 2), np.uint8)
cv2.floodFill(inverted, mask2, (0, 0), 255)
holes_inv2 = cv2.bitwise_not(inverted)

plt.imshow(inverted, cmap="gray")
plt.axis("off")
plt.show()

fully_filled = cv2.bitwise_or(filled_white_holes, holes_inv2)
fully_filled = (fully_filled > 0).astype(np.uint8) * 255

plt.imshow(fully_filled, cmap="gray")
plt.axis("off")
plt.show()