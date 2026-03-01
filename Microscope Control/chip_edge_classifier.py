import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tkinter import filedialog

def chip_filter(image, threshold=50):

    blue = image[:, :, 0] 

    binary = (blue >= threshold).astype(np.uint8) * 255
    h, w = binary.shape[:2]

    holes = binary.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(holes, mask, (0, 0), 255)
    holes_inv = cv2.bitwise_not(holes)
    filled = cv2.bitwise_or(binary, holes_inv)
    filled = (filled > 0).astype(np.uint8) * 255

    filled_rgb = cv2.cvtColor(filled, cv2.COLOR_GRAY2RGB)

    return filled_rgb

'''
image_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(chip_filter(image))
plt.axis("off")
plt.show()
'''