import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from tkinter import filedialog

#----------------------------
# Load Image
#----------------------------

'''
image_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blue_image = image[:, :, 0]
green_image = image[:, :, 1]
red_image = image[:, :, 2]
'''

#----------------------------
# Image Processing
#----------------------------

#
# Filters
#

# Apply gaussian blur to reduce noise
#blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1.4)

# Apply average blur to reduce noise
#blurred_image = cv2.blur(gray_image, (100, 100))

# Apply bilateral filter to reduce noise
#smoothed_image = cv2.bilateralFilter(gray_image, d=9, sigmaColor=75, sigmaSpace=20)

#----------------------------
# Functions
#----------------------------

# 3D surface graphing
def surface_graphing(image):
    Z = image.squeeze()

    Z_small = Z[::10, ::10]

    x = np.arange(Z_small.shape[1])
    y = np.arange(Z_small.shape[0])
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z_small, linewidth=0, antialiased=False)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z value")

    plt.show()

# 3D visualization of channel data
def channel_data_3D_plot(image):
    pixels = image.reshape(-1, 3)
    pixels = pixels[::100] 
    pixels = np.unique(pixels, axis=0) 

    r = pixels[:, 0]
    g = pixels[:, 1]
    b = pixels[:, 2]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(b, g, r, c=pixels/255, s=2)

    ax.set_xlabel("Blue")
    ax.set_ylabel("Green")
    ax.set_zlabel("Red")

    plt.show()

# Displays colors
def display_colors(rgb_colors, grid_shape):

    rows, cols = grid_shape
    n = len(rgb_colors)

    if rows * cols < n:
        raise ValueError("Grid shape is too small for the number of colors")

    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))

    axes = np.atleast_2d(axes)

    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c]

        if idx < n:
            color = rgb_colors[idx]
            swatch = np.ones((50, 50, 3), dtype=np.uint8) * color
            ax.imshow(swatch)
            ax.set_title(f"Cluster {idx}")
        else:
            ax.axis("off")

        ax.axis("off")

    plt.tight_layout()
    plt.show()

#----------------------------
# Testing
#----------------------------

'''
surface_graphing(gray_image)
surface_graphing(blurred_image)
surface_graphing(smoothed_image)
'''

#channel_data_3D_plot(image)