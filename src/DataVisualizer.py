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
def channel_data_3D_plot(image, colorspace="bgr", sample_step=100, point_size=2):

    if colorspace.lower() == "bgr":
        plot_img = image
        display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        labels = ("Blue", "Green", "Red")

    elif colorspace.lower() == "rgb":
        plot_img = image
        display_img = image
        labels = ("Red", "Green", "Blue")

    elif colorspace.lower() == "lab":
        plot_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        display_img = cv2.cvtColor(plot_img, cv2.COLOR_LAB2RGB)
        labels = ("L*", "a*", "b*")

    elif colorspace.lower() == "hsv":
        plot_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        display_img = cv2.cvtColor(plot_img, cv2.COLOR_HSV2RGB)
        labels = ("Hue", "Saturation", "Value")

    else:
        raise ValueError("colorspace must be 'bgr', 'rgb', 'lab', or 'hsv'")

    pixels = plot_img.reshape(-1, 3)
    colors = display_img.reshape(-1, 3)

    pixels = pixels[::sample_step]
    colors = colors[::sample_step]

    pixels, unique_idx = np.unique(pixels, axis=0, return_index=True)
    colors = colors[unique_idx]

    colors = colors / 255.0

    x, y, z = pixels[:, 0], pixels[:, 1], pixels[:, 2]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x, y, z, c=colors, s=point_size)

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])

    ax.set_title(f"3D Color Distribution ({colorspace.upper()})")

    plt.tight_layout()
    plt.show()

# Displays provided colors
def display_colors_small(rgb_colors, grid_shape):

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

# Displays all colors of an image sorted by intensity
def display_colors_many(rgb_colors, sorting="lab"):
    rgb_colors = np.asarray(rgb_colors, dtype=np.uint8)

    if rgb_colors.ndim != 2 or rgb_colors.shape[1] != 3:
        raise ValueError("rgb_colors must have shape (N, 3)")

    if sorting == "intensity":
        intensities = (
            0.2126 * rgb_colors[:, 0] +
            0.7152 * rgb_colors[:, 1] +
            0.0722 * rgb_colors[:, 2]
        )
        sort_idx = np.argsort(intensities)

    elif sorting == "lab":
        lab = cv2.cvtColor(
            rgb_colors.reshape(-1, 1, 3),
            cv2.COLOR_RGB2LAB
        ).reshape(-1, 3)
        sort_idx = np.lexsort((lab[:, 2], lab[:, 1], lab[:, 0]))

    elif sorting == "hsv":
        lab = cv2.cvtColor(
            rgb_colors.reshape(-1, 1, 3),
            cv2.COLOR_RGB2HSV
        ).reshape(-1, 3)
        sort_idx = np.lexsort((lab[:, 0], lab[:, 2], lab[:, 1]))

    else:
        raise ValueError("sorting must be 'intensity', 'lab', or 'hsv'")

    rgb_colors = rgb_colors[sort_idx]

    fig = plt.figure(figsize=(16, 9))
    plt.axis("off")
    fig.canvas.draw()

    width_px, height_px = fig.canvas.get_width_height()
    N = len(rgb_colors)

    aspect = width_px / height_px

    cols = int(np.ceil(np.sqrt(N * aspect)))
    rows = int(np.ceil(N / cols))

    cell_size = min(width_px // cols, height_px // rows)

    cols = width_px // cell_size
    rows = height_px // cell_size
    n_slots = cols * rows

    if N > n_slots:
        idx = np.linspace(0, N - 1, n_slots).astype(int)
        colors = rgb_colors[idx]
    else:
        colors = rgb_colors

    grid = np.zeros(
        (rows * cell_size, cols * cell_size, 3),
        dtype=np.uint8
    )

    for i, color in enumerate(colors):
        r = i // cols
        c = i % cols
        if r >= rows:
            break

        y0, y1 = r * cell_size, (r + 1) * cell_size
        x0, x1 = c * cell_size, (c + 1) * cell_size
        grid[y0:y1, x0:x1] = color

    plt.imshow(grid, interpolation="nearest")
    plt.title(
        f"{N} colors | {len(colors)} displayed | "
        f"cell={cell_size}px | {sorting.upper()} sort",
        fontsize=12
    )
    plt.show()

#----------------------------
# Testing
#----------------------------

#surface_graphing(gray_image)

'''
surface_graphing(gray_image)
surface_graphing(blurred_image)
surface_graphing(smoothed_image)
'''

#channel_data_3D_plot(image, colorspace="bgr")

#rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#pixels = rgb.reshape(-1, 3)
#unique_colors = np.unique(pixels)
#display_colors_many(rgb_colors = unique_colors, sorting="lab")