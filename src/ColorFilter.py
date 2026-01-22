import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from sklearn.mixture import GaussianMixture
import scipy.ndimage as ndi
import hdbscan
# from scipy.spatial.distance import cdist

import Util
import DataVisualizer
import InitialPassFilter
import time

#----------------------------
# Load Image
#----------------------------

image_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

def gaussian_clustering(colors, n_components=10, random_state=42):

    gmm = GaussianMixture(n_components=n_components, covariance_type="diag", random_state=random_state)
    gmm.fit(colors)

    centers = gmm.means_

    return centers

def hdbscan_clustering_colors(colors_rgb, color_type="lab", min_cluster_size=25, min_samples=5):

    if (color_type == "lab"):
        colors = cv2.cvtColor(colors_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    else:
        colors = colors_rgb

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean")

    start_time = time.time()
    labels = clusterer.fit_predict(colors)
    print(f"HDBSCAN finished in {time.time() - start_time:.2f}s")

    unique_labels = [l for l in np.unique(labels) if l != -1]

    centers = np.array([colors[labels == label].mean(axis=0) for label in unique_labels])

    if (color_type == "lab"):
        centers = centers.astype(np.uint8)
        centers = cv2.cvtColor(centers.reshape(-1, 1, 3), cv2.COLOR_LAB2RGB).reshape(-1, 3)

    return centers

def hdbscan_clustering_image(image_bgr, color_type="lab", subsample=2, min_cluster_size=1000, min_samples=100):

    start_time = time.time()

    if (color_type == "lab"):
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    elif (color_type == "rgb"):
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    else:
        image = image_bgr

    if subsample > 1:
        image_sub = image[::subsample, ::subsample, :]
    else:
        image_sub = image

    h_sub, w_sub, _ = image_sub.shape

    colors = image_sub.reshape(-1, 3)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean")

    labels = clusterer.fit_predict(colors)
    
    print(f"HDBSCAN finished in {time.time() - start_time:.2f}s")

    unique_labels = [l for l in np.unique(labels) if l != -1]
    centers = np.array([
        colors[labels == label].mean(axis=0)
        for label in unique_labels
    ])

    labels_2d = labels.reshape(h_sub, w_sub)

    plt.imshow(labels_2d)
    plt.axis("off")
    plt.show()

    if (color_type == "lab"):
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        centers = centers.astype(np.uint8)
        centers = cv2.cvtColor(centers.reshape(-1, 1, 3), cv2.COLOR_LAB2RGB).reshape(-1, 3)

    return centers # rgb

def remove_small_regions(mask, min_size=500, connectivity=1):

    labels, num = ndi.label(mask, structure=ndi.generate_binary_structure(2, connectivity))

    sizes = ndi.sum(np.ones_like(labels), labels, index=range(1, num + 1))

    cleaned_mask = np.zeros_like(mask, dtype=mask.dtype)

    for label_id, size in enumerate(sizes, start=1):
        if size >= min_size:
            cleaned_mask[labels == label_id] = 1

    return cleaned_mask

#clusters = 10
#cluster_colors, segmented_image = gaussian_clustering(image)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

original_unique_colors = np.unique(image_rgb.reshape(-1, 3), axis=0)
print("Original Unique Colors: " + str(len(original_unique_colors)))

#filtered_image = cv2.GaussianBlur(image_rgb, (5, 5), 1.4)
filtered_image = cv2.bilateralFilter(image_rgb, d=9, sigmaColor=150, sigmaSpace=75)

image_sub = filtered_image[::1, ::1, :]
colors = np.unique(image_sub.reshape(-1, 3), axis=0)
print("Filtered Unique Colors: " + str(len(colors)))

print(f"Percent Decrease: {round(1-len(colors)/len(original_unique_colors), 2)}")

cluster_colors = hdbscan_clustering_colors(colors)

#cluster_colors = hdbscan_clustering_image(image)

print(len(cluster_colors))
DataVisualizer.display_colors_many(cluster_colors, sorting="hsv")

plt.imshow(image_rgb)
plt.axis("off")
plt.show()

cleaned_image = InitialPassFilter.find_nearest_colors(image, cluster_colors, cluster_colors, use_lab=False)
plt.imshow(cleaned_image)
plt.axis("off")
plt.show()

start_time = time.time()
DataVisualizer.surface_graphing(cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2GRAY))

'''
cluster_centers = hdbscan_clustering_image(image)
print(len(cluster_centers))
'''