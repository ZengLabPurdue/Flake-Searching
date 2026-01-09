import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from sklearn.mixture import GaussianMixture
import scipy.ndimage as ndi
import hdbscan
from sklearn.neighbors import KDTree
from skimage.color import rgb2lab
# from scipy.spatial.distance import cdist

import Util
import DataVisualizer
import time

#----------------------------
# Load Image
#----------------------------

image_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

def gaussian_clustering(image, n_components=10):

    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    filtered = cv2.bilateralFilter(image_lab, d=9, sigmaColor=75, sigmaSpace=75)
    
    h, w = image.shape[:2]
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))

    L = filtered[:, :, 0] / 255.0
    a = filtered[:, :, 1] / 255.0
    b = filtered[:, :, 2] / 255.0
    
    features = np.column_stack([
        L.flatten(), a.flatten(), b.flatten(),
        (xs * 0.05).flatten(), (ys * 0.05).flatten()
    ])
    
    gmm = GaussianMixture(n_components=n_components, covariance_type='diag', random_state=42)
    labels = gmm.fit_predict(features)

    cluster_colors = gmm.means_[:, :3]
    segmented_pixels = cluster_colors[labels]
    
    segmented_image = segmented_pixels.reshape(h, w, 3)
    segmented_image = (segmented_image * 255).astype(np.uint8)
    
    segmented_image_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_LAB2RGB)

    return cluster_colors, segmented_image_bgr

def hdbscan_clustering(image_bgr, min_cluster_size=200, min_samples=20, min_center_distance=15.0):

    # --- Convert image to Lab color space ---
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_lab = rgb2lab(image_rgb)
    image_lab_sub = image_lab[::2, ::2, :]

    # Now flatten
    pixels = image_lab_sub.reshape(-1, 3)
    h, w, _ = image_lab_sub.shape

    # --- HDBSCAN clustering ---
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean")

    startTime = time.time()
    print("Started!")
    labels = clusterer.fit_predict(pixels)
    endTime = time.time()
    print(f"Finished! Time took: {Util.format_time(endTime-startTime)}")

    unique_labels = [l for l in np.unique(labels) if l != -1]

    centers = {label: pixels[labels == label].mean(axis=0) for label in unique_labels}
    label_list = list(centers.keys())

    if len(label_list) > 1:
        centers_array = np.array([centers[l] for l in label_list])

        tree = KDTree(centers_array)
        neighbors = tree.query_radius(centers_array, r=min_center_distance)

        parent = {l: l for l in label_list}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, nbrs in enumerate(neighbors):
            for j in nbrs:
                if i != j:
                    union(label_list[i], label_list[j])

        merged_labels = labels.copy()
        label_mapping = {l: find(l) for l in label_list}
        for old, new in label_mapping.items():
            merged_labels[merged_labels == old] = new
    else:
        merged_labels = labels.copy()

    final_labels = np.unique(merged_labels)
    final_centers = {}
    for label in final_labels:
        if label == -1:
            continue
        final_centers[label] = pixels[merged_labels == label].mean(axis=0)

    return np.vstack(list(final_centers.values())), merged_labels.reshape(h, w).astype(np.uint8)

def remove_small_regions(mask, min_size=500, connectivity=1):

    labels, num = ndi.label(mask, structure=ndi.generate_binary_structure(2, connectivity))

    sizes = ndi.sum(np.ones_like(labels), labels, index=range(1, num + 1))

    cleaned_mask = np.zeros_like(mask, dtype=mask.dtype)

    for label_id, size in enumerate(sizes, start=1):
        if size >= min_size:
            cleaned_mask[labels == label_id] = 1

    return cleaned_mask

clusters = 10
#cluster_colors, segmented_image = gaussian_clustering(image)
cluster_colors, segmented_image = hdbscan_clustering(image)

lab_uint8 = np.zeros_like(cluster_colors, dtype=np.uint8)
lab_uint8[:, 0] = (cluster_colors[:, 0] * 255 / 100).astype(np.uint8)    # L
lab_uint8[:, 1] = (cluster_colors[:, 1] + 128).astype(np.uint8)          # a
lab_uint8[:, 2] = (cluster_colors[:, 2] + 128).astype(np.uint8)          # b

lab_uint8 = lab_uint8.reshape(-1, 1, 3)

bgr = cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2BGR)
rgb_cluster_colors = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
rgb_cluster_colors = rgb_cluster_colors.reshape(-1, 3)

DataVisualizer.display_colors(rgb_cluster_colors, (1, rgb_cluster_colors.shape[0]))
print(rgb_cluster_colors)
Util.save_image(segmented_image)