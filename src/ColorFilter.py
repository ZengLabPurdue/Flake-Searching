import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from sklearn.mixture import GaussianMixture
import scipy.ndimage as ndi

import Util
import time

#----------------------------
# Load Image
#----------------------------

image_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

def gaussian_clustering(image, n_components=5):

    totalProgress = 5
    start_time = time.time()

    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    filtered = cv2.bilateralFilter(image_lab, d=9, sigmaColor=75, sigmaSpace=75)

    Util.progress_bar(1, totalProgress, start_time)
    print("\n"+"Filtered!")
    print("Adding spatial features...")
    
    h, w = image.shape[:2]
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))

    L = filtered[:, :, 0] / 255.0
    a = filtered[:, :, 1] / 255.0
    b = filtered[:, :, 2] / 255.0
    
    features = np.column_stack([
        L.flatten(), a.flatten(), b.flatten(),
        (xs * 0.05).flatten(), (ys * 0.05).flatten()
    ])

    Util.progress_bar(2, totalProgress, start_time)
    print("\n"+"Added spatial features!")
    print("Running Gaussian Clustering...")
    
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    labels = gmm.fit_predict(features)
    
    Util.progress_bar(3, totalProgress, start_time)
    print("\n"+"Added spatial features!")
    print("Running Gaussian Clustering...")

    cluster_colors = gmm.means_[:, :3]
    segmented_pixels = cluster_colors[labels]

    Util.progress_bar(4, totalProgress, start_time)
    print("\n"+"Gaussian Clustering Finished!")
    print("Segmenting...")
    
    segmented_image = segmented_pixels.reshape(h, w, 3)
    segmented_image = (segmented_image * 255).astype(np.uint8)
    
    segmented_image_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_LAB2BGR)
    
    Util.progress_bar(5, totalProgress, start_time)
    print("\n"+"Segmenting Finished!")
    print("Finished!")

    return cluster_colors, segmented_image_bgr

def remove_small_regions(mask, min_size=500, connectivity=1):

    labels, num = ndi.label(mask, structure=ndi.generate_binary_structure(2, connectivity))

    sizes = ndi.sum(np.ones_like(labels), labels, index=range(1, num + 1))

    cleaned_mask = np.zeros_like(mask, dtype=mask.dtype)

    for label_id, size in enumerate(sizes, start=1):
        if size >= min_size:
            cleaned_mask[labels == label_id] = 1

    return cleaned_mask

cluster_colors, segmented_image = gaussian_clustering(image)
Util.save_image(segmented_image)