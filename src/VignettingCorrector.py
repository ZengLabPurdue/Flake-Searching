import cv2
import numpy as np
from tkinter import filedialog
import DataVisualizer
import Util

#----------------------------
# Load Image
#----------------------------

image_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
flatfield_path =  filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])

#----------------------------
# Vignetting Correction Function
#----------------------------

def vignetting_correction(image_path, flatfield_path):

    image = cv2.imread(image_path).astype(np.float32)
    flatfield = cv2.imread(flatfield_path).astype(np.float32)

    if image.shape != flatfield.shape:
        raise ValueError("Image and flat-field must have the same dimensions and channels")

    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-6
    flatfield_corrected = image / (flatfield + epsilon)

    mean_flatfield = np.mean(flatfield, axis=(0, 1))
    corrected_image = flatfield_corrected * mean_flatfield

    corrected_image = np.clip(corrected_image, 0, 255).astype(np.uint8)

    return cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB)

#DataVisualizer.SurfaceGraphing(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
#DataVisualizer.ChannelData3DPlot(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
corrected_image = vignetting_correction(image_path, flatfield_path)
Util.save_image(corrected_image)
#DataVisualizer.SurfaceGraphing(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY))
#DataVisualizer.ChannelData3DPlot(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY))