import cv2
import numpy as np
import os
from tkinter import filedialog

outputFile = "output.txt"

background_color = np.array([0, 0, 0]) 
value = 1

image_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
image_sub = image.astype(np.int16) - background_color
image_sub = np.clip(image_sub, 0, 255).astype(np.uint8)

height, width, _ = image_sub.shape

with open(outputFile, "w") as f:
    f.write("  R   G   B   V.\n")
    
    for y in range(height):
        for x in range(width):
            r, g, b = image_sub[y, x]
            f.write(f"{r:3d}{g:4d}{b:4d}{value:5d}\n")

print(f"Pixel data written to {outputFile}")