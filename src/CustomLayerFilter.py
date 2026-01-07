import cv2
from tkinter import filedialog

#---------------------------
# Program Parameters
#---------------------------

num_contours = 100
contour_thickness = 10
contour_color = (0,0,0) # in BGR

#----------------------------
# Load Image
#----------------------------

image_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
image = cv2.imread(image_path, cv2.IMREAD_COLOR)