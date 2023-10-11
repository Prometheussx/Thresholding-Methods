"""
Created on Sat Nov 26, 2022

@author: erdem
"""

# Thresholding is a technique that allows us to extract specific details from an image, helping us classify it.
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image and convert it to grayscale
img = cv2.imread("foto.jpg", 0)

cv2.imshow("img", img)

# Apply global thresholding
ret, th1 = cv2.threshold(img, 150, 200, cv2.THRESH_BINARY)
# The input numerical values aim to obtain the most distinct visual separation.

# Apply adaptive thresholding using the mean of the neighborhood area
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 2)
# The last two values need to be such that the result of their division leaves a remainder of 1.
# The first value controls the block size, and the second value influences the subtraction from the mean.

# Apply adaptive thresholding using the Gaussian-weighted sum of the neighborhood area
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
# Similar to the previous method, but uses a Gaussian-weighted sum instead of the mean.

cv2.imshow("img-th1", th1)
cv2.imshow("TH2", th2)
cv2.imshow("TH3", th3)
cv2.waitKey(0)
cv2.destroyAllWindows()