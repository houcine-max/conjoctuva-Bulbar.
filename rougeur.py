
"""
Created on Thu Jan 26 18:45:00 2023

@author: Acer
"""

import cv2
import numpy as np

# Read in the image
img = cv2.imread("BT_03_1")

# Convert the image to the HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the range of red color in the HSV color space
lower_red = (0, 50, 50)
upper_red = (10, 255, 255)

# Threshold the image to only select red pixels
mask = cv2.inRange(hsv, lower_red, upper_red)

# Extract the region of interest (ROI) for the bulbar conjunctiva
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)
_,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
_,contours,_= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Extract the bulbar conjunctiva contour
for i in range(len(contours)):
    if cv2.contourArea(contours[i]) > 100:
        bulbar_conjunctiva = contours[i]
        break

# Create a mask for the ROI
mask = np.zeros_like(img)
cv2.drawContours(mask, [bulbar_conjunctiva], -1, (255,255,255), -1)

# Extract the red pixels within the ROI
red_pixels = cv2.bitwise_and(img, img, mask=mask)

# Count the number of red pixels in the ROI
red_pixels = cv2.cvtColor(red_pixels, cv2.COLOR_BGR2HSV)
red_pixels = cv2.inRange(red_pixels, lower_red, upper_red)
red_pixel_count = cv2.countNonZero(red_pixels)

# Calculate the percentage of red pixels in the ROI
bulbar_conjunctiva_area = cv2.contourArea(bulbar_conjunctiva)
redness_percentage = (red_pixel_count / bulbar_conjunctiva_area) * 100

print("Percentage of red pixels in the bulbar conjunctiva: {:.2f}%".format(redness_percentage))

# Show the image
cv2.imshow("Redness Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
