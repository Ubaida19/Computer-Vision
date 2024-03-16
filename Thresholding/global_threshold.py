import cv2 as cv
import numpy as np

org_img = cv.imread('sunflower.png', cv.IMREAD_GRAYSCALE)

# Changing image size according to my screen size
width = 1366
height = 768
dimensions = (width, height)

# Resize the image
gray_img = cv.resize(org_img, dimensions)
cv.imshow('Image', gray_img)
cv.waitKey(0)


# The following method is correct for converting an image to greyscale but for the purpose of thresholding,
# the preferred one is the one mentioned above.


# # Converting the image in greyscale
# gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
# cv.imshow('Image', gray_img)


# Set a threshold value
threshold_value = 127
thresh = np.zeros_like(gray_img)
thresh[gray_img >= threshold_value] = 255
# The expression gray_img >= threshold_value creates a binary mask where all pixels in the gray_img
# that are greater than or equal to threshold_value are marked as True, and all others are marked as
# False.
# Then All pixels where the mask is True
# (i.e., where the original image’s intensity was above the threshold) are set to 255


thresh[gray_img < threshold_value] = 0
cv.imshow('Threshold', thresh)


cv.waitKey(0)
cv.destroyAllWindows()
