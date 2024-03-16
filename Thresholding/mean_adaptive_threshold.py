import cv2 as cv
import numpy as np


org_img = cv.imread('sunflower.png')
# Changing image size according to my screen size
width = 1366
height = 768
dimensions = (width, height)
# Resize the image
org_img = cv.resize(org_img, dimensions)


img = cv.imread('sunflower.png', cv.IMREAD_GRAYSCALE)
# Changing image size according to my screen size
# Resize the image
gray_img = cv.resize(img, dimensions)


const = 5
blocK_size = 7
# Padding is not necessary.
padded_img = np.pad(gray_img, blocK_size//2, mode='constant')
new_img = np.zeros_like(gray_img)

for i in range(gray_img.shape[0]):
    for j in range(gray_img.shape[1]):
        local_mean = np.mean(gray_img[i:i+blocK_size, j:j+blocK_size])
        if gray_img[i, j] > (local_mean+const):
            new_img[i, j] = 255
        else:
            new_img[i, j] = 0


cv.imshow('Colored Image', org_img)
cv.imshow('Gray Image', gray_img)
cv.imshow('Threshold Image', new_img)
cv.waitKey()
cv.destroyAllWindows()
