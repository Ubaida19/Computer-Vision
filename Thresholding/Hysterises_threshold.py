import cv2 as cv
import numpy as np


def hysteresis_threshold(image, high_thresh, low_thresh):
    output_img = np.zeros_like(image)

    output_img[image >= high_thresh] = 255
    output_img[(image >= low_thresh) & (image < high_thresh)] = 0

    weak_edges_rows, weak_edges_cols = np.where((image >= low_thresh) & (image < high_thresh))

    for i in range(len(weak_edges_rows)):
        row, col = weak_edges_rows[i], weak_edges_cols[i]
        for neighbor_row in [-1, 0, 1]:
            for neighbor_column in [-1, 0, 1]:
                if ((0 <= (row + neighbor_row) < image.shape[0]) and (0 <= (col + neighbor_column) < image.shape[1]) and (output_img[row + neighbor_row, col + neighbor_column] == 255)):
                    output_img[row, col] = 255
                    break

    cv.imshow('Image', output_img)
    cv.waitKey(0)


img = cv.imread('sunflower.png', cv.IMREAD_GRAYSCALE)
# Changing image size according to my screen size
width = 1366
height = 768
dimensions = (width, height)
gray_img = cv.resize(img, dimensions)

cv.imshow('Image', gray_img)
cv.waitKey(0)

high_threshold = int(input('Enter high threshold value: '))
low_threshold = int(input('Enter low threshold value: '))

hysteresis_threshold(gray_img, high_threshold, low_threshold)
