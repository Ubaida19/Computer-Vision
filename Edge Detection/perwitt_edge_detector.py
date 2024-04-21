import math
import cv2 as cv
import numpy as np


def apply_prewit(img):
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

    # Dimensions of input image and kernel
    img_height, img_width = img.shape

    # Add padding
    padded_img = np.pad(img, 1, mode='constant', constant_values=0)
    # Initialize the output image
    gradient_x = np.zeros((img_height, img_width))
    gradient_y = np.zeros((img_height, img_width))
    gradient = np.zeros((img_height, img_width))

    for y in range(img_height):
        for x in range(img_width):
            window = padded_img[y:y + 3, x:x + 3]
            gradient_x[y, x] = np.sum(window * kernel_x)
    print('Prewit X done!')

    for y in range(img_height):
        for x in range(img_width):
            window = padded_img[y:y + 3, x:x + 3]
            gradient_y[y, x] = np.sum(window * kernel_y)
    print('Prewit Y done!')

    # Normalization (divide by the maximum gradient magnitude)
    max_gradient = np.max(np.abs(gradient_x) + np.abs(gradient_y))
    gradient_x /= max_gradient
    gradient_y /= max_gradient

    # Applying the formula.
    gradient_x_squared = gradient_x * gradient_x
    gradient_y_squared = gradient_y * gradient_y

    gradient = np.zeros((img_height, img_width))
    for i in range(img_height):
        for j in range(img_width):
            gradient[i][j] = math.sqrt(gradient_x_squared[i][j] + gradient_y_squared[i][j])

    cv.imshow('GradientX', gradient_x)
    cv.waitKey(0)
    cv.imshow('GradientY', gradient_y)
    cv.waitKey(0)
    cv.imshow('Gradient', gradient)
    cv.waitKey(0)
    cv.destroyAllWindows()


img = cv.imread('scotter.jpg', cv.IMREAD_GRAYSCALE)
width = 1366
height = 768
dim = (width, height)
gray_resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
apply_prewit(gray_resized)
