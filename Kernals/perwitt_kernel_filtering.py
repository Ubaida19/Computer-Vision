import cv2 as cv
import numpy as np


def apply_perwitt(img):
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

    # Dimensions of input image and kernel
    img_height, img_width = img.shape
    # Dimension of output image
    output_height = img_height - 3 + 1
    output_width = img_width - 3 + 1
    # Initialize the output image
    gradient_x = np.zeros((output_height, output_width))
    gradient_y = np.zeros((output_height, output_width))
    for y in range(output_height):
        for x in range(output_width):
            gradient_x[y, x] = np.sum(img[y:y + 3, x:x + 3] * kernel_x)
    print('Perwitt X done!')

    for y in range(output_height):
        for x in range(output_width):
            gradient_y[y, x] = np.sum(img[y:y + 3, x:x + 3] * kernel_y)
    print('Sobel Y done!')

    cv.imshow('GradientX', gradient_x)
    cv.waitKey(0)
    cv.imshow('GradientY', gradient_y)
    cv.waitKey(0)
    cv.destroyAllWindows()


img = cv.imread('img1.jpg', cv.IMREAD_GRAYSCALE)
width = 1366
height = 768
dim = (width, height)
gray_resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
apply_perwitt(gray_resized)
