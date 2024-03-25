import numpy as np
import cv2 as cv

def apply_sobel(image: np.ndarray):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    # Dimensions of input image and kernel
    img_height, img_width = image.shape
    # Dimension of output image
    output_height = img_height - 3 + 1
    output_width = img_width - 3 + 1
    # Initialize the output image
    gradient_x = np.zeros((output_height, output_width))
    gradient_y = np.zeros((output_height, output_width))
    for y in range(output_height):
        for x in range(output_width):
            gradient_x[y, x] = np.sum(image[y:y + 3, x:x + 3] * sobel_x)
    print('Sobel X done!')
    for y in range(output_height):
        for x in range(output_width):
            gradient_y[y, x] = np.sum(image[y:y + 3, x:x + 3] * sobel_y)
    print('Sobel Y done!')
    gradient = gradient_x + gradient_y

    cv.imshow('GradientX', gradient_x)
    cv.waitKey(0)
    cv.imshow('GradientY', gradient_y)
    cv.waitKey(0)
    cv.imshow('Gradient', gradient)
    cv.waitKey(0)
    cv.destroyAllWindows()


img = cv.imread('img2.jpg', cv.IMREAD_GRAYSCALE)
width = 1366
height = 768
dim = (width, height)
gray_resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)

apply_sobel(gray_resized)
