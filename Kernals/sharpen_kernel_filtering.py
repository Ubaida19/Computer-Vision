import numpy as np
import cv2 as cv


def generate_sharpen_kernel():
    # The 3x3 kernel looks like this:
    kernel = np.array([[0, -1, 0], [0, 9, 0], [0, -1, 0]])
    return kernel


def apply_sharpen_kernel(kernel, img):
    img_height, img_width = img.shape
    kernel_height, kernel_width = kernel.shape
    output_height = img_height - kernel_height + 1
    output_width = img_width - kernel_width + 1

    output = np.zeros((output_height, output_width))

    for y in range(output_height):
        for x in range(output_width):
            output[y, x] = np.sum(img[y:y+kernel_height, x:x+kernel_width] * kernel)
    # Normalize the output image to [0, 255]
    output_normalized = ((output - output.min()) * (255 / (output.max() - output.min()))).astype(np.uint8)
    cv.imshow('Sharpen', output_normalized)
    cv.waitKey(0)
    cv.destroyAllWindows()

def main():
    img = cv.imread('img2.jpg', cv.IMREAD_GRAYSCALE)
    width = 1366
    height = 768
    dim = (width, height)
    gray_resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    kernel = generate_sharpen_kernel()
    apply_sharpen_kernel(kernel, gray_resized)

main()