import cv2 as cv
import numpy as np

import gaussian_kernel_filtering


def apply_laplacian(img: np.ndarray, kernel: np.ndarray):
    # Dimensions of input image and kernel
    img_height, img_width = img.shape
    kernel_height, kernel_width = kernel.shape

    # Dimension of output image
    output_height = img_height - kernel_height + 1
    output_width = img_width - kernel_width + 1
    output = np.zeros(shape=(output_height, output_width), dtype=np.float32)

    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.sum(img[i: i+kernel_height, j: j+kernel_width] * kernel)

    # Normalize the output image to [0, 255]
    output_normalized = ((output - output.min()) * (255 / (output.max() - output.min()))).astype(np.uint8)

    cv.imshow('Laplacian Filter', output_normalized)
    cv.waitKey(0)
    cv.destroyAllWindows()
def create_laplacian_filter(size: int):
    # Forward difference use for calculating double derivatives.
    first_derivative1 = [1, -1]
    first_derivative2 = [1, -1]
    length = len(first_derivative1)

    # Appending zeros at the end of the lists.
    while length < size:
        first_derivative1.append(0)
        first_derivative2.append(0)
        length += 1

    # Add padding.
    pad_length = length//2
    pad = 0
    while pad < pad_length:
        first_derivative2.append(0)
        first_derivative2.insert(0, 0)
        pad += 1

    # Converting to numpy array so the convolution is easy.
    first_derivative1_np = np.array(first_derivative1)
    first_derivative2_np = np.array(first_derivative2)

    # Performing convolution, which will result in double derivative along horizontal direction.
    output = [0] * size
    output_np = np.array(output)
    for i in range(3):
        output_np[i] = np.sum(first_derivative2_np[i: i+size] * first_derivative1_np)

    # Now add padding to this double derivative.
    horizontal = []
    row = []
    middle = size // 2
    for i in range(size):
        for j in range(size):
            if i == middle:
                row.append(output_np[j])
            else:
                row.append(0)
        horizontal.append(row)
        row = []

    # Taking transpose of horizontal would lead to vertical double derivative matrix.
    vertical = [[row[i] for row in horizontal] for i in range(len(horizontal))]
    # Now simply adding horizontal and vertical double derivatives to make the laplacian filter.
    result = [[horizontal[i][j] + vertical[i][j] for j in range(len(horizontal[0]))] for i in range(len(horizontal))]

    # Normalizing the result.
    for i in range(len(result)):
        for j in range(len(result[0])):
            result[i][j] = -1 * result[i][j]

    output_np = np.array(result)
    return output_np


def main():
    # 1. Read the image and convert it into gray-scale
    img = cv.imread('scotter.jpg', cv.IMREAD_GRAYSCALE)
    width = 1366
    height = 768
    dim = (width, height)
    gray_resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)

    # 2. Remove noise using Gaussian blur
    sig = float(input('Enter the value of sigma: '))
    shape = int(input('Enter filter shape. It must be odd.'))
    if (shape % 2) == 0:
        raise ValueError('Filter shape must be odd')
    filter_s = (shape, shape)
    gaussian_kernel = gaussian_kernel_filtering.generate_gaussian_kernel(sig, filter_s)
    new_img = gaussian_kernel_filtering.apply_gaussian(gaussian_kernel, gray_resized)
    print('Now going for laplacian')
    laplacian_kernel = create_laplacian_filter(3)
    apply_laplacian(new_img, laplacian_kernel)


main()
