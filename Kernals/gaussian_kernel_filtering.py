import numpy as np
import cv2 as cv


def generate_gaussian_kernel(sigma: int | float, filter_shape: list | tuple | None):
    # Create a gaussian kernel
    m, n = filter_shape
    gaussian_kernel = np.zeros(shape=(m, n), dtype=np.float32)
    # Calculate the mid-point of the kernel
    m_half = m//2

    # From formula, calculate this constant value once and for all.
    normal = 1 / (2.0 * np.pi * sigma ** 2.0)
    for y in range(-m_half, m_half+1):
        for x in range(-m_half, m_half+1):
            # Calculate the exponential term.
            exp_term = np.exp(-(x**2.0 + y**2.0) / (2.0 * sigma ** 2.0))
            gaussian_value = round((normal * exp_term), 2)
            # Fill the array with the gaussian value
            gaussian_kernel[y+m_half, x+m_half] = gaussian_value
    print('Gaussian filter generated')
    gaussian_kernel /= np.sum(gaussian_kernel)
    return gaussian_kernel


def apply_gaussian(kernel:np.ndarray, img:np.ndarray):
    # Dimensions of input image and kernel
    img_height, img_width = img.shape
    kernel_height, kernel_width = kernel.shape

    # Dimension of output image
    output_height = img_height - kernel_height + 1
    output_width = img_width - kernel_width + 1

    # Initialize the output image
    output = np.zeros((output_height, output_width))

    # Apply the convolution
    for y in range(output_height):
        for x in range(output_width):
            output[y, x] = np.sum(img[y:y + kernel_height, x:x + kernel_width] * kernel)
    print('Convolution done')

    # Normalize the output image to [0, 255]
    output_normalized = ((output - output.min())*(255 / (output.max() - output.min()))).astype(np.uint8)
    cv.imshow('Gaussian Filter', output_normalized)
    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    img = cv.imread('img3.jpg', cv.IMREAD_GRAYSCALE)
    width = 1366
    height = 768
    dim = (width, height)
    gray_resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    sig = float(input('Enter the value of sigma: '))
    shape = int(input('Enter filter shape. It must be odd.'))
    if (shape % 2) == 0:
        raise ValueError('Filter shape must be odd')
    filter_s = (shape, shape)

    kernel = generate_gaussian_kernel(sig, filter_s)
    apply_gaussian(kernel, gray_resized)


main()
