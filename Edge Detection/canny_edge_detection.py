import cv2 as cv
import numpy as np


def generate_gaussian_filter(sigma: int | float, filter_shape: list | tuple | None):
    m, n = filter_shape
    m_half = m//2
    n_half = n//2

    gaussian_filter = np.zeros(shape=(m, n), dtype=np.float32)

    for y in range(-m_half, m_half+1):
        for x in range(-n_half, n_half+1):
            normal = 1/(2.0*np.pi*sigma**2.0)
            exp_term = np.exp(-(x**2.0 + y**2.0) / (2.0 * sigma ** 2.0))
            gaussian_filter[y+m_half, x+n_half] = normal * exp_term
    print('Gaussian filter generated')
    gaussian_filter /= np.sum(gaussian_filter)
    return gaussian_filter


def convolution(image: np.ndarray, kernel: np.ndarray):
    # Dimensions of input image and kernel
    img_height, img_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Dimension of output image
    output_height = img_height - kernel_height + 1
    output_width = img_width - kernel_width + 1

    # Initialize the output image
    output = np.zeros((output_height, output_width))

    # Apply the convolution
    for y in range(output_height):
        for x in range(output_width):
            output[y, x] = np.sum(image[y:y + kernel_height, x:x + kernel_width] * kernel)
    print('Convolution done')

    # Normalize the output image to [0, 255]
    output_normalized = ((output - output.min())*(255 / (output.max() - output.min()))).astype(np.uint8)
    return output_normalized


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

    gradient = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient = gradient/gradient.max()*255

    theta = np.arctan2(gradient_y, gradient_x)

    return gradient, theta


def non_maximum_suppression(gradient, theta):
    height, width = gradient.shape
    angle = theta * 180 / np.pi
    angle[angle < 0] += 180

    result = np.zeros((height, width))
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            q = 255
            r = 255

            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                r = gradient[i, j - 1]
                q = gradient[i, j + 1]

            elif 22.5 <= angle[i, j] < 67.5:
                r = gradient[i - 1, j + 1]
                q = gradient[i + 1, j - 1]

            elif 67.5 <= angle[i, j] < 112.5:
                r = gradient[i - 1, j]
                q = gradient[i + 1, j]

            elif 112.5 <= angle[i, j] < 157.5:
                r = gradient[i + 1, j + 1]
                q = gradient[i - 1, j - 1]

            if (gradient[i, j] >= q) and (gradient[i, j] >= r):
                result[i, j] = gradient[i, j]
            else:
                result[i, j] = 0
    print('Non-maximum suppression Done')
    return result


def hysteresis_threshold(image, high_thresh, low_thresh):
    output_img = np.zeros_like(image)

    output_img[image >= high_thresh] = 0
    output_img[(image >= low_thresh) & (image < high_thresh)] = 255

    weak_edges_rows, weak_edges_cols = np.where((image >= low_thresh) & (image < high_thresh))

    for i in range(len(weak_edges_rows)):
        row, col = weak_edges_rows[i], weak_edges_cols[i]
        for neighbor_row in [-1, 0, 1]:
            for neighbor_column in [-1, 0, 1]:
                if ((0 <= (row + neighbor_row) < image.shape[0]) and (0 <= (col + neighbor_column) < image.shape[1]) and (output_img[row + neighbor_row, col + neighbor_column] == 255)):
                    output_img[row, col] = 255
                    break
    print('Hysteresis threshold done')
    return output_img


def main():
    # 1. Convert image to grayscale image:
    path = input('Enter image path along with image name: ')
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
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
    kernel = generate_gaussian_filter(sig, filter_s)
    gray_resized_np = np.array(gray_resized, np.float32)

    blurred_img = convolution(gray_resized_np, kernel)

    # 3. Apply sobel operator
    magnitude, direction = apply_sobel(blurred_img)

    # 4. Apply non-maximum suppression
    non_maximum = non_maximum_suppression(magnitude, direction)

    # 5. Apply hysterises filter
    high_thresh = float(input('Enter the value of high threshold: '))
    low_thresh = float(input('Enter the value of low threshold: '))
    final_result = hysteresis_threshold(non_maximum, high_thresh, low_thresh)

    # 6. Show the result
    cv.imshow('Original image', img)
    cv.imshow('Grayed convolution', gray_resized)
    cv.imshow('Result', final_result)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
