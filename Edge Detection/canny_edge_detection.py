import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def generate_gaussian_filter(sigma: int | float, filter_shape: list | tuple | None):
    m, n = filter_shape
    m_half = m//2
    n_half = n//2

    gaussian_filter = np.zeros(shape=(m, n), dtype=np.float32)
    normal = 1/(2.0*np.pi*sigma**2.0)
    for y in range(-m_half, m_half+1):
        for x in range(-n_half, n_half+1):
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


def double_threshold(image, high_thresh_ratio=0.05, low_thresh_ratio=0.09):
    high_thresh = high_thresh_ratio * image.max()
    low_thresh = low_thresh_ratio * high_thresh
    height, width = image.shape
    result = np.zeros((height, width), dtype=np.int8)

    weak = np.uint(0)
    strong = np.uint8(255)

    strong_i, strong_j = np.where(image >= high_thresh)

    weak_i, weak_j = np.where((image >= low_thresh) & (image <= high_thresh))

    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak
    return result


def hysteresis_threshold(image, high_thresh, low_thresh):
    threshold_img = double_threshold(image, high_thresh, low_thresh)
    height, width = threshold_img.shape
    strong = 255
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if threshold_img[i, j] == 0:
                if ((threshold_img[i + 1, j - 1] == strong) or
                    (threshold_img[i + 1, j] == strong) or
                    (threshold_img[i + 1, j + 1] == strong) or
                    (threshold_img[i, j - 1] == strong) or
                    (threshold_img[i, j + 1] == strong) or
                    (threshold_img[i - 1, j - 1] == strong) or
                    (threshold_img[i - 1, j] == strong) or
                    (threshold_img[i - 1, j + 1] == strong)):
                    threshold_img[i, j] = strong
                else:
                    threshold_img[i, j] = 0
    result = threshold_img.astype(np.float32)
    result *= (255.0 / strong)
    result = result.astype(np.uint8)
    print('Hysteresis threshold done')
    return result


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
    high_thresh = float(input('Enter the value of high threshold Ratio: '))
    low_thresh = float(input('Enter the value of low threshold Ratio: '))
    final_result = hysteresis_threshold(non_maximum, high_thresh, low_thresh)
    canny_img = cv.Canny(image=gray_resized, threshold1=100, threshold2=150, apertureSize=3)
    # 6. Show the result
    cv.imshow('Original image', img)
    cv.imshow('Grayed convolution', gray_resized)
    cv.imshow('Result', final_result)
    cv.imshow('Canny', canny_img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
