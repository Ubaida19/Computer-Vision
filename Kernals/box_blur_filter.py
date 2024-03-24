import cv2 as cv
import numpy as np

def apply_box_blur_filter(img):
    box = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
    img_height, img_width = img.shape
    box_height, box_width = box.shape
    output_height = img_height - box_height + 1
    output_width = img_width - box_width + 1

    output = np.zeros((output_height, output_width))

    for y in range(output_height):
        for x in range(output_width):
            output[y, x] = np.sum(img[y:y+box_height, x:x+box_width] * box)
    # Normalize the output image to [0, 255]
    output_normalized = ((output - output.min()) * (255 / (output.max() - output.min()))).astype(np.uint8)
    cv.imshow('Box Blur Filter', output_normalized)
    cv.waitKey(0)
    cv.destroyAllWindows()

img = cv.imread('img4.jpg', cv.IMREAD_GRAYSCALE)
if img is None:
    print('Could not open or find the image')
width = 1366
height = 768
dim = (width, height)
# Use cv.INTER_LINEAR or another interpolation method
grey_resized = cv.resize(img, dim, interpolation=cv.INTER_LINEAR)

apply_box_blur_filter(grey_resized)
