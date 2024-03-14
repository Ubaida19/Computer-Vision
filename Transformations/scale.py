import cv2 as cv
import numpy as np

def scale(img):
    # Calculate height and width of original image
    height = len(img)
    width = len(img[0])

    # Input scale factor
    scale_factor = int(input('Enter the scale factor: '))

    new_height = height * scale_factor
    new_width = width * scale_factor


    # Create new image
    new_img = []
    for _ in range(new_height):
        row = []
        for _ in range(new_width):
            pixels = [0, 0, 0]
            row.append(pixels)
        new_img.append(row)

    # Scale image
    for i in range(new_height):
        for j in range(new_width):
            # Map the location in the new image to the corresponding location in the original image
            org_i = int(i // scale_factor)
            org_j = int(j // scale_factor)
            new_img[i][j] = img[org_i][org_j]
    # Converting it to numpy array, so that it can be displayed.
    new_image = np.array(new_img, dtype=np.uint8)
    cv.imshow('Scaled Image', new_image)
    print(img.shape)
    print(new_image.shape)
    cv.waitKey(0)


img = cv.imread('img.png')
cv.imshow('Image', img)
scale(img)
cv.waitKey(0)
