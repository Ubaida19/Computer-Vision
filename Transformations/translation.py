import cv2 as cv
import numpy as np


def translate(img):
    #[
    # 0->[1,2],
    # 1->[3,4],
    # 2->[5,6]
    # ]
    # Calculate height and width of original image
    height = len(img)   # 3
    width = len(img[0])     # 2

    # Enter inputs for translated axis
    translated_x_axis = int(input('Enter the x-axis translation: '))
    translated_y_axis = int(input('Enter the y-axis translation: '))

    # Calculate new height and width of translated image
    new_height = height + abs(translated_y_axis)
    new_width = width + abs(translated_x_axis)

    # Create new image
    new_img = []
    for _ in range(new_height):
        row = []
        for _ in range(new_width):
            pixels = [0, 0, 0]

            row.append(pixels)
            # [[0, 0, 0], [0, 0, 0], [0, 0, 0]..]
        new_img.append(row)
        # [
        # [0, 0, 0], [0, 0, 0], [0, 0, 0]..
        # [0, 0, 0], [0, 0, 0], [0, 0, 0]..
        # ]

    # Translate image
    for i in range(height):
        for j in range(width):
            new_i = i + translated_y_axis
            new_j = j + translated_x_axis
            # [5, 2] =
            if 0 <= new_i < new_height and 0 <= new_j < new_width:
                new_img[new_i][new_j] = img[i][j]

    # Converting it to numpy array
    new_img = np.array(new_img, dtype=np.uint8)

    cv.imshow('New image', new_img)
    cv.waitKey(0)


img = cv.imread('img.png')
cv.imshow('Image', img)
translate(img)
cv.waitKey(0)
