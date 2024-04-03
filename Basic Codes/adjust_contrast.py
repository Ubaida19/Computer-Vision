import cv2 as cv
import numpy as np


def adjust_contrast(image, contrast_factor):
    # Convert the image to float32 for accurate calculations
    image = image.astype(np.float32)

    # Calculate the average brightness
    # avg_brightness = np.mean(image)             OR
    avg_brightness = 0
    count = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            avg_brightness += image[i][j]
            count += 1
    avg_brightness /= count

    # Calculate the new pixel values:
    # 1) subtract the average brightness from the current pixel value.
    # 2) Multiply the result by the contrast factor.
    # 3) Add the mean back to the current pixel value
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i][j] = (((image[i][j]-avg_brightness)*contrast_factor)+avg_brightness)

    # Clip the pixel values to the range [0, 255]. Value greater than 255 will be set to 255
    # and value less than 0 will be set to 0
    image = np.clip(image, 0, 255)
    # Convert the image back to uint8
    new_img = image.astype(np.uint8)
    # Show the adjusted image
    cv.imshow('adjusted img', new_img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    img_name = input('Enter image path: ')
    img = cv.imread(img_name)
    if img is None:
        ValueError('Could not open or find the image')
    width = 1366
    height = 768
    dim = (width, height)
    resized_img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    cv.imshow('img', resized_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    contrast_factor = float(input('Enter contrast factor: '))
    adjust_contrast(resized_img, contrast_factor)


if __name__ == '__main__':
    main()
