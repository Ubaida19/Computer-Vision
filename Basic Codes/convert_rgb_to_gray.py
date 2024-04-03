import cv2 as cv
import numpy as np


def convert_rgb_to_gray(img):
    print('Converting to gray')
    grey_image = np.zeros_like(img)
    height, width = img.shape[:2]
    # Converting RGB to gray using weighted average.
    for i in range(height):
        for j in range(width):
            grey_image[i, j] = int((img[i, j, 0]*0.2989)+(img[i, j, 1]*0.5870)+(img[i, j, 2] * 0.1140))
    cv.imshow('gray', grey_image)
    cv.waitKey(0)

    cv.destroyAllWindows()


def main():
    name = input('Enter image path: ')
    img = cv.imread(name)
    if img is None:
        ValueError ('Could not open or find the image')
    width = 1366
    height = 768
    dim = (width, height)
    resized_img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    cv.imshow('img',resized_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    convert_rgb_to_gray(resized_img)


if __name__ == '__main__':
    main()
