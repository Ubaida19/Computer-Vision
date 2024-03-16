import cv2 as cv

image = cv.imread('sunflower.png', cv.IMREAD_GRAYSCALE)

_, binary_image = cv.threshold(image, 127, 255, cv.THRESH_BINARY)

# Changing image size according to my screen size
width = 1366
height = 768
dimensions = (width, height)
binary_img_resize = cv.resize(binary_image, dimensions)

cv.imshow('Binary Image', binary_img_resize)
cv.waitKey(0)

