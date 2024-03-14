import cv2 as cv
import numpy as np


points = []


def extract_roi():
    mask = np.zeros_like(img)
    mask = cv.fillPoly(mask, np.array([points]), (255, 255, 255))

    region_of_interest = cv.bitwise_and(img, mask)

    x, y, w, h = cv.boundingRect(np.array([points]))
    region_of_interest = region_of_interest[y:y + h, x:x + w]

    return region_of_interest


def scale(polygon_roi):
    scale_factor = float(input('Enter the scale factor'))
    scaled_roi = cv.resize(polygon_roi, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_CUBIC)
    cv.imshow('ROI', polygon_roi)
    cv.imshow('Scaled Image', scaled_roi)
    cv.waitKey(0)
    cv.destroyAllWindows()


def rotate(polygon_roi):
    angle = int(input('Enter the value of angle in degrees to rotate the roi:'))
    scal = int(input('Enter the scale factor if necessary else enter 1'))
    width, height = polygon_roi.shape[:2]
    rotation_matrix = cv.getRotationMatrix2D(center=(width, height), angle=angle, scale=scal)
    dst_img = cv.warpAffine(src=polygon_roi, M=rotation_matrix, dsize=(width, height))
    cv.imshow('Original ROI: ', polygon_roi)
    cv.imshow('Rotated ROI: ', dst_img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def translate(polygon_roi):
    x = float(input('Enter the value of x for translation: '))
    y = float(input('Enter the value of y for translation: '))

    translation_matrix = np.float32([[1, 0, x], [0, 1, y]])
    dst_img= cv.warpAffine(src=polygon_roi, M=translation_matrix, dsize=(polygon_roi.shape[1], polygon_roi.shape[0]))
    cv.imshow('Original ROI: ', polygon_roi)
    cv.imshow('Translated ROI: ', dst_img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def draw_shape(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        points.append([x, y])
    if len(points) == sides:
        cv.imshow('Image', cv.polylines(img, np.array([points]), isClosed=True, color=(0, 0, 255), thickness=5))


################ Reading the specified image from the provided path and displaying it using matplotlib ################
path = input('Enter the image name you want to read: ')
img = cv.imread(path)
cv.imshow('Image', img)
cv.namedWindow('Image')

##################### Selecting Closed Shape #####################
print('An irregular polygon is a polygon that does not have all sides equal in length and all angles equal in measure such as scalene triangle, rectangle, etc. \n')
sides = int(input('So how many corners does it have: '))
cv.setMouseCallback('Image', draw_shape)
cv.waitKey(0)
print('The points of polygon are: ', points)



##################### Transformations #####################
# For applying transformations on the polygon, we need to extract the ROI of the image
# Not calling it inside each function as it will reduce performance
roi = extract_roi()
# 1. Scaling:
scale(roi)
# 2. Rotation:
rotate(roi)
# 3. Translate:
translate(roi)
