import cv2 as cv
import numpy as np
import math


def rotate_image(img):
    height, width, color_channel = img.shape
    angle = int(input("Enter the angle of rotation: "))

    # Convert the angle to radians.
    angle_radians = math.radians(angle)

    # Multiply the pixel value [x, y] in the original image with the following matrix depending upon the case.

    # For anti-clockwise rotation the matrix use is:
    # [cos(angle_radians)    -sin(angle_radians)
    #  sin(angle_radians)    cos(angle_radians) ]

    # The resultant pixels would be:
    # x' = x * cos(angle_rad) - y * sin(angle_rad)
    # y' = x * sin(angle_rad) + y * cos(angle_rad)

    # For clockwise rotation the matrix use is:
    # [cos(angle_radians)     sin(angle_radians)
    #  -sin(angle_radians)    cos(angle_radians) ]

    # The resultant pixels would be :
    # x' = x * cos(angle_rad) + y * sin(angle_rad)
    # y' = -x * sin(angle_rad) + y * cos(angle_rad)

    # Define the height and width of the new image that is to be formed
    new_width = round(abs(width*math.cos(angle_radians)) + abs(height*math.sin(angle_radians)))
    new_height = round(abs(height*math.cos(angle_radians)) + abs(width*math.sin(angle_radians)))

    # define another image variable of dimensions of new_height and new _column filled with zeros
    new_img = np.zeros((new_height, new_width, color_channel), np.uint8)

    # Find the centre of the image about which we have to rotate the image
    original_center_height = round(((height+1)/2)-1)
    original_center_width = round(((width+1)/2)-1)

    # Find the centre of the new image that will be obtained
    new_center_height = round(((new_height+1)/2)-1)
    new_center_width = round(((new_width+1)/2)-1)

    for i in range(height):
        for j in range(width):
            # Co-ordinates of pixel with respect to the centre of original image
            y = height - 1 - i - original_center_height
            x = width - 1 - j - original_center_width

            # Assuming the counter-clockwise rotation
            new_x = round(x*math.cos(angle_radians) + y*math.sin(angle_radians))
            new_y = round(-x*math.sin(angle_radians) + y*math.cos(angle_radians))

            '''since image will be rotated the centre will change too, 
                so to adjust to that we will need to change new_x and new_y with respect to the new centre'''
            new_y = new_center_height-new_y
            new_x = new_center_width-new_x

            if 0 <= new_x < new_width and 0 <= new_y < new_height:
                new_img[new_y][new_x] = img[i][j]
            else:
                raise ValueError("new_x and new_y are out of bound")

    cv.imshow('Rotated image', new_img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    img = cv.imread('img.png')
    cv.imshow('image', img)
    rotate_image(img)
    cv.waitKey(0)


if __name__ == '__main__':
    main()
