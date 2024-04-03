import cv2 as cv

# Load the image in grayscale
image = cv.imread("sunflower.png", cv.IMREAD_GRAYSCALE)
width = 1366
height = 768
dimensions = (width, height)
image = cv.resize(image, dimensions)
resized_img = cv.resize(image, dimensions)
# Set the parameters for adaptive thresholding
block_size = 11  # Size of the neighborhood area
C = 2  # Constant subtracted from the mean

# Apply adaptive mean thresholding
adaptive_mean_threshold = cv.adaptiveThreshold(resized_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, block_size, C)

# Display the original and thresholded images
cv.imshow("Original Image", image)
cv.imshow("Adaptive Mean Thresholding", adaptive_mean_threshold)
cv.waitKey(0)
cv.destroyAllWindows()
