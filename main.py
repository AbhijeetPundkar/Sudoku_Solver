
import cv2
import numpy as np
import os
import sudoku
from utils import *
path_image = "resources\img.png"

height = 378
width = 378
# model = initializePredictionModel()

##PREPARING THE IMAGE
img = cv2.imread(path_image)
# denoised_image = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

# # Step 3: Sharpening the image using a kernel
# kernel = np.array([[0, -1, 0], 
#                    [-1, 5, -1], 
#                    [0, -1, 0]])  # Sharpening kernel
# img= cv2.filter2D(denoised_image, -1, kernel)

img = cv2. resize(img, (width, height))
imgBlank = np.zeros((height, width, 3),np.uint8)
imgThreshold = preProcess(img)

##FINDING CONTOURS
imgContours = img.copy()
imgBigContour = img.copy()
contours,hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours,contours,-1,(0,255,0),3)

## FINDING BIGGEST CONTOUR
biggest, maxArea = biggestContour(contours)
if biggest.size!=0:
    biggest = reorder(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0,0,255), 20)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img,matrix,(width,height))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)

## SPLIT AND PREDICT
imgSolvedDigits = imgBlank.copy()
boxes = splitBoxes(imgWarpColored)
numbers = getPrediction(boxes)
print(numbers)

imgDetectedDigits = displayNumbers(imgDetectedDigits,numbers,color=(255,0,255))
numbers = np.asarray(numbers)
posArray = np.where(numbers>0,0,1)
print(posArray)

## finding Solution of the Board

board = np.array_split(numbers,9)
try:
    sudoku.solve(board)
except:
    pass
 
flatList = []
for subList in board:
    for item in subList:
        flatList.append(item)
solvedNumbers = flatList*posArray
imgSolvedDigits = displayNumbers(imgSolvedDigits,solvedNumbers)


## Overlay Solution
pts2 = np.float32(biggest)
pts1 = np.float32([[0,0], [width,0], [0,height], [width,height]])
matrix = cv2.getPerspectiveTransform(pts1,pts2)
imgInvWarpColored = img.copy()
imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (width,height))
inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
# imgDetectedDigits = drawGrid(imgDetectedDigits)
# imgSolvedDigits = drawGrid(imgSolvedDigits)

imageArray = ([img,imgThreshold,imgContours, imgBigContour] ,
              [imgWarpColored, imgDetectedDigits,imgSolvedDigits,inv_perspective])


# cv2.imshow('Enhanced and Resized Image', inv_perspective)
stackedlmage = stackImages(imageArray, 1)
cv2.imshow('Stacked Images' ,stackedlmage)

# from screeninfo import get_monitors
# # Step 1: Get screen resolution
# monitor = get_monitors()[0]  # Get the primary monitor
# screen_width, screen_height = monitor.width, monitor.height


# # Step 3: Resize image to fit the screen while maintaining aspect ratio
# aspect_ratio = width / height

# if screen_width / width < screen_height / height:
#     new_width = screen_width
#     new_height = int(new_width / aspect_ratio)
# else:
#     new_height = screen_height
#     new_width = int(new_height * aspect_ratio)

# # Resize the image
# resized_image = cv2.resize(inv_perspective, (new_width, new_height))

# # Step 4: Add padding to fit the exact screen size
# # Calculate padding (if the resized image doesn't fill the entire screen)
# top_padding = (screen_height - new_height) // 2
# bottom_padding = screen_height - new_height - top_padding
# left_padding = (screen_width - new_width) // 2
# right_padding = screen_width - new_width - left_padding

# # Add padding to the resized image
# padded_image = cv2.copyMakeBorder(
#     resized_image, 
#     top_padding, bottom_padding, left_padding, right_padding, 
#     cv2.BORDER_CONSTANT, value=[0, 0, 0]  # Padding with black color
# )

# Step 5: Display the padded image
# cv2.imshow('Image Fitted to Screen', padded_image)

cv2.waitKey(0)

