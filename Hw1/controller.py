
import numpy as np
import cv2

# 1-1 Load Image
def loadImageButtonClicked():
    print("loadImageButtonClicked")
    img = cv2.imread('.\Dataset_OpenCvDl_Hw1\Q1_Image\Sun.jpg')
    cv2.imshow('My Image', img)
    imgHeight, imgWidth, imgChannels = img.shape
    print("Height : " + str(imgHeight))
    print("Width : " + str(imgWidth))

# 1-2 Color Separation 
def colorSeparateButtonClicked():
    print("colorSeparateButtonClicked")
    img = cv2.imread('.\Dataset_OpenCvDl_Hw1\Q1_Image\Sun.jpg', cv2.IMREAD_COLOR)

    imgGreen = img.copy()
    imgBlue = img.copy()
    imgRed = img.copy()

    # set green and red channels to 0
    imgBlue[:, :, 1] = 0
    imgBlue[:, :, 2] = 0
    # set blue and red channels to 0
    imgGreen[:, :, 0] = 0
    imgGreen[:, :, 2] = 0
    # set blue and green channels to 0
    imgRed[:, :, 0] = 0
    imgRed[:, :, 1] = 0
    # RGB - Blue
    cv2.imshow('B-RGB', imgBlue)
    # RGB - Green
    cv2.imshow('G-RGB', imgGreen)
    # RGB - Red
    cv2.imshow('R-RGB', imgRed)