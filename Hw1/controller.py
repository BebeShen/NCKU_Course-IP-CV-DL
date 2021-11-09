
import numpy as np
import cv2
from PIL import Image

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

# 1-3 Color Transformation 
def colorTransformateButtonClicked():
    print("colorTransformateButtonClicked")
    img = cv2.imread('.\Dataset_OpenCvDl_Hw1\Q1_Image\Sun.jpg')
    # Cv2 function
    (imgBlue, imgGreen, imgRed) = cv2.split(img)
    imgMerged = cv2.merge([imgBlue, imgGreen, imgRed])
    imgGray = cv2.cvtColor(imgMerged, cv2.COLOR_BGR2GRAY)
    print(imgGray)
    cv2.imshow('Merged by cv2', imgGray)
    # Weighted formula
    # img = cv2.imread('.\Dataset_OpenCvDl_Hw1\Q1_Image\Sun.jpg')
    # (imgBlue, imgGreen, imgRed) = cv2.split(img)
    # imgBlue = imgBlue*0.33
    # imgGreen = imgGreen*0.33
    # imgRed = imgRed*0.33
    # dst = cv2.add(imgBlue, imgGreen)
    # imgMergedW = np.round(cv2.add(dst, imgRed))
    # print(imgMergedW)
    # cv2.imshow('W', imgMergedW)

# 1-4 Blending
def blending(x):
    pass

def blendingButtonClicked():
    
    sd = cv2.imread('.\Dataset_OpenCvDl_Hw1\Q1_Image\Dog_Strong.jpg')
    wd = cv2.imread('.\Dataset_OpenCvDl_Hw1\Q1_Image\Dog_Weak.jpg')
    blendingImage = cv2.addWeighted(sd, 1, wd, 0, 0)

    windowName = "Blending Images"
    cv2.namedWindow(windowName)
    cv2.createTrackbar('Blending', windowName, 0, 255, blending)

    while True:
        cv2.imshow(windowName, blendingImage)
        if cv2.waitKey(50) == 27 or cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) < 1:
            break
        b = cv2.getTrackbarPos('Blending', windowName) / 255
        a = 1 - b
        blendingImage = cv2.addWeighted(sd, a, wd, b, 0)