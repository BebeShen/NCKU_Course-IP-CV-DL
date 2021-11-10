import numpy as np
import cv2

# 2-1 Gaussian blur 
def gaussianBlurButtonClicked():
    print("gaussianBlurButtonClicked")
    src = cv2.imread('.\Dataset_OpenCvDl_Hw1\Q2_Image\Lenna_whiteNoise.jpg')
    dst = cv2.GaussianBlur(src, (5,5), cv2.BORDER_ISOLATED)
    cv2.imshow("Gaussian Smoothing", np.hstack((src, dst)))

# 2-2 Bilateral filter 
def bilateralFilterButtonClicked():
    print("bilateralFilterButtonClicked")
    src = cv2.imread('.\Dataset_OpenCvDl_Hw1\Q2_Image\Lenna_whiteNoise.jpg')
    dst = cv2.bilateralFilter(src, 9, 90, 90)
    cv2.imshow("Bilateral filter", np.hstack((src, dst)))

# 2-3 Median filter 
def medianFilterButtonClicked():
    print("medianFilterButtonClicked")
    src = cv2.imread('.\Dataset_OpenCvDl_Hw1\Q2_Image\Lenna_whiteNoise.jpg')
    dst1 = cv2.medianBlur(src, 3)
    dst2 = cv2.medianBlur(src, 5)
    cv2.imshow("3*3 Median filter", dst1)
    cv2.imshow("5*5 Median filter", dst2)