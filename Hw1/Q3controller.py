import numpy as np
from numpy.core.fromnumeric import shape
import cv2
import matplotlib.pyplot as plt
import math

# 3-1 Gaussian blur 
def dnorm(x, mu, sd):
    '''Calculated the density using the formula of Univariate Normal Distribution (mean=0).

    '''
    return 1/(np.sqrt(2*np.pi)*sd)*np.e**(-np.power((x - mu)/sd, 2)/2)

def convolution(image, kernel, average=False, verbose=False):
    '''Implement convolution
    
    Args:
        image (image)   :
        kernel (int)    : 
        average (int)   : The average argument will be used only for smoothing filter
    '''
    if len(image.shape) == 3:
        print("Found 3 Channels : {}".format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Converted to Gray Channel. Size : {}".format(image.shape))
    else:
        print("Image Shape : {}".format(image.shape))
 
    print("Kernel Shape : {}".format(kernel.shape))
 
    if verbose:
        plt.imshow(image, cmap='gray')
        plt.title("Image")
        plt.show()

    # Since our convolution() function only works on image with single channel
    # we will convert the image to gray scale in case we find the image has 3 channels ( Color Image ).
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape
 
    output = np.zeros(image.shape)
 
    # zero padding
    pad_height = int((kernel_row - 1)/2)
    pad_width = int((kernel_col - 1)/2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2*pad_width)))
 
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image
 
    # if verbose:
    #     plt.imshow(padded_image, cmap='gray')
    #     plt.title("Padded Image")
    #     plt.show()
 
    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]
 
    print("Output Image size : {}".format(output.shape))
 
    # if verbose:
    #     plt.imshow(output, cmap='gray')
    #     plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))
    #     plt.show()
 
    #subplot(r,c) provide the no. of rows and columns
    fig = plt.figure(figsize=(10, 7))
    # Adds a subplot at the 1st position
    fig.add_subplot(1, 2, 1)
    
    # showing image
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title("First")
    
    # Adds a subplot at the 2nd position
    fig.add_subplot(1, 2, 2)
    
    # showing image
    plt.imshow(output, cmap='gray')
    plt.axis('off')
    plt.title("Second")

    plt.show()
    return output

def gaussianKernel(size, sigma=1, verbose=False):
    kernel_1D = np.linspace(-(size//2), size//2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
 
    kernel_2D *= 1.0 / kernel_2D.max()
    if verbose:
        plt.imshow(kernel_2D, interpolation='none',cmap='gray')
        plt.title("{}X{} Kernel Image".format(size, size))
        plt.show()
 
    return kernel_2D

def gaussianBlurEdgeButtonClicked(verbose):
    print("gaussianBlurButtonClicked")
    img = cv2.imread('.\Dataset_OpenCvDl_Hw1\Q3_Image\House.jpg')
    # 3x3 kernel image
    kernel = gaussianKernel(3, sigma=math.sqrt(3), verbose=verbose)
    return convolution(img, kernel, average=True, verbose=verbose)
    

# 3-2 Sobel X
def bilateralFilterButtonClicked():
    print("bilateralFilterButtonClicked")
    src = cv2.imread('.\Dataset_OpenCvDl_Hw1\Q3_Image\House.jpg')
    dst = cv2.bilateralFilter(src, 9, 90, 90)
    cv2.imshow("Bilateral filter", np.hstack((src, dst)))

# 3-3 Sobel Y
def medianFilterButtonClicked():
    print("medianFilterButtonClicked")
    src = cv2.imread('.\Dataset_OpenCvDl_Hw1\Q3_Image\House.jpg')
    dst1 = cv2.medianBlur(src, 3)
    dst2 = cv2.medianBlur(src, 5)
    cv2.imshow("3*3 Median filter", dst1)
    cv2.imshow("5*5 Median filter", dst2)

# 3-4 Magnitude