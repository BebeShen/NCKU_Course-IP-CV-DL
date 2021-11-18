# Homework 1

作業1分成5大部分

1. [Image Processing](#image-processing)
2. [Image Smoothing](#image-smoothing)
3. [Edge Detection](#edge-detection)
4. [Transforms](#transforms)
5. [Training Cifar-10 Classifier Using VGG16](#training-cifar-10-classifier-using-vgg16)

## Environment

* Python 3.8.2
* opencv-python 4.5.4.58
* numpy 1.18.2
* PyQt5 5.15.6

## Image Processing

>Image Processing分為4小題

### **Load Image File**

作業要求：

1. 開啟新視窗呈現給予之圖片
2. console出圖片之長寬

### **Color Separation**

作業要求：

1. 將給予之圖片分別以RGB三色呈現

### **Color Transformation**

作業要求：

1. 將給予之圖片透過opencv函式以灰階圖片呈現
2. 將1-2三個不同channels以`(R+B+G)/3`的方式合併成一張灰階圖片

### **Blending**

作業要求：

1. 將給予之兩張圖片透過opencv函式以`Blending`的方式呈現
2. 利用`trackbar`調整blending的值

---

## Image Smoothing

>Image Smoothing分為3小題

### **Gaussian Blur**

作業要求：

1. 利用Gaussian Blur進行5x5的影像平滑化

### **Bilateral Filter**

作業要求：

1. 利用Bilateral Filter進行9x9的影像平滑化

### **Median Filter**

作業要求：

1. 利用Median Filter進行3x3和5x5的影像平滑化

---

## Edge Detection

>Edge Detection分為4小題

### **Gaussian Blur(self defined function)**

作業要求：

1. 利用Gaussian Blur的方式進行Image Smooth
2. **不能使用opencv之function**

### **Sobel X**

作業要求：

1. 利用Sobel Operation(vertical edge)的方式進行3x3的Edge Detection
2. **不能使用opencv之function**

### **Sobel Y**

作業要求：

1. 利用Sobel Operation(horizontal edge)的方式進行3x3的Edge Detection
2. **不能使用opencv之function**

### **Magnitude**

作業要求：

1. 開啟新視窗呈現給予之圖片
2. console出圖片之長寬

---

## Transforms

>Transforms分為4小題

---

## Training Cifar-10 Classifier Using VGG16

---

## Usage

1. Pre-install PyQt & OpenCV

    ```shell
    pip install PyQt5 opencv-python numpy
    ```

2. Compile `.ui` after designing GUI by using Qt Designer

    ```shell
    pyuic5 -x <ui file> -o <.py file>
    ```

3. Run python script

    ```shell
    python <.py file>
    ```

---

## 參考資料

* [PyQt使用](https://www.wongwonggoods.com/python/pyqt5-2/)
* [Python 與 OpenCV 基本讀取、顯示與儲存圖片教學](https://blog.gtwang.org/programming/opencv-basic-image-read-and-write-tutorial/)
* [運用 OpenCV 顯示圖片直方圖、分離與合併RGB通道](https://www.wongwonggoods.com/python/python_opencv/opencv-histogram-split-merge-rgb-channel/)
* [How to Convert an RGB Image to Grayscale](https://e2eml.school/convert_rgb_to_grayscale.html)
* [Python OpenCV Tutorial Part 3 : RGB to Gray Conversion of an Image](https://www.youtube.com/watch?v=TfVW1iFfmto)
* [OpenCV 從零開始的影像處理](https://ithelp.ithome.com.tw/users/20126965/ironman/3364)
* [OpenCV Python Image Smoothing – Gaussian Blur](https://www.tutorialkart.com/opencv/python/opencv-python-gaussian-image-smoothing/)
* [Bilateral Filtering in Python OpenCV with cv2.bilateralFilter()](https://machinelearningknowledge.ai/bilateral-filtering-in-python-opencv-with-cv2-bilateralfilter/)
* [python-與-opencv-模糊處理](https://chtseng.wordpress.com/2016/11/17/python-%E8%88%87-opencv-%E6%A8%A1%E7%B3%8A%E8%99%95%E7%90%86/)
* [What is a mathematical relation of diameter and sigma arguments in bilateral filter function?](https://stackoverflow.com/questions/59505866/what-is-a-mathematical-relation-of-diameter-and-sigma-arguments-in-bilateral-fil)
* [[Python]Gaussian Filter-概念與實作](https://medium.com/@bob800530/python-gaussian-filter-%E6%A6%82%E5%BF%B5%E8%88%87%E5%AF%A6%E4%BD%9C-676aac52ea17)
* [Applying Gaussian Smoothing to an Image using Python from scratch](Applying Gaussian Smoothing to an Image using Python from scratch)
* [How to implement Sobel edge detection using Python from scratch](http://www.adeveloperdiary.com/data-science/computer-vision/how-to-implement-sobel-edge-detection-using-python-from-scratch/)
