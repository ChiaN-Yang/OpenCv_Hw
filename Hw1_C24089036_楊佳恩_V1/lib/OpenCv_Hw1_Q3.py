import cv2
import numpy as np
from numpy import uint8
from scipy import signal

def convolve(img, kernel):
    result = signal.convolve2d(img, kernel, boundary = 'symm')
    result = np.abs(result)
    result [ result > 255 ] = 255
    return result

def Q3_1():
    img = cv2.imread("Q3/House.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x, y = np.mgrid[-1:2, -1:2]
    gaussian_filter = np.exp(-(x**2 + y**2))
    gaussian_filter = gaussian_filter / gaussian_filter.sum()
    gaussian_blur = convolve(gray, gaussian_filter)
    gaussian_blur = gaussian_blur.astype(np.uint8)
    cv2.imshow("Gaussian Blur", gaussian_blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("Q3/Gaussian.jpg", gaussian_blur)

def Q3_2():
    blur = cv2.imread("Q3/Gaussian.jpg")
    blur = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    sobel_x_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_x = convolve(blur, sobel_x_filter)
    sobel_x = sobel_x.astype(np.uint8)
    cv2.imshow("Sobel X", sobel_x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Q3_3():
    blur = cv2.imread("Q3/Gaussian.jpg")
    blur = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    sobel_y_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_y = convolve(blur, sobel_y_filter)
    sobel_y = sobel_y.astype(np.uint8)
    cv2.imshow("Sobel Y", sobel_y)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Q3_4():
    blur = cv2.imread("Q3/Gaussian.jpg")
    blur = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    sobel_x_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_x = convolve(blur, sobel_x_filter)
    sobel_y = convolve(blur, sobel_y_filter)
    sobel = (sobel_x **2 + sobel_y **2) ** 0.5
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX)
    sobel = sobel.astype(np.uint8)
    cv2.imshow("Sobel", sobel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()