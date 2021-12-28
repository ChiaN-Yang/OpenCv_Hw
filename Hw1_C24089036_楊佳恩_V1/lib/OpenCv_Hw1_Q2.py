import cv2
import random
import numpy as np
def medianFilter(image):
    m1=cv2.medianBlur(image,3)
    m2=cv2.medianBlur(image,5)
    cv2.imshow("Origin image",image)
    cv2.imshow("Median Filter 3x3",m1)
    cv2.imshow("Median Filter 5x5",m2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def gaussianFilter(image):
    gaussian_image=cv2.GaussianBlur(image, (5,5),0)
    cv2.imshow("origin",image)
    cv2.imshow("gaussian_image",gaussian_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def bilateralFilter(image):
    bilateral_image=cv2.bilateralFilter(image,9,90,90)
    cv2.imshow("origin",image)
    cv2.imshow("bilateral_image",bilateral_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def clicked_2_1():
    image=cv2.imread("Q2/Lenna_whiteNoise.jpg")
    gaussianFilter(image)

def clicked_2_2():
    image=cv2.imread("Q2/Lenna_whiteNoise.jpg")
    bilateralFilter(image)

def clicked_2_3():
    image2=cv2.imread("Q2/Lenna_pepperSalt.jpg")
    medianFilter(image2)