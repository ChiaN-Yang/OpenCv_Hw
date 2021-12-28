import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QLineEdit
import cv2
import numpy as np

def clicked_1_1():
    img = cv2.imread("Q1/Sun.jpg")
    if img is None:
        sys.exit("Could not read the image.")
    cv2.imshow("Hw1-1", img)
    height, width = img.shape[:2]
    print("Height : ", height)
    print("Width : ", width)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def clicked_1_2():
    img = cv2.imread("Q1/Sun.jpg")
    if img is None:
        sys.exit("Could not read the image.")
    cv2.imshow("HW1-2", img)
    b, g, r = cv2.split(img)
    zeros = np.zeros(b.shape, np.uint8)
    b_img = cv2.merge((b,zeros,zeros))
    g_img = cv2.merge((zeros,g,zeros))
    r_img = cv2.merge((zeros,zeros,r))
    cv2.imshow("Blue", b_img)
    cv2.imshow("Green", g_img)
    cv2.imshow("Red", r_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def clicked_1_3():
    img = cv2.imread("Q1/Sun.jpg") 
    if img is None:
        sys.exit("Could not read the image.")
    cv2.imshow("HW1-3", img)
    b, g, r = cv2.split(img)
    avg = np.uint8(b/3+g/3+r/3)
    merged = cv2.merge((avg, avg, avg))
    cv2.imshow('Average_Gray', merged[:, :, 1])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def update(x):
    pass

def clicked_1_4():
    img = cv2.imread("Q1/Dog_Strong.jpg")
    if img is None:
        sys.exit("Could not read the image.")
    cv2.namedWindow("Blend")
    cv2.createTrackbar('Blend', "Blend", 0, 255, update)
    cv2.setTrackbarPos("Blend", "Blend", 127)
    weak_img = cv2.imread("Q1/Dog_Weak.jpg")
    if weak_img is None:
        sys.exit("Could not read the image.")
    alpha = 0.5  # weighting
    while(1):
        dst = cv2.addWeighted(img, 1-alpha, weak_img, alpha, 0)
        cv2.imshow("Blend", dst)
        if(cv2.waitKey(1) != -1):
            break
        alpha = float(cv2.getTrackbarPos('Blend', "Blend")) / float(255)
    cv2.destroyAllWindows()