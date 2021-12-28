import numpy as np
import cv2
import os
import math

def Q4_1():
    image = cv2.imread('Q4/SQUARE-01.png')

    resizedImage = cv2.resize(image, (256, 256))

    cv2.imshow('image', resizedImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('Q4/SQUARE-01-resized.png', resizedImage)


def Q4_2():
    image = cv2.imread('Q4/SQUARE-01-resized.png')

    oldLocation = np.array([[0, 0], [60, 0], [0, 60]], dtype=np.float32)
    newLocation = np.array([[0, 60], [60, 60], [0, 120]], dtype=np.float32)
    affineMatrix = cv2.getAffineTransform(oldLocation, newLocation)

    translatedImage = cv2.warpAffine(image, affineMatrix, (400, 300))

    cv2.imshow('image', translatedImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('Q4/SQUARE-01-translated.png', translatedImage)


def Q4_3():
    image = cv2.imread('Q4/SQUARE-01-translated.png')

    affineMatrix = cv2.getRotationMatrix2D((image.shape[0]/2, image.shape[1]/2), 10, 0.5)
    rotatedImage = cv2.warpAffine(image, affineMatrix, (400, 300))

    cv2.imshow('image', rotatedImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('Q4/SQUARE-01-rotated.png', rotatedImage)


def Q4_4():
    image = cv2.imread('Q4/SQUARE-01-rotated.png')

    oldLocation = np.array([[50, 50], [200, 50], [50, 200]], dtype=np.float32)
    newLocation = np.array([[10, 100], [200, 50], [100, 250]], dtype=np.float32)
    affineMatrix = cv2.getAffineTransform(oldLocation, newLocation)

    shearedImage = cv2.warpAffine(image, affineMatrix, (400, 300))

    cv2.imshow('image', shearedImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('Q4/SQUARE-01-sheared.png', shearedImage)