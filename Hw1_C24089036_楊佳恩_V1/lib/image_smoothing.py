# Opencvdl 2021 Hw1_2
import cv2


class ImageSmoothing():
    def __init__(self) -> None:
        self.lenna = cv2.imread(
            "./Dataset_OpenCvDl_Hw1/Q2_Image/Lenna_whiteNoise.jpg")
        self.pepper = cv2.imread(
            "./Dataset_OpenCvDl_Hw1/Q2_Image/Lenna_pepperSalt.jpg")

    def gaussianBlur(self):
        blur = cv2.GaussianBlur(self.lenna, (5, 5), 0)
        cv2.imshow('Origin image', self.lenna)
        cv2.imshow('Gaussian Blur', blur)

    def bilateralFilter(self):
        bilateral = cv2.bilateralFilter(self.lenna, 9, 90, 90)
        cv2.imshow('Origin image', self.lenna)
        cv2.imshow('Bilateral Filter', bilateral)

    def medianFilter(self):
        median3 = cv2.medianBlur(self.pepper, 3)
        median5 = cv2.medianBlur(self.pepper, 5)
        cv2.imshow('Origin image', self.pepper)
        cv2.imshow('Median Filter 3x3', median3)
        cv2.imshow('Median Filter 5x5', median5)


if __name__ == '__main__':
    test = ImageSmoothing()
    test.medianFilter()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
