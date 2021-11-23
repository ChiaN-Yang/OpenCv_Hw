# Opencvdl 2021 Hw1_1
import numpy as np
import cv2


class ImageProcessing():
    def __init__(self) -> None:
        self.img = cv2.imread("./Dataset_OpenCvDl_Hw1/Q1_Image/Sun.jpg")
        self.dog_strong = cv2.imread(
            "./Dataset_OpenCvDl_Hw1/Q1_Image/Dog_Strong.jpg")
        self.dog_week = cv2.imread(
            "./Dataset_OpenCvDl_Hw1/Q1_Image/Dog_Weak.jpg")

    def loadImage(self):
        print(f'Height : {self.img.shape[0]}\nWidth : {self.img.shape[1]}')
        cv2.namedWindow('Sun', cv2.WINDOW_NORMAL)
        cv2.imshow('Sun', self.img)

    def colorSeperation(self):
        B, G, R = cv2.split(self.img)
        zeros = np.zeros(self.img.shape[:2], dtype="uint8")
        cv2.imshow("BLUE", cv2.merge([B, zeros, zeros]))
        cv2.imshow("GREEN", cv2.merge([zeros, G, zeros]))
        cv2.imshow("RED", cv2.merge([zeros, zeros, R]))

    def colorTransformations(self):
        B, G, R = cv2.split(self.img)
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        img_merge = B/3 + G/3 + R/3
        img_merge = img_merge.astype(np.uint8)
        cv2.imshow('I1', img_gray)
        cv2.imshow('I2', img_merge)

    def blending(self):
        self.blending_off = True
        cv2.namedWindow('blending')
        cv2.createTrackbar('Blend', 'blending', 0, 255, self.blending_update)
        cv2.setTrackbarPos('Blend', 'blending', 0)
        self.dog = self.dog_strong
        while (self.blending_off):
            cv2.imshow('blending', self.dog)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyWindow('blending')
                break

    def blending_update(self, x):
        weak = cv2.getTrackbarPos('Blend', 'blending')
        weak /= 255
        strong = 1 - weak
        self.dog = cv2.addWeighted(
            self.dog_strong, strong, self.dog_week, weak, 0)


if __name__ == '__main__':
    test = ImageProcessing()
    test.colorTransformations()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
