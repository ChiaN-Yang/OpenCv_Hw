# Opencvdl 2021 Hw1_4
import cv2
import numpy as np


class Transforms():
    def __init__(self) -> None:
        self.square = cv2.imread(
            "./Dataset_OpenCvDl_Hw1/Q4_Image/SQUARE-01.png")
        self.window_size = (400, 300)

    def imgResize(self):
        self.img_resize = cv2.resize(self.square, (256, 256))
        cv2.imshow('resize', self.img_resize)

    def translation(self):
        M = np.float32([[1, 0, 0], [0, 1, 60]])
        self.img_trans = cv2.warpAffine(self.img_resize, M, self.window_size)
        cv2.imshow('translation', self.img_trans)

    def rotationScaling(self):
        M = cv2.getRotationMatrix2D((128, 188), 10, 0.5)
        self.img_rotation = cv2.warpAffine(self.img_trans, M, self.window_size)
        cv2.imshow('rotation scaling', self.img_rotation)

    def shearing(self):
        old_loc = np.float32([[50, 50], [200, 50], [50, 200]])
        new_loc = np.float32([[10, 100], [200, 50], [100, 250]])
        M = cv2.getAffineTransform(old_loc, new_loc)
        sheared_img = cv2.warpAffine(self.img_rotation, M, self.window_size)
        cv2.imshow('shearing', sheared_img)


if __name__ == '__main__':
    test = Transforms()
    test.shearing()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
