# hw2_1
# https://blog.csdn.net/u010128736/article/details/52875137
import cv2
import numpy as np
import copy


class CameraCalibration():
    def __init__(self) -> None:
        # 棋盤格模板規格
        self.w = 11
        self.h = 8
        self.images = []

        # findIntrinsic
        # 儲存棋盤格角點的世界坐標和圖像坐標對
        objpoints = []  # 在世界坐標系中的三維點
        imgpoints = []  # 在圖像平面的二維點

        # 世界坐標系中的棋盤格點,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐標，記為二維矩陣
        objp = np.zeros((self.w*self.h, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.w, 0:self.h].T.reshape(-1, 2)

        for fname in range(15):
            img = cv2.imread(f"./Dataset_OpenCvDl_Hw2/Q2_Image/{fname+1}.bmp")
            self.images.append(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 找到棋盤格角點
            _, corners = cv2.findChessboardCorners(
                gray, (self.w, self.h), None)
            objpoints.append(objp)
            imgpoints.append(corners)

        # 標定
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)

    def findCorner(self):
        images = copy.deepcopy(self.images)
        # 找棋盤格角點
        # 閾值
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 找到棋盤格角點
            ret, corners = cv2.findChessboardCorners(
                gray, (self.w, self.h), None)
            # 如果找到足夠點對，將其存儲起來
            if ret == True:
                cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                # 將角點在圖像上顯示
                cv2.drawChessboardCorners(img, (self.w, self.h), corners, ret)
                cv2.namedWindow(f'find corners', 0)
                cv2.resizeWindow(f'find corners', 500, 500)
                cv2.moveWindow(f'find corners', 300, 100)
                cv2.imshow(f'find corners', img)
                cv2.waitKey(500)

    def findIntrinsic(self):
        print(f'\nIntrinsic:\n{self.mtx}')

    def findExtrinsic(self, n):
        try:
            img = self.images[n]
        except AttributeError:
            print("\nFailed to access the number")
            img = self.images[1]

        objpoints = []
        imgpoints = []
        objp = np.zeros((self.w*self.h, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.w, 0:self.h].T.reshape(-1, 2)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, corners = cv2.findChessboardCorners(
            gray, (self.w, self.h), None)
        objpoints.append(objp)
        imgpoints.append(corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
        rvecs = cv2.Rodrigues(rvecs[0])

        etmat = rvecs[0]
        etmat = np.insert(etmat, 3, values=tvecs[0].reshape(-1), axis=1)
        print(f'\nExtrinsic(img_{n}):\n{etmat}')

    def findDistortion(self):
        print(f'\nDistortion:\n{self.dist}')

    def showResult(self):
        for img in self.images:
            h,  w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                self.mtx, self.dist, (w, h), 0, (w, h))  # 自由比例参数
            dst = cv2.undistort(img, self.mtx, self.dist, None, newcameramtx)
            imgs = np.hstack([img, dst])
            cv2.namedWindow(f'undistorted  result', 0)
            cv2.resizeWindow(f'undistorted  result', 1000, 500)
            cv2.moveWindow(f'undistorted  result', 150, 50)
            cv2.imshow(f'undistorted  result', imgs)
            cv2.waitKey(500)


if __name__ == "__main__":
    cc = CameraCalibration()
    cc.findIntrinsic()
    # cc.showResult()
