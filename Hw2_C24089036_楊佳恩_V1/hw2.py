from PyQt5 import QtWidgets, uic
import sys
import libs


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Load the UI Page
        uic.loadUi('./libs/opencvdl_hw2.ui', self)
        # self.pushButton_1_1.clicked.connect(self.drawContour)
        # self.pushButton_1_2.clicked.connect(self.countRings)
        self.pushButton_2_1.clicked.connect(self.findCorner)
        self.pushButton_2_2.clicked.connect(self.findIntrinsic)
        self.pushButton_2_3.clicked.connect(self.findExtrinsic)
        self.pushButton_2_4.clicked.connect(self.findDistortion)
        self.pushButton_2_5.clicked.connect(self.showResult)
        # self.pushButton_3_1.clicked.connect(self.showWordsonBoard)
        # self.pushButton_3_2.clicked.connect(self.showWordsVertically)
        # self.pushButton_4_1.clicked.connect(self.stereoDisparityMap)

        # Create object
        # self.Q1 = libs.find_contour.FindContour()
        self.Q2 = libs.camera_calibration.CameraCalibration()
        # self.Q3 = libs.augmented_reality.AugmentedReality()
        # self.Q4 = libs.stereo_disparity_map.StereoDisparityMap()

    # def loadImage(self):
    #     self.Q1.loadImage()

    # def colorSeperation(self):
    #     self.Q1.colorSeperation()

    def findCorner(self):
        self.Q2.findCorner()

    def findIntrinsic(self):
        self.Q2.findIntrinsic()

    def findExtrinsic(self):
        n = int(self.lineEdit_2.text())
        self.Q2.findExtrinsic(n-1)

    def findDistortion(self):
        self.Q2.findDistortion()

    def showResult(self):
        self.Q2.showResult()

    # def gaussianBlur3(self):
    #     self.Q3.gaussianBlur()

    # def sobelX(self):
    #     self.Q3.sobelX()

    # def sobelY(self):
    #     self.Q3.sobelY()

    # def magnitude(self):
    #     self.Q3.magnitude()

    # def imgResize(self):
    #     self.Q4.imgResize()

    # def translation(self):
    #     self.Q4.translation()

    # def rotationScaling(self):
    #     self.Q4.rotationScaling()

    # def shearing(self):
    #     self.Q4.shearing()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
