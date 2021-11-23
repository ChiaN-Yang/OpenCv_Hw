from PyQt5 import QtWidgets, uic
import sys
import lib
from matplotlib.pyplot import close
from cv2 import destroyAllWindows


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Load the UI Page
        uic.loadUi('./doc/opencvdl_hw1.ui', self)
        self.pushButton_1_1.clicked.connect(self.loadImage)
        self.pushButton_1_2.clicked.connect(self.colorSeperation)
        self.pushButton_1_3.clicked.connect(self.colorTransformations)
        self.pushButton_1_4.clicked.connect(self.blending)
        self.pushButton_2_1.clicked.connect(self.gaussianBlur)
        self.pushButton_2_2.clicked.connect(self.bilateralFilter)
        self.pushButton_2_3.clicked.connect(self.medianFilter)
        self.pushButton_3_1.clicked.connect(self.gaussianBlur3)
        self.pushButton_3_2.clicked.connect(self.sobelX)
        self.pushButton_3_3.clicked.connect(self.sobelY)
        self.pushButton_3_4.clicked.connect(self.magnitude)
        self.pushButton_4_1.clicked.connect(self.imgResize)
        self.pushButton_4_2.clicked.connect(self.translation)
        self.pushButton_4_3.clicked.connect(self.rotationScaling)
        self.pushButton_4_4.clicked.connect(self.shearing)
        self.pushButton_5.clicked.connect(self.closeAllWindows)

        # Create object
        self.Q1 = lib.image_processing.ImageProcessing()
        self.Q2 = lib.image_smoothing.ImageSmoothing()
        self.Q3 = lib.edge_detection.EdgeDetection()
        self.Q4 = lib.transforms.Transforms()

    def loadImage(self):
        self.Q1.loadImage()

    def colorSeperation(self):
        self.Q1.colorSeperation()

    def colorTransformations(self):
        self.Q1.colorTransformations()

    def blending(self):
        self.Q1.blending()

    def gaussianBlur(self):
        self.Q2.gaussianBlur()

    def bilateralFilter(self):
        self.Q2.bilateralFilter()

    def medianFilter(self):
        self.Q2.medianFilter()

    def gaussianBlur3(self):
        self.Q3.gaussianBlur()

    def sobelX(self):
        self.Q3.sobelX()

    def sobelY(self):
        self.Q3.sobelY()

    def magnitude(self):
        self.Q3.magnitude()

    def imgResize(self):
        self.Q4.imgResize()

    def translation(self):
        self.Q4.translation()

    def rotationScaling(self):
        self.Q4.rotationScaling()

    def shearing(self):
        self.Q4.shearing()

    def closeAllWindows(self):
        close('all')
        destroyAllWindows()
        self.Q1.blending_off = False


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
