from PyQt5 import QtWidgets, uic
import sys
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from transforms import random_erasing


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Load the UI Page
        uic.loadUi('./doc/opencvdl_hw2_5.ui', self)
        self.pushButton_1.clicked.connect(self.showModelShortcut)
        self.pushButton_2.clicked.connect(self.showTensorBoard)
        self.pushButton_3.clicked.connect(self.test)
        self.pushButton_4.clicked.connect(self.dataAugmantation)

    def showModelShortcut(self):
        model = ResNet50(input_shape=(224, 224, 3),
                         include_top=False,
                         weights='imagenet')
        model.summary()

    def showTensorBoard(self):
        img = cv2.imread("./doc/tensorboard.png")
        cv2.namedWindow('training loss and accuracy', cv2.WINDOW_NORMAL)
        cv2.imshow('training loss and accuracy', img)

    def test(self):
        x = int(self.lineEdit.text())
        img_path = f'./PetImages/{x}.jpg'
        self.inferenceImage(img_path)

    def inferenceImage(self, path):
        img = load_img(path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        model = load_model('./doc/model-resnet50-final.h5')
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        class_names = ['cat', 'dog']

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )

        plt.imshow(img)
        plt.title(f'Class:{class_names[np.argmax(score)]}')
        plt.axis("off")
        plt.show()

    def dataAugmantation(self):
        x = int(self.lineEdit.text())
        img_path = f'./PetImages/{x}.jpg'

        img = load_img(img_path, target_size=(224, 224))
        img = img_to_array(img)
        img /= 255.
        img = random_erasing(img, 1)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis('off')

        classes = ['Before Random-Erasing', 'After Random-Erasing']
        acc = [0.9923, 0.9961]
        plt.subplot(1, 2, 2)
        plt.bar(classes, acc)
        plt.text(0, acc[0], str(acc[0]), ha='center')
        plt.text(1, acc[1], str(acc[1]), ha='center')
        plt.ylabel("Accuracy")
        plt.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
