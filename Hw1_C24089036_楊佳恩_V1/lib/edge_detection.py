# Opencvdl 2021 Hw1_3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class EdgeDetection():
    def __init__(self) -> None:
        self.house = mpimg.imread("./Dataset_OpenCvDl_Hw1/Q3_Image/House.jpg")

    def gaussianBlur(self):
        # origin image
        plt.subplot(1, 3, 1)
        plt.imshow(self.house, cmap=plt.get_cmap('gray'))
        plt.axis('off')
        plt.title('House.jpg')

        # Creat 3*3 Gassian Kernel
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x**2+y**2))

        # Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

        # Convolve 3*3 Gassian Kernel with Image
        gray = self.grayscale(self.house)
        plt.subplot(1, 3, 2)
        plt.imshow(gray, cmap=plt.get_cmap('gray'))
        plt.axis('off')
        plt.title('Grayscale')

        self.gaussian_blur = self.convolution(gray, gaussian_kernel)
        plt.subplot(1, 3, 3)
        plt.imshow(self.gaussian_blur, cmap=plt.get_cmap('gray'))
        plt.axis('off')
        plt.title('Gaussian Blur')
        plt.show()

    def sobelX(self):
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        house_x = self.convolution(self.gaussian_blur, sobel_x)
        self.house_x = self.convertAbs(house_x)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(self.gaussian_blur, cmap=plt.get_cmap('gray'))
        plt.axis('off')
        plt.title('Grayscale')

        plt.subplot(1, 2, 2)
        plt.imshow(self.house_x, cmap=plt.get_cmap('gray'))
        plt.axis('off')
        plt.title('Sobel X')
        plt.show()

    def sobelY(self):
        sobel_y = np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]])
        house_y = self.convolution(self.gaussian_blur, sobel_y)
        self.house_y = self.convertAbs(house_y)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(self.gaussian_blur, cmap=plt.get_cmap('gray'))
        plt.axis('off')
        plt.title('Grayscale')

        plt.subplot(1, 2, 2)
        plt.imshow(self.house_y, cmap=plt.get_cmap('gray'))
        plt.axis('off')
        plt.title('Sobel Y')
        plt.show()

    def magnitude(self):
        mag = self.addXandY(self.house_x, self.house_y)
        plt.figure()
        plt.imshow(mag, cmap=plt.get_cmap('gray'))
        plt.axis('off')
        plt.title('Magnitude')
        plt.show()

    def grayscale(self, data):
        """ 灰階 """
        b, g, r = self.split(data)
        img_gray = b*0.299 + g*0.587 + r*0.114
        img_gray = img_gray.round()
        img_gray = img_gray.astype(np.uint8)
        return img_gray

    def split(self, img):
        b = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
        g = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
        r = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
        b[:, :] = img[:, :, 0]
        g[:, :] = img[:, :, 1]
        r[:, :] = img[:, :, 2]
        return b, g, r

    def convolution(self, data, k):
        """ 卷積 """
        n, m = data.shape
        img_new = []
        for i in range(n-3):
            line = []
            for j in range(m-3):
                a = data[i:i+3, j:j+3]
                line.append(np.sum(np.multiply(k, a)))
            img_new.append(line)
        return np.array(img_new)

    def convertAbs(self, data):
        """ 取絕對值 """
        data = np.abs(data)
        data.round()
        data = data.astype(np.uint8)
        return data

    def addXandY(self, sob_x, sob_y):
        img_x = sob_x.astype(np.int32)
        img_y = sob_y.astype(np.int32)
        img = np.sqrt(img_x**2 + img_y**2).round()
        img = img.astype(np.uint8)
        return img


if __name__ == '__main__':
    test = EdgeDetection()
    test.gaussianBlur()
    test.sobelX()
    test.sobelY()
    test.magnitude()
