import cv2
import numpy as np


class Question1:
    def drawContour(self):
        self.numberOfRing = []
        for i in range(1, 3):
            image = cv2.imread('Q1/img' + str(i) + '.jpg')
            image = cv2.resize(image,(int(image.shape[1]/2),int(image.shape[0]/2)),interpolation=cv2.INTER_AREA)
            cv2.imshow('img' + str(i) + '.jpg',image)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9,9), 0)
            canny = cv2.Canny(blurred, 40, 70)

            #find all your connected components (white blobs in your image)
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(canny,connectivity=8)

            #connectedComponentswithStats yields every seperated component with information on each of them, such as size
            #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
            sizes = stats[1:, -1]; nb_components = nb_components - 1

            # minimum size of particles we want to keep (number of pixels)
            #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
            min_size = 80

            connected = np.zeros((gray.shape))
            num = 0
            for j in range(0, nb_components):
                if sizes[j] >= min_size:
                    num +=1
                    connected[output == j + 1] = 255
            connected = connected.astype(np.uint8)
            # print(num)
            _, contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
            # print(np.size(cnts))

            self.numberOfRing.append(int(len(contours)/4))

            # print(hierarchy)
            cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
            # isinstance(connected[0][0][0], (int, np.uint))
            cv2.imshow('Result'+str(i), image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    

    def showNumberOfRings(self, label1, label2):
        label1.setText('There are ' + str(self.numberOfRing[0]) + ' rings in img1.jpg')
        label2.setText('There are ' + str(self.numberOfRing[1]) + ' rings in img2.jpg')










