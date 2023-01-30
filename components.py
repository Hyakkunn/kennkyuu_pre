import numpy as np
import cv2

class Component:
    def __init__(self, pixs, label):
        self.pixs = pixs
        self.label = label
        self.index = label-1
        self.num_pixs = len(self.pixs[0])
        self.connected_components = []

    #def get_moment(self, size=(400, 400)):
    def get_moment(self, size=(401, 401)):
        img = np.zeros(size)
        img[self.pixs[0], self.pixs[1]] = 1
        mu = cv2.moments(img, False)
        x, y = int(mu["m10"] / mu["m00"]), int(mu["m01"] / mu["m00"])
        return x, y

    def __str__(self):
        return "label: {}, index{}".format(self.label, self.index)

