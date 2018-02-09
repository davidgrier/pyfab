#!/usr/bin/env python

"""QCameraDevice.py: pyqtgraph module for OpenCV video camera."""

import cv2
from pyqtgraph.Qt import QtCore
import numpy as np


def is_cv2():
    return cv2.__version__.startswith("2.")


class QCameraDevice(QtCore.QObject):
    """OpenCV camera"""

    sigNewFrame = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, cameraId=0, size=None, **kwargs):
        super(QCameraDevice, self).__init__(**kwargs)
        self.camera = cv2.VideoCapture(cameraId)
        self.size = size
        _, self.frame = self.camera.read()
        self.running = False

    def loop(self):
        while self.running:
            ready, self.frame = self.camera.read()
            if ready:
                self.sigNewFrame.emit(self.frame)

    @QtCore.pyqtSlot()
    def start(self):
        if not self.running:
            self.running = True
            self.loop()

    @QtCore.pyqtSlot()
    def stop(self):
        self.running = False

    @QtCore.pyqtSlot(bool)
    def pause(self, state):
        if state:
            self.stop()
        else:
            self.start()

    @QtCore.pyqtSlot()
    def close(self):
        self.stop()
        # self.camera.release() # triggers V4L2 error

    @property
    def size(self):
        if is_cv2():
            h = int(self.camera.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
            w = int(self.camera.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        else:
            h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        return QtCore.QSize(w, h)

    @size.setter
    def size(self, size):
        if size is None:
            return
        if is_cv2():
            self.camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, size[1])
            self.camera.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, size[0])
        else:
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])

    @property
    def roi(self):
        return QtCore.QRectF(0., 0., self.size.width(), self.size.height())
