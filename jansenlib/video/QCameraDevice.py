"""QCameraDevice.py: pyqtgraph module for OpenCV video camera."""

import cv2
from pyqtgraph.Qt import QtCore
import numpy as np


class QCameraDevice(QtCore.QObject):
    """OpenCV camera

    Continuously captures frames from a video camera,
    emitting sigNewFrame when each frame becomes available.
    """

    sigNewFrame = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, cameraId=0, size=None):
        super(QCameraDevice, self).__init__()
        self.camera = cv2.VideoCapture(cameraId)

        if cv2.__version__.startswith('2.'):
            self._WIDTH = cv2.cv.CV_CAP_PROP_FRAME_WIDTH
            self._HEIGHT = cv2.cv.CV_CAP_PROP_FRAME_HEIGHT
        else:
            self._WIDTH = cv2.CAP_PROP_FRAME_WIDTH
            self._HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT
        self.size = size

        self.running = False
        self.emitting = True
        _, self.frame = self.camera.read()

    def run(self):
        while self.running:
            ready, self.frame = self.camera.read()
            if ready and self.emitting:
                self.sigNewFrame.emit(self.frame)
        self.camera.release()

    @QtCore.pyqtSlot()
    def start(self):
        if not self.running:
            self.running = True
            self.run()

    @QtCore.pyqtSlot()
    def stop(self):
        self.running = False

    @QtCore.pyqtSlot(bool)
    def pause(self, paused):
        self.emitting = not paused

    @property
    def width(self):
        return int(self.camera.get(self._WIDTH))

    @width.setter
    def width(self, width):
        self.camera.set(self._WIDTH, width)

    @property
    def height(self):
        return int(self.camera.get(self._HEIGHT))

    @height.setter
    def height(self, height):
        self.camera.set(self._HEIGHT, height)

    @property
    def size(self):
        return QtCore.QSize(self.width, self.height)

    @size.setter
    def size(self, size):
        if size is None:
            return
        self.width = size[0]
        self.height = size[1]

    @property
    def roi(self):
        return QtCore.QRectF(0., 0., self.width, self.height)
