# -*- coding: utf-8 -*-

"""QCameraDevice.py: pyqtgraph module for OpenCV video camera."""

import cv2
from pyqtgraph.Qt import QtCore
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class QCameraDevice(QtCore.QObject):
    """OpenCV camera

    Continuously captures frames from a video camera,
    emitting sigNewFrame when each frame becomes available.
    """

    sigNewFrame = QtCore.pyqtSignal(np.ndarray)

    def __init__(self,
                 cameraID=0,
                 size=None,
                 mirrored=False,
                 flipped=True,
                 transposed=False,
                 gray=False):
        super(QCameraDevice, self).__init__()

        self.camera = cv2.VideoCapture(cameraID)

        if cv2.__version__.startswith('2.'):
            self._WIDTH = cv2.cv.CV_CAP_PROP_FRAME_WIDTH
            self._HEIGHT = cv2.cv.CV_CAP_PROP_FRAME_HEIGHT
            self._toRGB = cv2.cv.CV_BGR2RGB
            self._toGRAY = cv2.cv.CV_BGR2GRAY
        else:
            self._WIDTH = cv2.CAP_PROP_FRAME_WIDTH
            self._HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT
            self._toRGB = cv2.COLOR_BGR2RGB
            self._toGRAY = cv2.COLOR_BGR2GRAY

        # camera properties
        self.defaultSize = self.size
        print(self.defaultSize)
        self.size = size
        print(self.size)
        self.mirrored = bool(mirrored)
        self.flipped = bool(flipped)
        self.transposed = bool(transposed)
        self.gray = bool(gray)

        self.running = False
        self.emitting = False

        while True:
            ready, image = self.camera.read()
            if ready:
                break
        self.frame = image

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
            self.emitting = True
            self.run()

    @QtCore.pyqtSlot()
    def stop(self):
        self.running = False

    @QtCore.pyqtSlot(bool)
    def pause(self, paused):
        self.emitting = not paused

    @property
    def frame(self):
        return self._frame

    @frame.setter
    def frame(self, image):
        if image.ndim == 3:
            image = cv2.cvtColor(image, self._conversion)
        if self.transposed:
            image = cv2.transpose(image)
        if self.flipped or self.mirrored:
            image = cv2.flip(image, self.mirrored * (1 - 2 * self.flipped))
        self._frame = image

    @property
    def width(self):
        return 1280 # int(self.camera.get(self._WIDTH))

    @width.setter
    def width(self, width):
        self.camera.set(self._WIDTH, width)
        logger.info('Setting camera width: %d (Default: %d)',
                    width, self.defaultSize.width())

    @property
    def height(self):
        return 1024 # int(self.camera.get(self._HEIGHT))

    @height.setter
    def height(self, height):
        self.camera.set(self._HEIGHT, height)
        logger.info('Setting camera height: %d (Default: %d)',
                    height, self.defaultSize.height())

    @property
    def size(self):
        return QtCore.QSize(self.width, self.height)

    @size.setter
    def size(self, size):
        if size is None:
            return
        if isinstance(size, QtCore.QSize):
            self.width = size.width
            self.height = size.height
        else:
            self.width = size[0]
            self.height = size[1]

    @property
    def roi(self):
        return QtCore.QRectF(0., 0., self.width, self.height)

    @property
    def gray(self):
        return (self._conversion == self._toGRAY)

    @gray.setter
    def gray(self, gray):
        self._conversion = self._toGRAY if gray else self._toRGB
