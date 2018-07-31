# -*- coding: utf-8 -*-

"""QCameraThread.py: OpenCV video camera running in a QThread"""

from pyqtgraph.Qt import QtCore
import cv2
import numpy as np

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class QCameraThread(QtCore.QThread):
    """OpenCV camera

    Continuously captures frames from a video camera,
    emitting sigNewFrame when each frame becomes available.

    NOTE: Subclassing QThread is appealing for this application
    because reading frames is blocking and I/O-bound, but not
    computationally expensive.  QThread moves the read operation
    into a separate thread via the overridden run() method
    while other methods and properties remain available in
    the calling thread.  This simplifies getting and setting
    the camera's properties.

    NOTE: This implementation only moves the camera's read()
    method into a separate thread, not the entire camera.
    FIXME: Confirm that this is acceptable practice.
    """

    sigNewFrame = QtCore.pyqtSignal(np.ndarray)

    def __init__(self,
                 parent=None,
                 cameraID=0,
                 size=(640, 480),
                 mirrored=False,
                 flipped=True,
                 transposed=False,
                 gray=False):
        super(QCameraThread, self).__init__(parent)

        self.camera = cv2.VideoCapture(cameraID)
        self.read = self.camera.read

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
        self.size = size
        self.mirrored = bool(mirrored)
        self.flipped = bool(flipped)
        self.transposed = bool(transposed)
        self.gray = bool(gray)

        while True:
            ready, image = self.read()
            if ready:
                break
        self.frame = image

    def run(self):
        self.running = True
        while self.running:
            ready, self.frame = self.read()
            if ready:
                self.sigNewFrame.emit(self.frame)
        self.camera.release()

    def stop(self):
        self.running = False

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
        (self._height, self._width) = image.shape[:2]
        self._frame = image

    @property
    def width(self):
        # width = int(self.camera.get(self._WIDTH))
        return self._width

    @width.setter
    def width(self, width):
        self.camera.set(self._WIDTH, width)
        logger.info('Setting camera width: {}'.format(width))

    @property
    def height(self):
        # height = int(self.camera.get(self._HEIGHT))
        return self._height

    @height.setter
    def height(self, height):
        self.camera.set(self._HEIGHT, height)
        logger.info('Setting camera height: {}'.format(height))

    @property
    def size(self):
        return QtCore.QSize(self.width, self.height)

    @size.setter
    def size(self, size):
        if size is None:
            return
        if isinstance(size, QtCore.QSize):
            self.width = size.width()
            self.height = size.height()
        else:
            (self.width, self.height) = size

    @property
    def roi(self):
        return QtCore.QRectF(0., 0., self.width, self.height)

    @property
    def gray(self):
        return (self._conversion == self._toGRAY)

    @gray.setter
    def gray(self, gray):
        self._conversion = self._toGRAY if gray else self._toRGB
