"""QVideoItem.py: pyqtgraph module for OpenCV video camera."""

import cv2
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import numpy as np
from QCameraDevice import QCameraDevice
from collections import deque


class QFPS(QtCore.QObject):

    def __init__(self, depth=24):
        super(QFPS, self).__init__()
        self.fifo = deque()
        self.depth = depth
        self._fps = 24.

    @QtCore.pyqtSlot(np.ndarray)
    def update(self, image):
        now = QtCore.QTime.currentTime()
        self.fifo.appendleft(now)
        if len(self.fifo) <= self.depth:
            return
        then = self.fifo.pop()
        self._fps = 1000. * self.depth / then.msecsTo(now)

    def value(self):
        return self._fps


class QVideoItem(pg.ImageItem):
    """Video source for pyqtgraph applications.
    Acts like a pyqtgraph ImageItem whose images are updated
    automatically.  Optionally applies filters to modify
    images obtained from source.
    """

    sigNewFrame = QtCore.pyqtSignal(np.ndarray)
    sigPause = QtCore.pyqtSignal(bool)

    def __init__(self,
                 mirrored=False,
                 flipped=True,
                 transposed=False,
                 gray=False,
                 **kwargs):
        pg.setConfigOptions(imageAxisOrder='row-major')
        super(QVideoItem, self).__init__(**kwargs)

        # image source
        self.source = QCameraDevice(**kwargs)
        self.source.sigNewFrame.connect(self.updateImage)
        self.sigPause.connect(self.source.pause)
        self._width = self.source.width
        self._height = self.source.height

        # run source in thread to reduce latency
        self.thread = QtCore.QThread()
        self.thread.start()
        self.source.moveToThread(self.thread)
        self.thread.started.connect(self.source.start)
        self.thread.finished.connect(self.source.close)

        # image conversions
        self._conversion = None
        if cv2.__version__.startswith('2.'):
            self._toRGB = cv2.cv.CV_BGR2RGB
            self._toGRAY = cv2.cv.CV_BGR2GRAY
        else:
            self._toRGB = cv2.COLOR_BGR2RGB
            self._toGRAY = cv2.COLOR_BGR2GRAY
        self.gray = bool(gray)
        self.mirrored = bool(mirrored)
        self.flipped = bool(flipped)
        self.transposed = bool(transposed)
        self._filters = list()

        # performance metrics
        self._fps = QFPS()
        self.sigNewFrame.connect(self._fps.update)
        self.fps = self._fps.value

    def close(self):
        self.source.close()
        self.thread.quit()
        self.thread.wait()
        self.thread = None

    def closeEvent(self):
        self.close()

    @QtCore.pyqtSlot(np.ndarray)
    def updateImage(self, image):
        if image.ndim == 3:
            image = cv2.cvtColor(image, self._conversion)
        if self.transposed:
            image = cv2.transpose(image)
        if self.flipped or self.mirrored:
            image = cv2.flip(image, self.mirrored * (1 - 2 * self.flipped))
        for filter in self._filters:
            image = filter(image)
        self.setImage(image, autoLevels=False)
        self.sigNewFrame.emit(image)

    def pause(self, state):
        """sigPause can be caught by video source to pause
        image stream."""
        self.emit.sigPause(state)

    def width(self):
        return self._width

    def height(self):
        return self._height

    @property
    def gray(self):
        return (self._conversion == self._toGRAY)

    @gray.setter
    def gray(self, gray):
        self._conversion = self._toGRAY if gray else self._toRGB

    def registerFilter(self, filter):
        self._filters.append(filter)

    def unregisterFilter(self, filter):
        if filter in self._filters:
            self._filters.remove(filter)
