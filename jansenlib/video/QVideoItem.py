"""QVideoItem.py: pyqtgraph module for OpenCV video camera."""

import cv2
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import numpy as np
from QCameraDevice import QCameraDevice


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

        # run video source in thread to reduce latency
        self.fps = 0.
        self._time = QtCore.QTime.currentTime()
        self.device = QCameraDevice(**kwargs)
        self.device.sigNewFrame.connect(self.updateImage)
        self.sigPause.connect(self.device.pause)
        self.camThread = QtCore.QThread()
        self.camThread.start()
        self.device.moveToThread(self.camThread)
        self.camThread.started.connect(self.device.start)
        self.camThread.finished.connect(self.device.stop)

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

    def close(self):
        self.camThread.quit()
        self.camThread.wait()
        self.camThread = None

    def closeEvent(self):
        self.close()

    def updateFPS(self):
        """Calculate frames per second."""
        now = QtCore.QTime.currentTime()
        try:
            self.fps = 1000. / (self._time.msecsTo(now))
        except ZeroDivisionError:
            self.fps = 24.
        self._time = now

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
        self.updateFPS()

    def pause(self, state):
        """sigPause can be caught by video source to pause
        image stream."""
        self.emit.sigPause(state)

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
