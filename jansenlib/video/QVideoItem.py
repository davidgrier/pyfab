# -*- coding: utf-8 -*-

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
        self._pause = False

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
    sigStop = QtCore.pyqtSignal()

    def __init__(self, parent=None,
                 source=None,
                 mirrored=False,
                 flipped=True,
                 transposed=False,
                 gray=False,
                 **kwargs):
        pg.setConfigOptions(imageAxisOrder='row-major')
        super(QVideoItem, self).__init__(**kwargs)
        self.parent = parent
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

        self.kwargs = kwargs
        self.source = self.defaultSource()

    def defaultSource(self):
        kwargs = self.kwargs
        return QCameraDevice(**kwargs)

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, source):
        """provide means to change video sources, including
        alternative cameras and video files."""

        # stop existing sources
        self.sigStop.emit()

        # disconnect existing source
        try:
            self._source.sigNewFrame.disconnect(self.updateImage)
        except AttributeError:
            pass

        # connect signals for new source
        source.sigNewFrame.connect(self.updateImage)
        self.sigPause.connect(source.pause)
        self.sigStop.connect(source.stop)
        self._width = source.width
        self._height = source.height

        # move source to background thread to reduce latency
        self.thread = QtCore.QThread()
        self.thread.start()
        source.moveToThread(self.thread)
        self.thread.started.connect(source.start)
        self.thread.finished.connect(self.cleanup)

        self._source = source

    def close(self):
        """Stopping the video source causes the thread to
        emit its finished() signal, which triggers cleanup()."""
        self.sigStop.emit()

    def cleanup(self):
        self.thread.quit()
        self.thread.wait()
        self.thread = None
        self.source = None

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

    @QtCore.pyqtSlot()
    def pause(self, paused=None):
        """sigPause can be caught by video source to pause
        image stream."""
        if paused is None:
            state = not self._pause
        else:
            state = bool(paused)
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
