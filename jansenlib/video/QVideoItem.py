# -*- coding: utf-8 -*-

"""QVideoItem.py: pyqtgraph module for OpenCV video camera."""

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import numpy as np
from .QCameraThread import QCameraThread
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
        try:
            self._fps = 1000. * self.depth / then.msecsTo(now)
        except ZeroDivisionError:
            pass

    def value(self):
        return self._fps


class QVideoItem(pg.ImageItem):
    """Video source for pyqtgraph applications.
    Acts like a pyqtgraph ImageItem whose images are updated
    automatically.  Optionally applies filters to modify
    images obtained from source.
    """

    sigNewFrame = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, parent=None,
                 source=None,
                 **kwargs):
        pg.setConfigOptions(imageAxisOrder='row-major')
        super(QVideoItem, self).__init__(parent=parent, **kwargs)
        self._filters = list()

        # performance metrics
        self._fps = QFPS()
        self.sigNewFrame.connect(self._fps.update)
        self.fps = self._fps.value

        # default source is a camera
        self.camera = QCameraThread(parent=self, **kwargs)
        self.source = self.camera
        self.camera.start()

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, source):
        try:
            self.source.sigNewFrame.disconnect(self.updateImage)
        except AttributeError:
            pass
        if source is None:
            source = self.camera
        source.sigNewFrame.connect(self.updateImage)
        self._source = source

    def close(self):
        self.camera.stop()
        self.camera.quit()
        self.camera.wait()

    def closeEvent(self):
        print('closeEvent')
        self.close()

    @QtCore.pyqtSlot(np.ndarray)
    def updateImage(self, image):
        self.source.blockSignals(True)
        for filter in self._filters:
            image = filter(image)
        self.source.blockSignals(False)
        self.setImage(image, autoLevels=False)
        self.sigNewFrame.emit(image)

    def registerFilter(self, filter):
        self._filters.append(filter)

    def unregisterFilter(self, filter):
        if filter in self._filters:
            self._filters.remove(filter)
