# -*- coding: utf-8 -*-

"""QVideoItem.py: pyqtgraph module for OpenCV video camera."""

import PyQt5
from PyQt5.QtCore import (QObject, QTime,
                          pyqtSignal, pyqtSlot, pyqtProperty)
import pyqtgraph as pg

from QOpenCV.QOpenCV import QOpenCV
import numpy as np
from collections import deque


class QFPS(QObject):

    def __init__(self, depth=24):
        super(QFPS, self).__init__()
        self.fifo = deque()
        self.depth = depth
        self._fps = 24.
        self._pause = False

    @pyqtSlot(np.ndarray)
    def update(self, image):
        now = QTime.currentTime()
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

    sigNewFrame = pyqtSignal(np.ndarray)

    def __init__(self,
                 parent=None,
                 camera=None,
                 **kwargs):
        pg.setConfigOptions(imageAxisOrder='row-major')
        super(QVideoItem, self).__init__(parent=parent, **kwargs)
        self._filters = list()

        # performance metrics
        self._fps = QFPS()
        self.sigNewFrame.connect(self._fps.update)
        self.fps = self._fps.value

        # default source is a camera
        if camera is None:
            self.camera = QOpenCV(**kwargs)
        else:
            camera.setParent(parent)
            self.camera = camera
        self.source = self.camera
        self.camera.start()

    @pyqtProperty(object)
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

    def gray(self):
        return self.source.gray

    def close(self):
        self.camera.stop()
        self.camera.quit()
        self.camera.wait()

    def closeEvent(self):
        print('closeEvent')
        self.close()

    @pyqtSlot(np.ndarray)
    def updateImage(self, image):
        self.source.blockSignals(True)
        for filter in self._filters:
            image = filter(image)
        self.gray = (image.ndim == 2)
        self.source.blockSignals(False)
        self.setImage(image, autoLevels=False)
        self.sigNewFrame.emit(image)

    def registerFilter(self, filter):
        self._filters.append(filter)

    def unregisterFilter(self, filter):
        if filter in self._filters:
            self._filters.remove(filter)
