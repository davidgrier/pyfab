# -*- coding: utf-8 -*-

"""QJansenScreen.py: PyQt GUI for live video with graphical overlay."""

from PyQt5.QtCore import (pyqtSignal, pyqtSlot, pyqtProperty, QThread, QSize)
from PyQt5.QtGui import (QMouseEvent, QWheelEvent)
import pyqtgraph as pg
import numpy as np
import time

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class QCameraThread(QThread):

    '''Class for reducing latency by moving cameras to separate thread'''

    sigNewFrame = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super(QCameraThread, self).__init__(parent)
        self.camera = self.parent().camera

    def run(self):
        logger.debug('Starting acquisition loop')
        self._running = True
        while self._running:
            ready, frame = self.camera.read()
            if ready:
                self._shape = frame.shape
                self.sigNewFrame.emit(frame)
            else:
                logger.warn('Failed to read frame')
        logger.debug('Stopping acquisition loop')
        self.camera.close()

    def stop(self):
        self._running = False

    @property
    def width(self):
        return self._shape[1]

    @property
    def height(self):
        return self._shape[0]


class FpsMeter(object):
    def __init__(self):
        self.nframes = 10
        self.frame = 0.
        self.start = time.time()
        self._value = 0.

    def tick(self):
        self.now = time.time()
        self.frame = (self.frame + 1) % self.nframes
        return self.frame == 0

    def tock(self):
        self._value = self.nframes / (self.now - self.start)
        self.start = self.now
        return self._value

    @property
    def value(self):
        return self._value


class QJansenScreen(pg.GraphicsLayoutWidget):

    """Interactive display for pyfab system.

    QJansenScreen incorporates a QVideoItem to display live video.
    Additional GraphicsItems can be added to the viewbox
    as overlays over the video stream.
    Interaction with graphical items is facilitated
    by custom signals that correspond to mouse events.

    ...

    Attributes
    ----------
    parent:
    camera: pyfab.jansenlib.video.QVideoItem

    Signals
    -------
    sigMousePress(QMouseEvent)
    sigMouseRelease(QMouseEvent)
    sigMouseMove(QMouseEvent)
    sigMouseWheel(QMouseEvent)
    sigNewFrame(np.ndarray)

    Slots
    -----
    """
    sigMousePress = pyqtSignal(QMouseEvent)
    sigMouseRelease = pyqtSignal(QMouseEvent)
    sigMouseMove = pyqtSignal(QMouseEvent)
    sigMouseWheel = pyqtSignal(QWheelEvent)
    sigNewFrame = pyqtSignal(np.ndarray)
    sigFPS = pyqtSignal(float)

    def __init__(self, parent=None, camera=None, **kwargs):

        pg.setConfigOptions(imageAxisOrder='row-major')

        super(QJansenScreen, self).__init__(parent)

        self.ci.layout.setContentsMargins(0, 0, 0, 0)
        # ImageItem displays video feed
        self.imageItem = pg.ImageItem()
        # ViewBox presents video and contains overlays
        self.viewBox = self.addViewBox(enableMenu=False,
                                       enableMouse=False,
                                       invertY=False,
                                       lockAspect=True)

        self.viewBox.addItem(self.imageItem)

        self._filters = []
        self.pauseSignals(False)

        self.fpsmeter = FpsMeter()
        self.camera = camera

    def close(self):
        logger.debug('Shutting down camera thread')
        self._thread.stop()
        self._thread.quit()
        self._thread.wait()
        self._thread = None
        self._camera = None

    def closeEvent(self):
        self.close()

    def sizeHint(self):
        if self.camera is None:
            size = QSize(640, 480)
        else:
            device = self.camera.device
            size = QSize(device.width, device.height)
        logger.debug('Size hint: {}'.format(size))
        return size

    def updateShape(self):
        device = self.camera.device
        self.resize(device.width, device.height)
        self.viewBox.setRange(xRange=(0, device.width),
                              yRange=(0, device.height),
                              padding=0, update=True)

    @property
    def filters(self):
        return self._filters

    @property
    def width(self):
        return self._shape[1]

    @property
    def height(self):
        return self._shape[0]

    @pyqtSlot(int)
    def setWidth(self, width):
        self.updateShape()

    @pyqtSlot(int)
    def setHeight(self, height):
        self.updateShape()

    @pyqtProperty(object)
    def camera(self):
        return self._camera

    @camera.setter
    def camera(self, camera):
        logger.debug('Setting Camera: {}'.format(type(camera)))
        self._camera = camera
        if camera is None:
            return
        self.updateShape()
        camera.widthChanged.connect(self.setWidth)
        camera.heightChanged.connect(self.setHeight)
        self._thread = QCameraThread(self)
        self._thread.start(QThread.TimeCriticalPriority)
        self.source = self.default

    @pyqtProperty(object)
    def default(self):
        return self._thread

    @pyqtProperty(object)
    def source(self):
        return self._source

    @source.setter
    def source(self, source):
        try:
            self.source.sigNewFrame.disconnect(self.updateImage)
        except AttributeError:
            pass
        self._source = source or self.default
        self._source.sigNewFrame.connect(self.updateImage)

    @pyqtSlot(np.ndarray)
    def updateImage(self, image):
        self._shape = image.shape
        self.source.blockSignals(True)
        for filter in self._filters:
            image = filter(image)
        self.source.blockSignals(False)
        self.sigNewFrame.emit(image)
        self.imageItem.setImage(image, autoLevels=False)
        if self.fpsmeter.tick():
            self.sigFPS.emit(self.fpsmeter.tock())

    @pyqtProperty(float)
    def fps(self):
        return self.fpsmeter.value

    def registerFilter(self, filter):
        self._filters.append(filter)

    def unregisterFilter(self, filter):
        if filter in self._filters:
            self._filters.remove(filter)

    def addOverlay(self, graphicsItem):
        """Convenience routine for placing overlays over video."""
        self.viewBox.addItem(graphicsItem)

    def removeOverlay(self, graphicsItem):
        """Convenience routine for removing overlays."""
        self.viewBox.removeItem(graphicsItem)

    @pyqtSlot(bool)
    def pauseSignals(self, pause):
        self._pause = bool(pause)

    def mouseMoveEvent(self, event):
        if not self._pause:
            self.sigMouseMove.emit(event)
        event.accept()

    def wheelEvent(self, event):
        if not self._pause:
            self.sigMouseWheel.emit(event)
        event.accept()

    def mousePressEvent(self, event):
        self.sigMousePress.emit(event)
        event.accept()

    def mouseReleaseEvent(self, event):
        self.sigMouseRelease.emit(event)
        event.accept()
