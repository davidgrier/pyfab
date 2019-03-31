# -*- coding: utf-8 -*-

"""QJansenScreen.py: PyQt GUI for live video with graphical overlay."""

import PyQt5
from PyQt5.QtCore import (pyqtSignal, pyqtSlot, pyqtProperty,
                          QThread, QSize)
from PyQt5.QtGui import (QMouseEvent, QWheelEvent)
import pyqtgraph as pg
import numpy as np

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class QCameraThread(QThread):

    '''Class for reducing latency by moving cameras to separate thread'''

    sigNewFrame = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super(QCameraThread, self).__init__(parent)
        self.camera = self.parent().camera
        self.shape = self.camera.shape

    def run(self):
        logger.debug('Starting acquisition loop')
        self._running = True
        while self._running:
            ready, frame = self.camera.read()
            if ready:
                self.sigNewFrame.emit(frame)
            else:
                logger.warn('Failed to read frame')
        logger.debug('Stopping acquisition loop')
        self.camera.close()

    def stop(self):
        self._running = False


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

    Slots
    -----
    """
    sigMousePress = pyqtSignal(QMouseEvent)
    sigMouseRelease = pyqtSignal(QMouseEvent)
    sigMouseMove = pyqtSignal(QMouseEvent)
    sigMouseWheel = pyqtSignal(QWheelEvent)

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

        self.camera = camera

    def close(self):
        self._thread.stop()
        self._thread.quit()
        self._thread.wait()

    def closeEvent(self):
        self.close()

    def sizeHint(self):
        if self.camera is None:
            return QSize(640, 480)
        else:
            shape = self.camera.shape
            return QSize(shape[1], shape[0])

    @pyqtProperty(object)
    def camera(self):
        return self._camera

    @camera.setter
    def camera(self, camera):
        logger.debug('Setting Camera: {}'.format(type(camera)))
        self._camera = camera
        if camera is None:
            return

        shape = self._camera.shape
        self.viewBox.setRange(xRange=(0, shape[1]),
                              yRange=(0, shape[0]),
                              padding=0, update=True)
        self._thread = QCameraThread(self)
        self._thread.start()
        self.source = self._thread

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
            source = self._thread
        source.sigNewFrame.connect(self.updateImage)
        self._source = source

    @pyqtSlot(np.ndarray)
    def updateImage(self, image):
        self.source.blockSignals(True)
        for filter in self._filters:
            image = filter(image)
        self.source.blockSignals(False)
        self.imageItem.setImage(image, autoLevels=False)

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
