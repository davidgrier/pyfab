# -*- coding: utf-8 -*-

"""QJansenScreen.py: PyQt GUI for live video with graphical overlay."""

import PyQt5
from PyQt5.QtCore import (pyqtSignal, pyqtSlot, pyqtProperty)
from PyQt5.QtGui import (QMouseEvent, QWheelEvent)
import pyqtgraph as pg
from video.QSpinnaker.QSpinnaker import QSpinnaker as Camera
import numpy as np


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

        if camera is None:
            self.camera = Camera(self, **kwargs)
        else:
            camera.setParent(self)
            self.camera = camera
        self.source = self.camera
        height, width = self.camera.device.size()

        # ImageItem displays video feed
        self.imageItem = pg.ImageItem()
        # ViewBox presents video and contains overlays
        self.viewBox = self.addViewBox(enableMenu=False,
                                       enableMouse=False,
                                       invertY=False,
                                       lockAspect=True)
        self.viewBox.setRange(xRange=(0, width), yRange=(0, height),
                              padding=0, update=True)
        self.viewBox.addItem(self.imageItem)
        self._filters = []
        self._pause = False

    def close(self):
        self.camera.close()

    def closeEvent(self):
        self.close()

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
