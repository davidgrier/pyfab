# -*- coding: utf-8 -*-

"""QJansenScreen.py: PyQt GUI for live video with graphical overlay."""

import PyQt5
from PyQt5.QtCore import (pyqtSignal, pyqtSlot)
from PyQt5.QtGui import (QMouseEvent, QWheelEvent)
from pyfab.jansenlib.video.QVideoItem import QVideoItem
import pyqtgraph as pg
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

    Methods
    -------
    width(): int
        Width of video feed [pixels]
    height(): int
        Height of video feed [pixels]

    Signals
    -------
    sigMousePress(QMouseEvent)
    sigMouseRelease(QMouseEvent)
    sigMouseMove(QMouseEvent)
    sigMouseWheel(QMouseEvent)
    sigNewFrame(numpy.ndarray)

    Slots
    -----
    """
    sigMousePress = pyqtSignal(QMouseEvent)
    sigMouseRelease = pyqtSignal(QMouseEvent)
    sigMouseMove = pyqtSignal(QMouseEvent)
    sigMouseWheel = pyqtSignal(QWheelEvent)
    sigNewFrame = pyqtSignal(np.ndarray)

    def __init__(self, parent=None, camera=None):
        super(QJansenScreen, self).__init__(parent)
        self.ci.layout.setContentsMargins(0, 0, 0, 0)
        # VideoItem displays video feed
        self.videoItem = QVideoItem(parent=self, camera=camera)
        # ViewBox presents video and contains overlays
        self.viewbox = self.addViewBox(enableMenu=False,
                                       enableMouse=False,
                                       invertY=False,
                                       lockAspect=True)
        self.viewbox.setRange(xRange=(0, self.videoItem.width),
                              yRange=(0, self.videoItem.height),
                              padding=0, update=True)
        self.viewbox.addItem(self.videoItem)
        self._pause = False

    def addOverlay(self, graphicsItem):
        """Convenience routine for placing overlays over video."""
        self.viewbox.addItem(graphicsItem)

    def removeOverlay(self, graphicsItem):
        """Convenience routine for removing overlays."""
        self.viewbox.removeItem(graphicsItem)

    def close(self):
        self.video.close()

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
