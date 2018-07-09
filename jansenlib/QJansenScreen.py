# -*- coding: utf-8 -*-

"""QJansenScreen.py: PyQt GUI for live video with graphical overlay."""

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from .video.QVideoItem import QVideoItem


class QJansenScreen(pg.GraphicsLayoutWidget):
    """Interactive display for pyfab system.

    QJansenScreen incorporates a QVideoItem to display live video.
    Additional GraphicsItems can be added to the viewbox
    as overlays over the video stream.
    Interaction with graphical items is facilitated
    by custom signals that correspond to mouse events.
    """
    sigMousePress = QtCore.pyqtSignal(QtGui.QMouseEvent)
    sigMouseRelease = QtCore.pyqtSignal(QtGui.QMouseEvent)
    sigMouseMove = QtCore.pyqtSignal(QtGui.QMouseEvent)
    sigMouseWheel = QtCore.pyqtSignal(QtGui.QWheelEvent)

    def __init__(self, parent=None, **kwargs):
        super(QJansenScreen, self).__init__()
        self.ci.layout.setContentsMargins(0, 0, 0, 0)
        self.parent = parent
        # VideoItem displays video feed
        self.video = QVideoItem(parent=self, **kwargs)
        source = self.video.source
        # ViewBox presents video and contains overlays
        self.viewbox = self.addViewBox(enableMenu=False,
                                       enableMouse=False,
                                       invertY=False,
                                       lockAspect=True)
        self.viewbox.setRange(source.roi, padding=0, update=True)
        self.viewbox.addItem(self.video)

    def addOverlay(self, graphicsItem):
        """Convenience routine for placing overlays over video."""
        self.viewbox.addItem(graphicsItem)

    def removeOverlay(self, graphicsItem):
        """Convenience routine for removing overlays."""
        self.viewbox.removeItem(graphicsItem)

    def close(self):
        self.video.close()

    def mousePressEvent(self, event):
        self.sigMousePress.emit(event)
        event.accept()

    def mouseReleaseEvent(self, event):
        self.sigMouseRelease.emit(event)
        event.accept()

    def mouseMoveEvent(self, event):
        self.sigMouseMove.emit(event)
        event.accept()

    def wheelEvent(self, event):
        self.sigMouseWheel.emit(event)
        event.accept()
