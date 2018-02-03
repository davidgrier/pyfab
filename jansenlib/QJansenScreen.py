#!/usr/bin/env python

"""QJansenScreen.py: PyQt GUI for live video with graphical overlay."""

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from video.QVideoItem import QVideoItem


class QJansenScreen(pg.GraphicsLayoutWidget):
    """Interactive display for pyfab system.
    QJansenScreen ncorporates a QVideoItem to display live video.
    Additional GraphicsItems can be added to the viewbox
    as overlays over the video stream.
    Interaction with graphical items is facilitated
    by emitting custom signals corresponding to mouse events.
    A separate module must interpret these signals and update
    the graphics display accordingly.
    """
    sigMousePress = QtCore.pyqtSignal(QtGui.QMouseEvent)
    sigMouseMove = QtCore.pyqtSignal(QtGui.QMouseEvent)
    sigMouseRelease = QtCore.pyqtSignal(QtGui.QMouseEvent)
    sigMouseWheel = QtCore.pyqtSignal(QtGui.QWheelEvent)

    def __init__(self, parent=None, **kwargs):
        super(QJansenScreen, self).__init__(parent)

        # VideoItem displays video feed
        self.video = QVideoItem(**kwargs)
        # ViewBox presents video and contains overlays
        self.viewbox = self.addViewBox(enableMenu=False,
                                       enableMouse=False,
                                       lockAspect=1.)
        self.viewbox.setRange(self.video.device.roi,
                              padding=0, update=True)
        self.viewbox.addItem(self.video)
        self.emitSignals = True

    def addOverlay(self, graphicsItem):
        """Convenience routine for placing overlays over video."""
        self.viewbox.addItem(graphicsItem)

    def removeOverlay(self, graphicsItem):
        """Convenience routine for removing overlays."""
        self.viewbox.removeItem(graphicsItem)

    def closeEvent(self, event):
        self.video.close()

    def mousePressEvent(self, event):
        if self.emitSignals:
            self.sigMousePress.emit(event)
        event.accept()

    def mouseMoveEvent(self, event):
        if self.emitSignals:
            self.sigMouseMove.emit(event)
        event.accept()

    def mouseReleaseEvent(self, event):
        if self.emitSignals:
            self.sigMouseRelease.emit(event)
        event.accept()

    def wheelEvent(self, event):
        if self.emitSignals:
            self.sigMouseWheel.emit(event)
        event.accept()
