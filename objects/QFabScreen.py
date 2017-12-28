#!/usr/bin/env python

"""QFabScreen.py: PyQt GUI for live video with graphical overlay."""

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from QVideoItem import QVideoItem


class QFabScreen(pg.GraphicsLayoutWidget):
    """Interactive display for pyfab system.
    Incorporates a QVideoItem to display live video and a
    ScatterPlotItem to present graphical representations of traps
    overlayed on the video stream.
    Interaction with traps is handled by emitting custom signals
    corresponding to mouse events.  A separate module must
    interpret these signals and update the trap display accordingly.
    """
    sigMousePress = QtCore.pyqtSignal(QtGui.QMouseEvent)
    sigMouseMove = QtCore.pyqtSignal(QtGui.QMouseEvent)
    sigMouseRelease = QtCore.pyqtSignal(QtGui.QMouseEvent)
    sigMouseWheel = QtCore.pyqtSignal(QtGui.QWheelEvent)

    def __init__(self, parent=None, **kwargs):
        super(QFabScreen, self).__init__(parent)

        # VideoItem displays video feed
        self.video = QVideoItem(**kwargs)
        # Graphical representations of traps
        self.plot = pg.ScatterPlotItem()
        # ViewBox presents video and plot of trap positions
        vb = self.addViewBox(enableMenu=False,
                             enableMouse=False,
                             lockAspect=1.)
        vb.setRange(self.video.device.roi, padding=0, update=True)
        vb.addItem(self.video)
        vb.addItem(self.plot)
        self.active = True

    def closeEvent(self, event):
        self.video.close()

    def selectedPoint(self, position):
        points = self.plot.pointsAt(position)
        if len(points) <= 0:
            return None
        index = self.plot.points().tolist().index(points[0])
        return index

    def pauseSignals(self, pause):
        self.active = not pause

    def mousePressEvent(self, event):
        if self.active:
            self.sigMousePress.emit(event)
        event.accept()

    def mouseMoveEvent(self, event):
        if self.active:
            self.sigMouseMove.emit(event)
        event.accept()

    def mouseReleaseEvent(self, event):
        if self.active:
            self.sigMouseRelease.emit(event)
        event.accept()

    def wheelEvent(self, event):
        if self.active:
            self.sigMouseWheel.emit(event)
        event.accept()

    def setData(self, **kwargs):
        '''
        Accepts keyword arguments for ScatterPlotItem
        '''
        self.plot.setData(**kwargs)
