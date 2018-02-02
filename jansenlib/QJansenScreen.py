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
        # Graphical representations of traps
        # self.plot = pg.ScatterPlotItem()
        # ViewBox presents video and plot of trap positions
        self.viewbox = self.addViewBox(enableMenu=False,
                                       enableMouse=False,
                                       lockAspect=1.)
        self.viewbox.setRange(self.video.device.roi, padding=0,
                              update=True)
        self.viewbox.addItem(self.video)
        # self.viewbox.addItem(self.plot)
        self.emitSignals = True

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

    def addOverlay(self, graphicsItem):
        self.viewbox.addItem(graphicsItem)

    # def selectedPoint(self, position):
    #    points = self.plot.pointsAt(position)
    #    if len(points) <= 0:
    #        return None
    #    index = self.plot.points().tolist().index(points[0])
    #    return index

    # def setData(self, **kwargs):
    #    '''
    #    Accepts keyword arguments for ScatterPlotItem
    #    '''
    #    self.plot.setData(**kwargs)
