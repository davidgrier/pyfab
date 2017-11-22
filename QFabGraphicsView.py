#!/usr/bin/env python

"""QFabGraphicsView.py: PyQt GUI for live video with graphical overlay."""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from QCameraItem import QCameraItem
from PyQt4.QtCore import Qt


class QFabGraphicsView(pg.GraphicsLayoutWidget):
    """Interactive display for pyfab system.
    Incorporates a QCameraItem to display live video and a
    ScatterPlotItem to present graphical representations of traps
    overlayed on the video stream.
    Interaction with traps is handled by emitting custom signals
    corresponding to mouse events.  A separate module must
    interpret these signals and update the trap display accordingly.
    """
    sigMousePress = QtCore.pyqtSignal(QtGui.QMouseEvent)
    sigMouseMove = QtCore.pyqtSignal(QtGui.QMouseEvent)
    sigMouseRelease = QtCore.pyqtSignal(QtGui.QMouseEvent)
    sigWheel = QtCore.pyqtSignal(QtGui.QWheelEvent)
    sigClosed = QtCore.pyqtSignal()

    def __init__(self, parent=None, **kwargs):
        super(QFabGraphicsView, self).__init__(parent)

        self.setAttribute(Qt.WA_DeleteOnClose, True)

        # CameraItem displays video feed
        vb = self.addViewBox(border=None,
                             lockAspect=True,
                             enableMenu=False,
                             enableMouse=False)
        self.camera = QCameraItem(**kwargs)
        vb.addItem(self.camera)

        size = self.camera.device.size
        vb.setLimits(xMin=0, yMin=0, xMax=size.width(), yMax=size.height())
        vb.setRange(xRange=[0, size.width()], yRange=[0, size.height()])

        # ScatterPlotItem shows graphical representations of traps
        pen = pg.mkPen('k', width=0.5)
        brush = pg.mkBrush(100, 255, 100, 120)
        self.traps = pg.ScatterPlotItem(size=10, pen=pen, brush=brush)
        vb.addItem(self.traps)

    def closeEvent(self, event):
        self.camera.close()
        self.sigClosed.emit()

    def selectedPoint(self, position):
        index = -1
        points = self.traps.pointsAt(position)
        if len(points) > 0:
            index = self.traps.points().tolist().index(points[0])
        return index

    def mousePressEvent(self, event):
        self.sigMousePress.emit(event)
        event.accept()

    def mouseMoveEvent(self, event):
        self.sigMouseMove.emit(event)
        event.accept()

    def mouseReleaseEvent(self, event):
        self.sigMouseRelease.emit(event)
        event.accept()

    def wheelEvent(self, event):
        self.sigWheel.emit(event)
        event.accept()

    def setData(self, **kwargs):
        '''
        Accepts keyword arguments for ScatterPlotItem
        '''
        self.traps.setData(**kwargs)


class demopattern(object):
    """Reference implementation of trapping pattern that creates
    and destroys graphical representations of traps, and
    allows them to be dragged.
    """

    def __init__(self, fabscreen):
        self.fabscreen = fabscreen
        # Connect to signals coming from fabscreen
        self.fabscreen.sigMousePress.connect(self.mousePress)
        self.fabscreen.sigMouseMove.connect(self.mouseMove)
        self.fabscreen.sigMouseRelease.connect(self.mouseRelease)
        # Graphics for traps
        self.brush = {'normal': pg.mkBrush(100, 255, 100, 120),
                      'selected': pg.mkBrush(255, 100, 100, 120)}
        self.pen = pg.mkPen('k', width=0.5)
        # Trap positions and index of selected trap
        self.xy = []
        self.index = None

    def updateScreen(self):
        """Draw traps on fabscreen
        """
        spots = []
        for index, xy in enumerate(self.xy):
            spots.append({'pos': xy,
                          'size': 10,
                          'pen': self.pen,
                          'brush': self.brush['normal'],
                          'symbol': 'o'})
        if self.index is not None:
            spots[self.index]['brush'] = self.brush['selected']
        self.fabscreen.setData(spots=spots)

    def mousePress(self, event):
        position = self.fabscreen.traps.mapFromScene(event.pos())
        button = event.button()
        modifier = event.modifiers()
        index = self.fabscreen.selectedPoint(position)
        # Manipulate traps
        if button == Qt.LeftButton:
            # Add trap
            if modifier == Qt.ShiftModifier:
                self.index = len(self.xy)
                xy = np.array([position.x(), position.y()])
                self.xy.append(xy)
            # Delete trap
            elif modifier == Qt.ControlModifier:
                if index >= 0:
                    self.index = None
                    self.xy.pop(index)
            # Select trap
            elif index >= 0:
                self.index = index
            # Not interacting with traps
            else:
                self.index = None
        # Manipulate ROI?
        elif button == Qt.RightButton:
            pass
        self.updateScreen()

    def mouseMove(self, event):
        """Move selected trap's graphic to new position
        """
        position = self.fabscreen.traps.mapFromScene(event.pos())
        if self.index is not None:
            self.xy[self.index] = np.array([position.x(), position.y()])
            self.updateScreen()

    def mouseRelease(self):
        self.index = None
        self.updateScreen()


def main():
    import sys
    from pyqtgraph.Qt import QtGui

    app = QtGui.QApplication(sys.argv)

    fabscreen = QFabGraphicsView(size=(640, 480), gray=True)
    fabscreen.show()

    d = demopattern(fabscreen)

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
