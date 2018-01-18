#!/usr/bin/env python

"""QJansenWidget.py: GUI for holographic video microscopy."""

from pyqtgraph.Qt import QtGui, QtCore
import objects
import tasks
import sys
import numpy as np
import cv2
import pyqtgraph as pg


class histogramTab(pg.PlotWidget):

    def __init__(self, parent):
        super(histogramTab, self).__init__(parent=parent,
                                           background='w',
                                           border=pg.mkPen('k'))
        self.title = 'Histogram'
        self.index = -1
        self.video = self.parent().fabscreen.video
        self.parent().tabs.currentChanged.connect(self.expose)
        self.setLabel('bottom', 'Intensity')
        self.setLabel('left', 'Counts')
        self.showGrid(x=True, y=True)
        self.rplot = self.plot()
        self.rplot.setPen('r', width=2)
        self.gplot = self.plot()
        self.gplot.setPen('g', width=2)
        self.bplot = self.plot()
        self.bplot.setPen('b', width=2)

    def expose(self, index):
        if index == self.index:
            self.video.registerFilter(self.histogramFilter)
        else:
            self.video.unregisterFilter(self.histogramFilter)

    def histogramFilter(self, frame):
        if self.video.gray:
            y, x = np.histogram(frame, bins=256, range=[0, 255])
            self.rplot.setData(x=x[:-1], y=y)
        else:
            b, g, r = cv2.split(frame)
            y, x = np.histogram(r, bins=256, range=[0, 255])
            self.rplot.setData(x=x[:-1], y=y)
            y, x = np.histogram(g, bins=256, range=[0, 255])
            self.gplot.setData(x=x[:-1], y=y)
            y, x = np.histogram(b, bins=256, range=[0, 255])
            self.bplot.setData(x=x[:-1], y=y)
        return frame


class QJansenWidget(QtGui.QWidget):

    def __init__(self, size=(640, 480)):
        super(QJansenWidget, self).__init__()
        self.init_hardware(size)
        self.init_ui()

    def init_hardware(self, size):
        # video screen
        self.fabscreen = objects.QFabScreen(size=size, gray=True)
        self.video = objects.QFabVideo(self.fabscreen.video)
        self.filters = objects.QFabFilter(self.fabscreen.video)
        self.config = objects.fabconfig(self)
        self.tasks = tasks.taskmanager(parent=self)
        # DVR
        self.dvr = objects.QFabDVR(source=self.fabscreen.video)
        self.dvr.recording.connect(self.handleRecording)

    def init_ui(self):
        layout = QtGui.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        layout.addWidget(self.fabscreen)
        self.tabs = QtGui.QTabWidget()
        self.tabs.addTab(self.videoTab(), 'Video')
        tab = histogramTab(self)
        index = self.tabs.addTab(tab, 'Histogram')
        tab.index = index
        layout.addWidget(self.tabs)
        layout.setAlignment(self.tabs, QtCore.Qt.AlignTop)
        self.setLayout(layout)

    def videoTab(self):
        wvideo = QtGui.QWidget()
        layout = QtGui.QVBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(1)
        layout.addWidget(self.dvr)
        layout.addWidget(self.video)
        layout.addWidget(self.filters)
        wvideo.setLayout(layout)
        return wvideo

    def handleRecording(self, recording):
        self.video.enabled = not recording


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    instrument = QJansenWidget()
    instrument.tasks.registerTask(tasks.maxtask())
    sys.exit(app.exec_())
