#!/usr/bin/env python

"""QJansenWidget.py: GUI for holographic video microscopy."""

from pyqtgraph.Qt import QtGui, QtCore
import objects
import tasks
import sys
import numpy as np
import pyqtgraph as pg

class histogramTab(pg.PlotWidget):

    def __init__(self, parent):
        super(histogramTab, self).__init__(parent=parent)
        self.title = 'Histogram'
        self.index = -1
        self.video = self.parent().fabscreen.video
        self.parent().tabs.currentChanged.connect(self.expose)
        self.setLabel('bottom', 'Intensity')
        self.setLabel('left', 'Counts')
        self.plot = self.plot()
        self.plot.setPen((255, 255, 0))

    def expose(self, index):
        if index == self.index:
            self.video.registerFilter(self.histogramFilter)
        else:
            self.video.unregisterFilter(self.histogramFilter)

    def histogramFilter(self, frame):
        y, x = np.histogram(frame, bins=256, range=[0, 255])
        self.plot.setData(x=x[1:], y=y)
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
