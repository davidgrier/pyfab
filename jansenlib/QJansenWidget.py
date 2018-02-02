#!/usr/bin/env python

"""QJansenWidget.py: GUI for holographic video microscopy."""
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from QJansenScreen import QJansenScreen
import video
import DVR
from tasks.taskmanager import taskmanager
import sys
import numpy as np
import cv2


def tabLayout():
    layout = QtGui.QVBoxLayout()
    layout.setAlignment(QtCore.Qt.AlignTop)
    layout.setSpacing(1)
    return layout


class histogramTab(QtGui.QWidget):

    def __init__(self, parent):
        super(histogramTab, self).__init__(parent=parent)

        self.title = 'Histogram'
        self.index = -1
        self.video = self.parent().screen.video
        self.parent().tabs.currentChanged.connect(self.expose)

        layout = tabLayout()
        self.setLayout(layout)

        histo = pg.PlotWidget(background='w')
        histo.setMaximumHeight(250)
        histo.setXRange(0, 255)
        histo.setLabel('bottom', 'Intensity')
        histo.setLabel('left', 'N(Intensity)')
        histo.showGrid(x=True, y=True)
        self.rplot = histo.plot()
        self.rplot.setPen('r', width=2)
        self.gplot = histo.plot()
        self.gplot.setPen('g', width=2)
        self.bplot = histo.plot()
        self.bplot.setPen('b', width=2)
        layout.addWidget(histo)

        xmean = pg.PlotWidget(background='w')
        xmean.setMaximumHeight(150)
        xmean.setLabel('bottom', 'x [pixel]')
        xmean.setLabel('left', 'I(x)')
        xmean.showGrid(x=True, y=True)
        self.xplot = xmean.plot()
        self.xplot.setPen('r', width=2)
        layout.addWidget(xmean)

        ymean = pg.PlotWidget(background='w')
        ymean.setMaximumHeight(150)
        ymean.setLabel('bottom', 'y [pixel]')
        ymean.setLabel('left', 'I(y)')
        ymean.showGrid(x=True, y=True)
        self.yplot = ymean.plot()
        self.yplot.setPen('r', width=2)
        layout.addWidget(ymean)

    def expose(self, index):
        if index == self.index:
            self.video.registerFilter(self.histogramFilter)
        else:
            self.video.unregisterFilter(self.histogramFilter)

    def histogramFilter(self, frame):
        if self.video.gray:
            y, x = np.histogram(frame, bins=256, range=[0, 255])
            self.rplot.setData(y=y)
            self.gplot.setData(y=[0, 0])
            self.bplot.setData(y=[0, 0])
            self.xplot.setData(y=np.mean(frame, 0))
            self.yplot.setData(y=np.mean(frame, 1))
        else:
            b, g, r = cv2.split(frame)
            y, x = np.histogram(r, bins=256, range=[0, 255])
            self.rplot.setData(y=y)
            y, x = np.histogram(g, bins=256, range=[0, 255])
            self.gplot.setData(y=y)
            y, x = np.histogram(b, bins=256, range=[0, 255])
            self.bplot.setData(y=y)
            self.xplot.setData(y=np.mean(r, 0))
            self.yplot.setData(y=np.mean(r, 1))
        return frame


class QJansenWidget(QtGui.QWidget):

    def __init__(self, size=(640, 480)):
        super(QJansenWidget, self).__init__()
        self.init_hardware(size)
        self.init_ui()

    def init_hardware(self, size):
        # video screen
        self.screen = QJansenScreen(size=size, gray=True)
        self.video = video.QFabVideo(self.screen.video)
        self.filters = video.QFabFilter(self.screen.video)
        self.tasks = taskmanager(parent=self)
        # DVR
        self.dvr = DVR.QFabDVR(source=self.screen.video)
        self.dvr.recording.connect(self.handleRecording)

    def init_ui(self):
        layout = QtGui.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        layout.addWidget(self.screen)
        self.tabs = QtGui.QTabWidget()
        self.tabs.setMaximumWidth(400)
        self.tabs.addTab(self.videoTab(), 'Video')
        tab = histogramTab(self)
        index = self.tabs.addTab(tab, 'Histogram')
        tab.index = index
        layout.addWidget(self.tabs)
        layout.setAlignment(self.tabs, QtCore.Qt.AlignTop)
        self.setLayout(layout)

    def videoTab(self):
        wvideo = QtGui.QWidget()
        layout = tabLayout()
        layout.addWidget(self.dvr)
        layout.addWidget(self.video)
        layout.addWidget(self.filters)
        wvideo.setLayout(layout)
        return wvideo

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_R:
            if self.dvr.isrecording():
                self.dvr.bstop.animateClick(100)
            else:
                self.dvr.brecord.animateClick(100)
        elif event.key() == QtCore.Qt.Key_S:
            self.dvr.bstop.animateClick(100)
        event.accept()

    def handleRecording(self, recording):
        self.video.enabled = not recording


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    instrument = QJansenWidget()
    sys.exit(app.exec_())
