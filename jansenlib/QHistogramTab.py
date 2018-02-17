import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from common.tabLayout import tabLayout
import numpy as np
import cv2


class QHistogramTab(QtGui.QWidget):

    def __init__(self, video_source):
        super(QHistogramTab, self).__init__()

        self.title = 'Histogram'
        self.index = -1
        self.video = video_source

        layout = tabLayout(self)
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
