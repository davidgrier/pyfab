# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSlot
from .QHistogram_UI import Ui_QHistogramWidget
import numpy as np
import cv2

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class QHistogram(QWidget):

    def __init__(self, parent=None, nskip=3):
        super(QHistogram, self).__init__(parent)

        self.ui = Ui_QHistogramWidget()
        self.ui.setupUi(self)
        self.configurePlots()
        self._n = 0
        self._nskip = nskip

    def configurePlots(self):
        histo = self.configurePlot(self.ui.histo, 'Intensity', 'N(Intensity)')
        self.rplot = histo.plot()
        self.rplot.setPen('r', width=2)
        self.gplot = self.ui.histo.plot()
        self.gplot.setPen('g', width=2)
        self.bplot = self.ui.histo.plot()
        self.bplot.setPen('b', width=2)

        xmean = self.configurePlot(self.ui.xmean, 'x [pixel]', 'I(x)')
        self.xplot = xmean.plot()
        self.xplot.setPen('r', width=2)

        ymean = self.configurePlot(self.ui.ymean, 'y [pixel]', 'I(y)')
        self.yplot = self.ui.ymean.plot()
        self.yplot.setPen('r', width=2)

    @staticmethod
    def configurePlot(plot, xtitle, ytitle):
        plot.setBackground('w')
        plot.getAxis('bottom').setPen(0.1)
        plot.getAxis('left').setPen(0.1)
        plot.showGrid(x=True, y=True)
        plot.setMouseEnabled(x=False, y=False)
        plot.setLabel('bottom', xtitle)
        plot.setLabel('left', ytitle)
        return plot

    @pyqtSlot(np.ndarray)
    def updateHistogram(self, frame):
        if not self.isVisible():
            return
        self._n = (self._n + 1) % self._nskip
        if self._n != 0:
            return
        if frame.ndim == 2:
            y = np.bincount(frame.flatten(), minlength=256)
            self.rplot.setData(y=y)
            self.gplot.setData(y=[0, 0])
            self.bplot.setData(y=[0, 0])
            self.xplot.setData(y=np.mean(frame, 0))
            self.yplot.setData(y=np.mean(frame, 1))
        else:
            b, g, r = cv2.split(frame)
            y = np.bincount(r.ravel(), minlength=256)
            self.rplot.setData(y=y)
            y = np.bincount(g.ravel(), minlength=256)
            self.gplot.setData(y=y)
            y = np.bincount(b.ravel(), minlength=256)
            self.bplot.setData(y=y)
            self.xplot.setData(y=np.mean(r, 0))
            self.yplot.setData(y=np.mean(r, 1))
    

if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    wid = QHistogram()
    wid.show()
    sys.exit(app.exec_())
