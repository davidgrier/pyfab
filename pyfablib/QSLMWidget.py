# -*- coding: utf-8 -*-

from PyQt5.QtCore import pyqtSlot
import pyqtgraph as pg
import numpy as np


class QSLMWidget(pg.PlotWidget):

    def __init__(self, parent=None):
        super(QSLMWidget, self).__init__(parent, background='w')

        self.getAxis('bottom').setPen(0.1)
        self.getAxis('left').setPen(0.1)
        self.setRange(xRange=[0, 640], yRange=[0, 480], padding=0)
        self.setAspectLocked(True)
        self.enableAutoRange(enable=False)
        self.image = pg.ImageItem()
        self.addItem(self.image)
        self.image.setOpts(axisOrder='row-major')

    @pyqtSlot(np.ndarray)
    def setData(self, hologram):
        if self.isVisible():
            self.image.setImage(hologram)
