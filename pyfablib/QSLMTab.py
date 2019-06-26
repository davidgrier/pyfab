# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import (QFrame, QLabel)
from PyQt5.QtCore import pyqtSlot
import pyqtgraph as pg
from common.tabLayout import tabLayout
import numpy as np


class QSLMTab(QFrame):

    def __init__(self, parent=None):
        super(QSLMTab, self).__init__(parent)
        self.title = 'SLM'

        self.setFrameShape(QFrame.Box)
        layout = tabLayout(self)

        title = QLabel('SLM data')
        layout.addWidget(title)
        graphics = pg.PlotWidget(background='w')
        graphics.getAxis('bottom').setPen(0.1)
        graphics.getAxis('left').setPen(0.1)
        graphics.setRange(padding=0)
        # graphics.setXRange(0, self.cgh.w)
        # graphics.setYRange(0, self.cgh.h)
        graphics.setAspectLocked(True)
        graphics.enableAutoRange(enable=False)
        self.image = pg.ImageItem()
        graphics.addItem(self.image)
        layout.addWidget(graphics)

    @pyqtSlot(np.ndarray)
    def setData(self, hologram):
        if self.isVisible():
            self.image.setImage(hologram)
