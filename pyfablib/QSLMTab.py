# -*- coding: utf-8 -*-

import PyQt5
from PyQt5.QtWidgets import (QFrame, QLabel)
from PyQt5.QtCore import pyqtSlot
import pyqtgraph as pg
from common.tabLayout import tabLayout
import numpy as np


class QSLMTab(QFrame):

    def __init__(self, cgh=None):
        super(QSLMTab, self).__init__()
        self.title = 'SLM'
        self.index = -1

        self.cgh = cgh

        self.setFrameShape(QFrame.Box)
        layout = tabLayout(self)

        title = QLabel('SLM data')
        layout.addWidget(title)
        graphics = pg.PlotWidget(background='w')
        graphics.getAxis('bottom').setPen(0.1)
        graphics.getAxis('left').setPen(0.1)
        graphics.setXRange(0, self.cgh.w)
        graphics.setYRange(0, self.cgh.h)
        graphics.setAspectLocked(True)
        graphics.enableAutoRange(enable=False)
        self.image = pg.ImageItem()
        graphics.addItem(self.image)
        layout.addWidget(graphics)

    def expose(self, index):
        if index == self.index:
            self.update(self.cgh.phi)
            self.cgh.sigHologramReady.connect(self.update)
        else:
            try:
                self.cgh.sigHologramReady.disconnect(self.update)
            except Exception:
                pass

    @pyqtSlot(np.ndarray)
    def update(self, hologram):
        self.image.setImage(hologram)
