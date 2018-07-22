# -*- coding: utf-8 -*-

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from common.tabLayout import tabLayout
import numpy as np


class QSLMTab(QtGui.QWidget):

    def __init__(self, cgh=None):
        super(QSLMTab, self).__init__()
        self.title = 'SLM'
        self.index = -1

        self.cgh = cgh

        layout = tabLayout(self)
        graphics = pg.GraphicsLayoutWidget()
        ax = graphics.addPlot(title='', autoLevels=False)
        ax.setXRange(0, self.cgh.w)
        ax.setYRange(0, self.cgh.h)
        ax.setAspectLocked(True)
        self.image = pg.ImageItem()
        ax.addItem(self.image)
        layout.addWidget(graphics)

    def expose(self, index):
        if index == self.index:
            self.update(self.cgh.phi.T)
            self.cgh.sigHologramReady.connect(self.update)
        else:
            try:
                self.cgh.sigHologramReady.disconnect(self.update)
            except Exception:
                pass

    @QtCore.pyqtSlot(np.ndarray)
    def update(self, hologram):
        self.image.setImage(hologram)
