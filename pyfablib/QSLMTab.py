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
        wgraphics = pg.GraphicsLayoutWidget()
        view = wgraphics.addViewBox(enableMenu=False,
                                    enableMouse=False,
                                    lockAspect=1.)
        self.image = pg.ImageItem()
        view.addItem(self.image)
        layout.addWidget(wgraphics)

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
