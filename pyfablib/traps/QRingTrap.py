# -*- coding: utf-8 -*-

"""QRingTrap.py: Ring Trap"""

from .QTrap import QTrap
import numpy as np
from pyqtgraph.Qt import QtGui
from scipy.special import jv


class QRingTrap(QTrap):

    def __init__(self, R=80, m=20, alpha=50, **kwargs):
        super(QRingTrap, self).__init__(alpha=alpha, **kwargs)
        self._R = R
        self._m = m
        self.registerProperty('R', tooltip=True)
        self.registerProperty('m', decimals=0, tooltip=True)

    def updateStructure(self):
        phi = jv(self.m, self.R * self.cgh.qr) * \
            np.exp((1j * self.m) * self.cgh.theta)
        self.structure = phi

    def plotSymbol(self):
        sym = QtGui.QPainterPath()
        font = QtGui.QFont('Sans Serif', 10, QtGui.QFont.Black)
        sym.addText(0, 0, font, 'o')
        # Scale symbol to unit square
        box = sym.boundingRect()
        scale = 1./max(box.width(), box.height())
        tr = QtGui.QTransform().scale(scale, scale)
        # Center symbol on (0, 0)
        tr.translate(-box.x() - box.width()/2., -box.y() - box.height()/2.)
        return tr.map(sym)

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, R):
        self._R = R
        self.updateStructure()
        self.valueChanged.emit(self)

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, m):
        self._m = np.int(m)
        self.updateStructure()
        self.valueChanged.emit(self)
