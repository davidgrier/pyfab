# -*- coding: utf-8 -*-

"""QRingTrap.py: Ring Trap"""

from .QTrap import QTrap
import numpy as np
from PyQt5.QtCore import pyqtProperty
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
        return self.letterSymbol('O')

    @pyqtProperty(float)
    def R(self):
        return self._R

    @R.setter
    def R(self, R):
        self._R = R
        self.updateStructure()

    @pyqtProperty(int)
    def m(self):
        return self._m

    @m.setter
    def m(self, m):
        self._m = np.int(m)
        self.updateStructure()
