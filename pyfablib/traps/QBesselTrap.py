# -*- coding: utf-8 -*-

"""QBesselTrap.py: Bessel Trap"""

from .QTrap import QTrap
import numpy as np
from PyQt5.QtCore import pyqtProperty


class QBesselTrap(QTrap):
    def __init__(self, shift=100, **kwargs):
        super(QBesselTrap, self).__init__(**kwargs)
        self._shift = shift
        self.registerProperty('shift', tooltip=True)
        
    def updateStructure(self):
        phi = np.remainder(np.angle(1) - self.shift*self.cgh.qr, 2*(np.pi))
        self.structure = np.exp(1j * phi)

    def plotSymbol(self):
        return self.letterSymbol('B')

    @pyqtProperty(int)
    def shift(self):
        return self._shift

    @shift.setter
    def shift(self, shift):
        self._shift = np.int(shift)
        self.updateStructure()
        self.valueChanged.emit(self)
