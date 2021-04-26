# -*- coding: utf-8 -*-

"""QBesselTrap.py: Bessel Trap"""

from .QTrap import QTrap
import numpy as np
from PyQt5.QtCore import pyqtProperty
from PyQt5.QtGui import (QPainterPath, QFont, QTransform)


class QBesselTrap(QTrap):
    def __init__(self, shift=100, **kwargs):
        super(QBesselTrap, self).__init__(**kwargs)
        self._shift = shift
        self.registerProperty('shift', tooltip=True)
        
    def updateStructure(self):
        phi = np.remainder(np.angle(1) - self.shift*self.cgh.qr, 2*(np.pi))
        self.structure = np.exp(1j * phi)

    def plotSymbol(self):
        sym = QPainterPath()
        font = QFont()
        font.setStyleHint(QFont.SansSerif, QFont.PreferAntialias)
        font.setPointSize(12)
        sym.addText(0, 0, font, 'B')
        # scale symbol to unit square
        box = sym.boundingRect()
        scale = 1./max(box.width(), box.height())
        tr = QTransform().scale(scale, scale)
        # center symbol on (0, 0)
        tr.translate(-box.x() - box.width()/2., -box.y() - box.height()/2.)
        return tr.map(sym)

    @pyqtProperty(int)
    def shift(self):
        return self._shift

    @shift.setter
    def shift(self, shift):
        self._shift = np.int(shift)
        self.updateStructure()
        self.valueChanged.emit(self)
