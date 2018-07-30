# -*- coding: utf-8 -*-

"""QBesselTrap.py: Bessel Trap"""

from .QTrap import QTrap
import numpy as np
from pyqtgraph.Qt import QtGui


class QBesselTrap(QTrap):
    def __init__(self, shift=1000, **kwargs):
        super(QBesselTrap, self).__init__()
        self._shift = shift

    def updateStructure(self):
        phi = np.remainder(np.angle(1) - self.shift*self.cgh.qr, 2*(np.pi))
        self.structure = np.exp(1j * phi)
        self.refresh()

    def plotSymbol(self):
        sym = QtGui.QPainterPath()
        font = QtGui.QFont('Sans Serif', 10, QtGui.QFont.Black)
        sym.addText(0, 0, font, 'b')
        # scale symbol to unit square
        box = sym.boundingRect()
        scale = 1./max(box.width(), box.height())
        tr = QtGui.QTransform().scale(scale, scale)
        # center symbol on (0, 0)
        tr.translate(-box.x() - box.width()/2., -box.y() - box.height()/2.)
        return tr.map(sym)

    @property
    def shift(self):
        return self._shift

    @shift.setter
    def shift(self, shift):
        self._shift = np.int(shift)
        self.update_structure()
        self.valueChanged.emit(self)