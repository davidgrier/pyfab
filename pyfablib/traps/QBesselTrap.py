# -*- coding: utf-8 -*-

"""QBesselTrap.py: Bessel Trap"""

from .QTrap import QTrap
import numpy as np
from pyqtgraph.Qt import QtGui

class QBesselTrap(QTrap):
    def __init__(self, **kwargs):
        super(QBesselTrap, self).__init__()
        

    def update_structure(self):
        phi = np.remainder(np.angle(1) - 100*self.cgh.qr, 2*(np.pi))
        self.structure = np.exp(1j * phi)
        self._update() 
        
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
    
        