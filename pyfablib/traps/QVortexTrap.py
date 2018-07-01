# -*- coding: utf-8 -*-

"""QVortexTrap.py: Optical vortex"""

from .QTrap import QTrap
import numpy as np
from pyqtgraph.Qt import QtGui

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)


class QVortexTrap(QTrap):
    def __init__(self, ell=10, **kwargs):
        super(QVortexTrap, self).__init__(**kwargs)
        self.ell = ell
        self.spot['symbol'] = self.plotSymbol()

    def update_structure(self):
        if self.cgh is None:
            logger.warn('Tried to update without CGH')
            return
        self.structure = np.exp((1j * self.ell) * self.cgh.theta)

    def plotSymbol(self):
        sym = QtGui.QPainterPath()
        sym.addText(0, 0, QtGui.QFont('San Serif', 10), 'V')
        # scale symbol to unit square
        box = sym.boundingRect()
        scale = 1./max(box.width(), box.height())
        tr = QtGui.QTransform().scale(scale, scale)
        # center symbol on (0, 0)
        tr.translate(-box.x() - box.width()/2., -box.y() - box.height()/2.)
        return tr.map(sym)

    @property
    def cgh(self):
        return self._cgh

    @cgh.setter
    def cgh(self, cgh):
        if cgh is None:
            return
        self._cgh = cgh
        self._cgh.sigUpdateGeometry.connect(self.update_structure)
        self.update_structure()
