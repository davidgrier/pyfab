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

    def update_field(self):
        if self.cgh is None:
            logger.info('Tried to update without CGH')
            return
        self.structure = np.exp((1j * self.ell) * self.cgh.theta)

    def plotSymbol(self):
        sym = QtGui.QPainterPath()
        sym.addText(0, 0, QtGui.QFont('San Serif', 10), 'V')
        box = sym.boundingRect()
        scale = min(1./box.width(), 1/box.height())
        tr = QtGui.QTransform()
        tr.scale(scale, scale)
        tr.translate(-box.x() - box.width()/2., -box.y() - box.height()/2.)
        sym = tr.map(sym)
        return sym

    @property
    def cgh(self):
        return self._cgh

    @cgh.setter
    def cgh(self, cgh):
        self._cgh = cgh
        self.update_field()
