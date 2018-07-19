# coding: utf-8

from .QTrap import QTrap
import numpy as np
import cmath 
from pyqtgraph.Qt import QtGui


class QBesselTrap(QTrap):
    def __init__(self, **kwargs):
        super(QBesselTrap, self).__init__()
        self.shift = 100
        self.shiftx = 0
        self.shifty = 0
        self.modetot = 1
    
    def update_structure(self):
        xv, yv = np.meshgrid(self.qx, self.qy)
        xv, yv = xv.T, yv.T
        qr = np.hypot.outer(self.qx, self.qy)
        phi = np.remainder(np.angle(self.modetot) - self.shift*qr -
                           self.shiftx*xv - self.shifty*yv, 2*(np.pi))
        self.structure = np.exp((1j) * self.cgh.phi)
        self._update()
