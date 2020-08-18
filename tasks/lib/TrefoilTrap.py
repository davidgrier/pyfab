# -*- coding: utf-8 -*-
# MENU: Add trap/Trefoil knot trap

from ..QTask import QTask
from pyfablib.traps.QTrefoilTrap import QTrefoilTrap
from PyQt5.QtGui import QVector3D


class TrefoilTrap(QTask):
    """Add a trefoil knot trap to the trapping pattern"""

    def __init__(self, center3=None, rho=20., m=0, s=0.01, **kwargs):
        super(TrefoilTrap, self).__init__(**kwargs)
        self.center3 = center3 or (self.parent().cgh.device.xc,
                                   self.parent().cgh.device.yc,
                                   0)
        self.rho = rho
        self.m = m
        self.s = s

    def complete(self):
        (xc, yc, zc) = self.center3
        trap = QTrefoilTrap(rho=self.rho, m=self.m, s=self.s,
                               r=QVector3D(xc, yc, zc))
        self.parent().pattern.addTrap(trap)