# -*- coding: utf-8 -*-
# MENU: Add trap/Tilted ring trap

from ..QTask import QTask
from pyfablib.traps.QTiltedRingTrap import QTiltedRingTrap
from PyQt5.QtGui import QVector3D


class TiltedRingTrap(QTask):
    """Add a tilted ring trap to the trapping pattern"""

    def __init__(self, center3=None, rho=5., m=1, s=1, **kwargs):
        super(TiltedRingTrap, self).__init__(**kwargs)
        self.center3 = center3 or (self.parent().cgh.device.xc,
                                   self.parent().cgh.device.yc,
                                   0)
        self.rho = rho
        self.m = m
        self.s = s

    def complete(self):
        (xc, yc, zc) = self.center3
        trap = QTiltedRingTrap(rho=self.rho, m=self.m, s=self.s,
                               r=QVector3D(xc, yc, zc))
        self.parent().pattern.addTrap(trap)

