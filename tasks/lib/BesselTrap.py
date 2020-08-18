# -*- coding: utf-8 -*-
# MENU: Add trap/Bessel beam

from ..QTask import QTask
from pyfablib.traps.QBesselTrap import QBesselTrap
from PyQt5.QtGui import QVector3D


class BesselTrap(QTask):
    """Add a Bessel beam to the trapping pattern"""

    def __init__(self, center3=None, **kwargs):
        super(BesselTrap, self).__init__(**kwargs)
        self.center3 = center3 or (self.parent().cgh.device.xc,
                                   self.parent().cgh.device.yc,
                                   0)

    def complete(self):
        (xc, yc, zc) = self.center3
        trap = QBesselTrap(r=QVector3D(xc, yc, zc))
        self.parent().pattern.addTrap(trap)
