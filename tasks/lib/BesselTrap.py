# -*- coding: utf-8 -*-
# MENU: Add trap/Bessel trap

from ..QTask import QTask
from pyfablib.traps.QBesselTrap import QBesselTrap
from PyQt5.QtGui import QVector3D


class BesselTrap(QTask):
    """Add a Bessel trapping pattern"""

    def __init__(self, **kwargs):
        super(BesselTrap, self).__init__(**kwargs)

    def complete(self):
        cgh = self.parent().cgh.device
        pos = QVector3D(cgh.xc, cgh.yc, 0)
        trap = QBesselTrap(r=pos)
        self.parent().pattern.addTrap(trap)
