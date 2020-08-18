# -*- coding: utf-8 -*-
# MENU: Add trap/Bessel IPH trap

from ..QTask import QTask
from pyfablib.traps.QBesselIPHTrap import QBesselIPHTrap
from PyQt5.QtGui import QVector3D


class BesselIPH(QTask):
    """Add a bessel trap to the trapping pattern using
    intermediate plane holography"""

    def __init__(self, center3=None, **kwargs):
        super(BesselIPH, self).__init__(**kwargs)
        self.center3 = center3 or (self.parent().cgh.device.xc,
                                   self.parent().cgh.device.yc,
                                   self.parent().cgh.device.zc)

    def complete(self):
        (xc, yc, zc) = self.center3
        trap = QBesselIPHTrap(r=QVector3D(xc, yc, zc))
        self.parent().pattern.addTrap(trap)
