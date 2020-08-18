# -*- coding: utf-8 -*-
# MENU: Add trap/Ring trap

from ..QTask import QTask
from pyfablib.traps.QRingTrap import QRingTrap
from PyQt5.QtGui import QVector3D


class RingTrap(QTask):
    """Add a ring trap to the trapping pattern"""

    def __init__(self, center3=None, **kwargs):
        super(RingTrap, self).__init__(**kwargs)
        self.center3 = center3 or (self.parent().cgh.device.xc,
                                   self.parent().cgh.device.yc,
                                   0)

    def complete(self):
        (xc, yc, zc) = self.center3
        trap = QRingTrap(r=QVector3D(xc, yc, zc))
        self.parent().pattern.addTrap(trap)
