# -*- coding: utf-8 -*-
# MENU: Add trap/Ring trap

from ..QTask import QTask
from pyfablib.traps import QRingTrap
from PyQt5.QtGui import QVector3D


class RingTrap(QTask):
    """Add a ring trap to the trapping pattern"""

    def __init__(self, **kwargs):
        super(RingTrap, self).__init__(**kwargs)

    def complete(self):
        cgh = self.parent().cgh.device
        trap = QRingTrap(r=QVector3D(cgh.xc, cgh.yc, 0))
        self.parent().pattern.addTrap(trap)
