# -*- coding: utf-8 -*-
# MENU: Add trap/Ring trap

from .Task import Task
from pyfablib.traps import QRingTrap
from PyQt5.QtGui import QVector3D


class RingTrap(Task):
    """Add a ring trap to the trapping pattern"""

    def __init__(self, **kwargs):
        super(RingTrap, self).__init__(**kwargs)

    def dotask(self):
        cgh = self.parent.cgh.device
        trap = QRingTrap(r=QVector3D(cgh.xc, cgh.yc, 0))
        self.parent.pattern.addTrap(trap)
