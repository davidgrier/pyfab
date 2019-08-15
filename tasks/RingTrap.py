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
        trap = QRingTrap(r=QVector3D(970, 682, 0))
        self.parent.pattern.addTrap(trap)
