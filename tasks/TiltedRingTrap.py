# -*- coding: utf-8 -*-
# MENU: Add trap/Tilted ring trap

from .Task import Task
from pyfablib.traps import QTiltedRingTrap
from PyQt5.QtGui import QVector3D


class TiltedRingTrap(Task):
    """Add a tilted ring trap to the trapping pattern"""

    def __init__(self, **kwargs):
        super(TiltedRingTrap, self).__init__(**kwargs)

    def dotask(self):
        trap = QTiltedRingTrap(r=QVector3D(100, 100, 0))
        self.parent.pattern.addTrap(trap)
