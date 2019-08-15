# -*- coding: utf-8 -*-
# MENU: Add trap/Bessel trap

from .Task import Task
from pyfablib.traps.QBesselTrap import QBesselTrap
from PyQt5.QtGui import QVector3D


class BesselTrap(Task):
    """Add a Bessel trapping pattern"""

    def __init__(self, **kwargs):
        super(BesselTrap, self).__init__(**kwargs)

    def dotask(self):
        trap = QBesselTrap(r=QVector3D(100, 100, 0))
        self.parent.pattern.addTrap(trap)
