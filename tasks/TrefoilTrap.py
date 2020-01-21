# -*- coding: utf-8 -*-
# MENU: Add trap/Trefoil knot trap

from .Task import Task
from pyfablib.traps import QTrefoilTrap
from PyQt5.QtGui import QVector3D


class TrefoilTrap(Task):
    """Add a trefoil knot trap to the trapping pattern"""

    def __init__(self, **kwargs):
        super(TrefoilTrap, self).__init__(**kwargs)

    def dotask(self):
        trap = QTrefoilTrap(s=.01, m=0, rho=20., r=QVector3D(100, 100, 0))
        self.parent.pattern.addTrap(trap)
