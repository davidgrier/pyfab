# -*- coding: utf-8 -*-
# MENU: Add trap/Bessel IPH trap

from .Task import Task
from pyfablib.traps.QBesselIPHTrap import QBesselIPHTrap
from PyQt5.QtGui import QVector3D


class BesselIPH(Task):
    """Add a bessel trap to the trapping pattern using
    intermediate plane holography"""

    def __init__(self, **kwargs):
        super(BesselIPH, self).__init__(**kwargs)

    def dotask(self):
        xc = self.parent.cgh.xc
        yc = self.parent.cgh.yc
        zc = self.parent.cgh.zc
        trap = QBesselIPHTrap(r=QVector3D(xc, yc, zc))
        self.parent.pattern.addTrap(trap)
