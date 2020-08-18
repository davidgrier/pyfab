# -*- coding: utf-8 -*-
# MENU: Add trap/Ultra trap

from ..QTask import QTask
from pyfablib.traps.QUltraTrap import QUltraTrap
from PyQt5.QtGui import QVector3D


class UltraTrap(QTask):
    """Add two combined single traps (an ultra trap) to the trapping pattern"""

    def __init__(self, center3=(880.75, 814.45, 0), **kwargs):
        super(UltraTrap, self).__init__(**kwargs)
        self.center3 = center3

    def dotask(self):
        (xc, yc, zc) = self.center3
        trap = QUltraTrap(r=QVector3D(xc, yc, zc))
        self.parent().pattern.addTrap(trap)