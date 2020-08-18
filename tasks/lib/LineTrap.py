# -*- coding: utf-8 -*-
# MENU: Add trap/Line trap

from ..QTask import QTask
from pyfablib.traps import QLineTrap
from PyQt5.QtGui import QVector3D


class LineTrap(QTask):
    """Add a line trap to the trapping pattern"""

    def __init__(self, **kwargs):
        super(LineTrap, self).__init__(**kwargs)

    def complete(self):
        cgh = self.parent().cgh.device
        pos = QVector3D(cgh.xc, cgh.yc, 0.)
        trap = QLineTrap(r=pos)
        self.parent().pattern.addTrap(trap)
