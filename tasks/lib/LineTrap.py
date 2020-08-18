# -*- coding: utf-8 -*-
# MENU: Add trap/Line trap

from ..QTask import QTask
from pyfablib.traps.QLineTrap import QLineTrap
from PyQt5.QtGui import QVector3D


class LineTrap(QTask):
    """Add a line trap to the trapping pattern"""

    def __init__(self, center3=None, **kwargs):
        super(LineTrap, self).__init__(**kwargs)
        self.center3 = center3 or (self.parent().cgh.device.xc,
                                   self.parent().cgh.device.yc,
                                   0)

    def complete(self):
        (xc, yc, zc) = self.center3
        trap = QLineTrap(r=QVector3D(xc, yc, zc))
        self.parent().pattern.addTrap(trap)