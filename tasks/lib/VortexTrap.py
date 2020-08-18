# -*- coding: utf-8 -*-
# MENU: Add trap/Optical vortex

from ..QTask import QTask
from pyfablib.traps import QVortexTrap
from PyQt5.QtGui import QVector3D


class VortexTrap(QTask):
    """Add an optical vortex to the trapping pattern"""

    def __init__(self, center3=None, **kwargs):
        super(VortexTrap, self).__init__(**kwargs)
        self.center3 = center3 or (self.parent().cgh.device.xc,
                           self.parent().cgh.device.yc,
                           0)

    def complete(self):
        (xc, yc, zc) = self.center3
        trap = QVortexTrap(r=QVector3D(xc, yc, zc))
        self.parent().pattern.addTrap(trap)