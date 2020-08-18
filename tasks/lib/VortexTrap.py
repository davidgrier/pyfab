# -*- coding: utf-8 -*-
# MENU: Add trap/Optical vortex

from ..QTask import QTask
from pyfablib.traps import QVortexTrap
from PyQt5.QtGui import QVector3D


class VortexTrap(QTask):
    """Add an optical vortex to the trapping pattern"""

    def __init__(self, **kwargs):
        super(VortexTrap, self).__init__(**kwargs)

    def complete(self):
        cgh = self.parent().cgh.device
        pos = QVector3D(cgh.xc, cgh.yc, 0.)
        trap = QVortexTrap(r=pos)
        self.parent().pattern.addTrap(trap)
