# -*- coding: utf-8 -*-
# MENU: Add trap/Optical vortex

from ..QTask import QTask
from pyfablib.traps import QVortexTrap
from pyqtgraph.Qt import QtGui


class VortexTrap(QTask):
    """Add an optical vortex to the trapping pattern"""

    def __init__(self, **kwargs):
        super(VortexTrap, self).__init__(**kwargs)

    def complete(self):
        trap = QVortexTrap(r=QtGui.QVector3D(100, 100, 0))
        self.parent().pattern.addTrap(trap)
