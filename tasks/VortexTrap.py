# -*- coding: utf-8 -*-
# MENU: Add trap/Optical vortex

from .Task import Task
from pyfablib.traps import QVortexTrap
from pyqtgraph.Qt import QtGui


class VortexTrap(Task):
    """Add an optical vortex to the trapping pattern"""

    def __init__(self, **kwargs):
        super(VortexTrap, self).__init__(**kwargs)

    def dotask(self):
        trap = QVortexTrap(r=QtGui.QVector3D(100, 100, 0))
        self.parent.pattern.addTrap(trap)
