# -*- coding: utf-8 -*-
# MENU: Optical vortex

from .task import task
from pyfablib.traps import QVortexTrap
from pyqtgraph.Qt import QtGui


class vortextrap(task):
    """Add an optical vortex to the trapping pattern"""

    def __init__(self, **kwargs):
        super(vortextrap, self).__init__(**kwargs)

    def dotask(self):
        trap = QVortexTrap(r=QtGui.QVector3D(100, 100, 0))
        self.parent.pattern.addTrap(trap)
