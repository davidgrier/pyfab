# -*- coding: utf-8 -*-
# MENU: Add trap/Ultra trap

from .Task import Task
from pyfablib.traps.QUltraTrap import QUltraTrap
from pyqtgraph.Qt import QtGui


class UltraTrap(Task):
    """Add two combined single traps (an ultra trap) to the trapping pattern"""

    def __init__(self, **kwargs):
        super(UltraTrap, self).__init__(**kwargs)

    def dotask(self):
        trap = QUltraTrap(r=QtGui.QVector3D(800, 500, 0))
        # setting the initial position of the trap
        self.parent.pattern.addTrap(trap)
