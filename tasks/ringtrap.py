# -*- coding: utf-8 -*-
# MENU: Ring trap

from .task import task
from pyfablib.traps.QRingTrap import QRingTrap
from pyqtgraph.Qt import QtGui


class ringtrap(task):
    """Add a ring trap to the trapping pattern"""

    def __init__(self, **kwargs):
        super(ringtrap, self).__init__(**kwargs)

    def dotask(self):
        trap = QRingTrap(r=QtGui.QVector3D(100, 100, 0))
        self.parent.pattern.addTrap(trap)
