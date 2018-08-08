# -*- coding: utf-8 -*-
# MENU: Line trap

from .task import task
from pyfablib.traps import QLineTrap
from pyqtgraph.Qt import QtGui


class linetrap(task):
    """Add a line trap to the trapping pattern"""

    def __init__(self, **kwargs):
        super(linetrap, self).__init__(**kwargs)

    def dotask(self):
        trap = QLineTrap(r=QtGui.QVector3D(100, 100, 0))
        self.parent.pattern.addTrap(trap)
