# -*- coding: utf-8 -*-
# MENU: Add trap/Line trap

from .Task import Task
from pyfablib.traps import QLineTrap
from PyQt5.QtGui import QVector3D


class LineTrap(Task):
    """Add a line trap to the trapping pattern"""

    def __init__(self, **kwargs):
        super(LineTrap, self).__init__(**kwargs)

    def dotask(self):
        trap = QLineTrap(r=QVector3D(100, 100, 0))
        self.parent.pattern.addTrap(trap)
