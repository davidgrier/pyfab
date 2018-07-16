
# -*- coding: utf-8 -*-
# MENU: Bessel Trapping

from .task import task
from pyfablib.traps import QBesselTrap
from pyqtgraph.Qt import QtGui


class besseltrap(task):
    """Add a Bessel trapping pattern"""

    def __init__(self, **kwargs):
        super(besseltrap, self).__init__(**kwargs)

    def dotask(self):
        trap = QBesselTrap(r=QtGui.QVector3D(100, 100, 0))
        self.parent.pattern.addTrap(trap)

