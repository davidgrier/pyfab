# -*- coding: utf-8 -*-
# MENU: Bessel IPH trap

from .task import task
from pyfablib.traps.QBesselIPHTrap import QBesselIPHTrap
from pyqtgraph.Qt import QtGui


class bessel_iph(task):
    """Add a bessel trap to the trapping pattern using
    intermediate plane holography"""

    def __init__(self, **kwargs):
        super(bessel_iph, self).__init__(**kwargs)

    def dotask(self):
        trap = QBesselIPHTrap(r=QtGui.QVector3D(100, 100, 0))
        self.parent.pattern.addTrap(trap)
